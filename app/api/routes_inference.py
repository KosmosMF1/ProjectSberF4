import base64
import math
import os
import shutil
import tempfile
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Path as ApiPath, UploadFile
from fastapi.responses import FileResponse

from app.schemas import (
    FrameInferenceResponse,
    OfftrackAnalysisResult,
    VideoInferenceResponse,
    ViolationAnalysisResponse,
    ViolationRegion,
    YoloMask,
)
from app.services import (
    CarSegmentationService,
    OfftrackDetector,
    TrackSegmentationService,
    WheelSegmentationService,
    encode_bgr_to_data_url,
    encode_mask_to_data_url,
)

from app.services.camera_calibration import (
    CAMERA_ID_PATTERN, CameraCalibration, load_calibration, outline_from_mask, save_calibration,
)
from app.services.car_tracking import CarTracker, unique_masks
from app.services.incidents import INCIDENT_DIR, IncidentDetector, evidence_frame, write_incident
from app.services.track_consensus import consensus_track_polygon, reference_frame_indices
from app.services.track_stabilization import TrackStabilizer

router = APIRouter()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_FRAME_UPLOAD_MB", "25")) * 1024 * 1024
MAX_VIDEO_UPLOAD_BYTES = int(os.getenv("MAX_VIDEO_UPLOAD_MB", "512")) * 1024 * 1024
MAX_FRAME_PIXELS = 25_000_000

track_service = TrackSegmentationService()
wheel_service = WheelSegmentationService()
car_service = CarSegmentationService()

SampledFrame = tuple[int, np.ndarray]


def _encode_frame_to_data_url(frame_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _read_upload_bytes(upload: UploadFile, max_bytes: int) -> bytes:
    payload = upload.file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(payload) > max_bytes:
        max_mb = max_bytes / 1024 / 1024
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file is too large. Maximum is {max_mb:.0f} MB.",
        )
    return payload


def _decode_uploaded_frame(frame: UploadFile) -> np.ndarray:
    frame_bytes = _read_upload_bytes(frame, MAX_UPLOAD_BYTES)

    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode uploaded data as image frame",
        )

    _validate_frame_size(frame_bgr)
    return frame_bgr


def _validate_frame_size(frame_bgr: np.ndarray) -> None:
    frame_pixels = int(frame_bgr.shape[0]) * int(frame_bgr.shape[1])
    if frame_pixels > MAX_FRAME_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=(
                "Decoded frame is too large. "
                f"Maximum is {MAX_FRAME_PIXELS} pixels, got {frame_pixels}."
            ),
        )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_point(x: int, y: int, width: int, height: int) -> list[float]:
    return [
        round(_clip01(x / float(width)), 6),
        round(_clip01(y / float(height)), 6),
    ]


def _to_yolo_segmentation(
    class_id: int,
    points: list[list[float]],
) -> list[float]:
    segmentation = [float(class_id)]
    for x, y in points:
        segmentation.extend([x, y])
    return segmentation


def _polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) * 0.5


def _select_track_payload(track_payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not track_payloads:
        return None

    return max(
        track_payloads,
        key=lambda payload: _polygon_area(payload.get("points", [])),
    )


def _clone_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(payload)
    if "points" in cloned:
        cloned["points"] = [list(point) for point in cloned["points"]]
    if "bbox_xyxy" in cloned and cloned["bbox_xyxy"] is not None:
        cloned["bbox_xyxy"] = list(cloned["bbox_xyxy"])
    if "yolo_segmentation" in cloned and cloned["yolo_segmentation"] is not None:
        cloned["yolo_segmentation"] = list(cloned["yolo_segmentation"])
    return cloned


def _make_static_track_payload(payload: dict[str, Any], frame_index: int, dynamic_count: int) -> dict[str, Any]:
    static_payload = _clone_payload(payload)
    static_payload["model_name"] = "track_seg_yolo_static_reference"
    static_payload["class_id"] = 0
    static_payload["class_name"] = static_payload.get("class_name") or "track"
    static_payload["source_frame_index"] = frame_index
    static_payload["reference_dynamic_masks"] = dynamic_count
    static_payload["reference_area"] = round(_polygon_area(static_payload.get("points", [])), 6)
    return static_payload


def _select_static_track_payload(
    sampled_frames: list[SampledFrame],
    reference_frames: int,
) -> tuple[dict[str, Any] | None, TrackStabilizer | None]:
    """Choose an anchor with few obstructions and combine compatible masks."""
    candidates: list[dict[str, Any]] = []

    for frame_index, frame_bgr in sampled_frames[:reference_frames]:
        track_payloads = track_service.build_track_mask_payload(frame_bgr, frame_index)
        track_payload = _select_track_payload(track_payloads)
        if not track_payload or not _polygon_area(track_payload["points"]):
            continue

        dynamic_count = 0
        try:
            dynamic_count += len(wheel_service.build_wheel_mask_payload(frame_bgr, frame_index))
        except Exception:
            # Track calibration should not fail only because a secondary model
            # could not detect wheels in the reference frame.
            pass

        try:
            dynamic_count += len(car_service.build_car_mask_payload(frame_bgr, frame_index))
        except Exception:
            # Same idea for the optional car model.
            pass

        candidates.append(_make_static_track_payload(track_payload, frame_index, dynamic_count))

    if not candidates:
        return None, None
    candidates.sort(key=lambda mask: (mask["reference_dynamic_masks"], -mask["reference_area"]))
    anchor = candidates[0]
    frames = dict(sampled_frames)
    stabilizer = TrackStabilizer(frames[anchor["source_frame_index"]], anchor["points"])
    alternatives = []
    for candidate in candidates[1:]:
        if not stabilizer.available:
            alternatives.append(candidate["points"])
            continue
        transform = stabilizer.estimate(frames[candidate["source_frame_index"]])
        if transform is not None:
            inverse = cv2.invertAffineTransform(transform)
            alternatives.append(stabilizer.transform(candidate["points"], inverse))
    consensus = consensus_track_polygon(anchor["points"], alternatives)
    if consensus is not None:
        points, count = consensus
        anchor.update(
            model_name="track_seg_yolo_static_consensus", points=points,
            yolo_segmentation=_to_yolo_segmentation(0, points), reference_samples=count,
            reference_area=round(_polygon_area(points), 6),
        )
    return anchor, stabilizer


def _mask_to_violation_regions(
    violation_mask: np.ndarray,
    violation_score: float,
) -> list[ViolationRegion]:
    height, width = violation_mask.shape[:2]
    mask_u8 = violation_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    regions: list[ViolationRegion] = []
    min_area = max(1.0, width * height * 0.000001)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.0, 0.01 * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            x2 = x + box_width - 1
            y2 = y + box_height - 1
            points_px = [(x, y), (x2, y), (x2, y2), (x, y2)]
        else:
            points_px = [
                (int(point[0]), int(point[1]))
                for point in approx.reshape(-1, 2)
            ]

        points = [
            _normalize_point(
                min(max(x, 0), width - 1),
                min(max(y, 0), height - 1),
                width,
                height,
            )
            for x, y in points_px
        ]

        x, y, box_width, box_height = cv2.boundingRect(contour)
        x2 = min(width - 1, x + box_width - 1)
        y2 = min(height - 1, y + box_height - 1)

        regions.append(
            ViolationRegion(
                model_name="offtrack_detector",
                class_id=2,
                class_name="track_limit_violation",
                confidence=round(_clip01(float(violation_score)), 6),
                bbox_xyxy=[
                    round(x / float(width), 6),
                    round(y / float(height), 6),
                    round(x2 / float(width), 6),
                    round(y2 / float(height), 6),
                ],
                points=points,
                yolo_segmentation=_to_yolo_segmentation(2, points),
            )
        )

    return regions


def _validate_thresholds(offtrack_threshold: float, hard_violation_threshold: float) -> None:
    if hard_violation_threshold < offtrack_threshold:
        raise HTTPException(
            status_code=422,
            detail="hard_violation_threshold must be greater than or equal to offtrack_threshold",
        )


def _analyze_frame(
    frame: np.ndarray,
    index: int,
    detector: OfftrackDetector,
    track_override: dict[str, Any] | None = None,
    cached_masks: tuple[np.ndarray, np.ndarray] | None = None,
    tracker: CarTracker | None = None,
    timestamp: float = 0.0,
    annotate: bool = True,
) -> tuple[OfftrackAnalysisResult, list[dict[str, Any]], list[dict[str, Any]], list[list[float]]]:
    try:
        track = track_override or _select_track_payload(track_service.build_track_mask_payload(frame, index))
        wheels = unique_masks(
            [mask for mask in wheel_service.build_wheel_mask_payload(frame, index)
             if mask["model_name"] != "wheel_segmentation_stub"],
            iou=0.45, containment=0.75,
        )
        cars = unique_masks(car_service.build_car_mask_payload(frame, index), iou=0.32, containment=0.8)
        if tracker is not None:
            cars = tracker.update(cars, timestamp)
        points = track["points"] if track else []
        analysis = detector.analyze(frame, points, wheels, cars, track_masks=cached_masks, annotate=annotate)
        return analysis, ([track] if track else []) + wheels + cars, wheels, points
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model inference failed: {type(exc).__name__}: {exc}",
        ) from exc


def _response(
    frame: np.ndarray, index: int, analysis: OfftrackAnalysisResult,
    masks: list[dict[str, Any]], source_index: int | None = None,
    timestamp: float | None = None, compact: bool = False,
) -> ViolationAnalysisResponse:
    if compact and frame.shape[1] > 960:
        height = round(frame.shape[0] * 960 / frame.shape[1])
        frame = cv2.resize(frame, (960, max(1, height)), interpolation=cv2.INTER_AREA)
    return ViolationAnalysisResponse(
        frame_index=index,
        source_frame_index=source_index,
        timestamp=timestamp,
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
        frame_data_url=encode_bgr_to_data_url(frame),
        annotated_frame_data_url=(
            encode_bgr_to_data_url(analysis.annotated_frame_bgr)
            if analysis.annotated_frame_bgr is not None else ""
        ),
        track_mask_data_url=encode_mask_to_data_url(analysis.track_mask) if analysis.track_mask is not None else "",
        violation_mask_data_url=(
            encode_mask_to_data_url(analysis.violation_mask) if analysis.violation_mask is not None else ""
        ),
        violation_detected=analysis.violation_detected,
        violation_score=analysis.violation_score,
        reason=analysis.reason,
        offtrack_wheels=analysis.offtrack_wheels,
        violation_regions=(
            _mask_to_violation_regions(analysis.violation_mask, analysis.violation_score)
            if analysis.violation_detected and analysis.violation_mask is not None else []
        ),
        masks=masks,
        cars=analysis.cars,
    )


def _store_video_upload(video: UploadFile, work: str) -> str:
    source = str(Path(work) / f"source{Path(video.filename or 'video').suffix}")
    total_bytes = 0
    with open(source, "wb") as output:
        while chunk := video.file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_VIDEO_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Uploaded video is too large")
            output.write(chunk)
    if not total_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return source


@router.get("/cameras/{camera_id}/calibration", response_model=CameraCalibration)
def get_camera_calibration(camera_id: Annotated[str, ApiPath(pattern=CAMERA_ID_PATTERN)]) -> CameraCalibration:
    calibration = load_calibration(camera_id)
    if calibration is None:
        raise HTTPException(status_code=404, detail="Для этой камеры ещё нет сохранённой границы")
    return calibration


@router.put("/cameras/{camera_id}/calibration", response_model=CameraCalibration)
def put_camera_calibration(
    camera_id: Annotated[str, ApiPath(pattern=CAMERA_ID_PATTERN)],
    calibration: CameraCalibration,
) -> CameraCalibration:
    save_calibration(camera_id, calibration)
    return calibration


@router.post("/calibration/outline")
def calibration_outline(mask: UploadFile = File(...)) -> dict:
    payload = _read_upload_bytes(mask, MAX_UPLOAD_BYTES)
    if len(payload) < 24 or not payload.startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"):
        raise HTTPException(status_code=400, detail="Ожидается PNG с закрашенной областью")
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    if not 0 < width * height <= MAX_FRAME_PIXELS:
        raise HTTPException(status_code=413, detail="Недопустимый размер маски")
    pixels = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or pixels.dtype != np.uint8:
        raise HTTPException(status_code=400, detail="Не удалось прочитать маску редактора")
    channel = 3 if pixels.shape[2] == 4 else 1
    binary = (pixels[:, :, channel] > 127).astype(np.uint8)
    try:
        points = outline_from_mask(binary)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"points": points}


@router.post("/calibration/preview")
def calibration_preview(
    video: UploadFile = File(...),
    time_seconds: Annotated[float, Form(ge=0, le=86400, allow_inf_nan=False)] = 0.0,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="track-preview-") as work:
        source = _store_video_upload(video, work)
        capture = cv2.VideoCapture(source)
        try:
            if not capture.isOpened():
                raise HTTPException(status_code=400, detail="Не удалось открыть видео")
            rate = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            fps = rate if math.isfinite(rate) and rate > 0 else 30.0
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            index = int(time_seconds * fps + 1e-6)
            if frame_count > 0 and index >= frame_count:
                raise HTTPException(status_code=422, detail="Указанное время за пределами видео")
            if index and not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise HTTPException(status_code=422, detail="Указанное время за пределами видео")
            ok, frame = capture.read()
            if not ok:
                raise HTTPException(status_code=422, detail="Кадр недоступен, выберите другое время")
            _validate_frame_size(frame)
            height, width = frame.shape[:2]
            if width > 1280:
                frame = cv2.resize(frame, (1280, max(1, round(height * 1280 / width))), interpolation=cv2.INTER_AREA)
            return {
                "frame_width": width,
                "frame_height": height,
                "frame_data_url": _encode_frame_to_data_url(frame),
                "timestamp": index / fps,
                "last_timestamp": (frame_count - 1) / fps if frame_count > 0 else None,
            }
        finally:
            capture.release()


def _sample_video_frames(
    capture: cv2.VideoCapture, source_fps: float, sample_fps: float,
) -> Iterator[tuple[int, float, np.ndarray]]:
    interval = 1.0 / min(source_fps, sample_fps)
    next_time = 0.0
    source_index = 0
    while capture.grab():
        timestamp = source_index / source_fps
        if timestamp + 1e-9 >= next_time:
            ok, frame = capture.retrieve()
            if not ok:
                break
            _validate_frame_size(frame)
            yield source_index, timestamp, frame
            next_time += interval
        source_index += 1


@router.post("/infer/frame", response_model=FrameInferenceResponse)
async def infer_frame(
    frame: UploadFile = File(...),
    frame_index: int = Form(0),
) -> FrameInferenceResponse:
    frame_bgr = _decode_uploaded_frame(frame)

    wheel_payloads = wheel_service.build_wheel_mask_payload(frame_bgr, frame_index)
    track_payloads = track_service.build_track_mask_payload(frame_bgr, frame_index)
    car_payloads = car_service.build_car_mask_payload(frame_bgr, frame_index)
    masks = [YoloMask(**mask) for mask in [*track_payloads, *wheel_payloads, *car_payloads]]

    return FrameInferenceResponse(
        frame_index=frame_index,
        frame_width=int(frame_bgr.shape[1]),
        frame_height=int(frame_bgr.shape[0]),
        frame_data_url=_encode_frame_to_data_url(frame_bgr),
        masks=masks,
    )


@router.post("/infer/violation", response_model=ViolationAnalysisResponse)
async def infer_violation(
    frame: UploadFile = File(...),
    frame_index: Annotated[int, Form(ge=0)] = 0,
    offtrack_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.12,
    hard_violation_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.8,
) -> ViolationAnalysisResponse:
    _validate_thresholds(offtrack_threshold, hard_violation_threshold)
    frame_bgr = _decode_uploaded_frame(frame)
    detector = OfftrackDetector(offtrack_threshold, hard_violation_threshold)
    analysis, masks, _, _ = _analyze_frame(frame_bgr, frame_index, detector)
    return _response(frame_bgr, frame_index, analysis, masks)


@router.post("/infer/video", response_model=VideoInferenceResponse)
def infer_video(
    video: UploadFile = File(...),
    sample_fps: Annotated[float, Form(ge=0.1, le=30.0)] = 2.0,
    max_frames: Annotated[int, Form(ge=1, le=500)] = 120,
    offtrack_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.12,
    hard_violation_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.8,
    static_track_mode: Annotated[bool, Form()] = True,
    track_reference_frames: Annotated[int, Form(ge=1, le=120)] = 30,
    camera_id: Annotated[str, Form(pattern=CAMERA_ID_PATTERN)] = "camera-1",
    compact: Annotated[bool, Form()] = False,
    use_camera_calibration: Annotated[bool, Form()] = False,
) -> VideoInferenceResponse:
    """Analyze a recording and export one package for each confirmed excursion."""
    _validate_thresholds(offtrack_threshold, hard_violation_threshold)
    filename = video.filename or "video"
    calibration = get_camera_calibration(camera_id) if use_camera_calibration else None
    detector = OfftrackDetector(offtrack_threshold, hard_violation_threshold)
    tracker = CarTracker()
    incidents = []
    run_id = uuid4()

    with tempfile.TemporaryDirectory(prefix="track-limits-") as work:
        source = _store_video_upload(video, work)
        capture = cv2.VideoCapture(source)
        try:
            if not capture.isOpened():
                raise HTTPException(status_code=400, detail="OpenCV could not open the uploaded video")
            rate = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            source_fps = rate if math.isfinite(rate) and rate > 0 else None
            fps = source_fps or 30.0
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            incident_detector = IncidentDetector(min(fps, sample_fps))
            static_track = calibration.track_mask() if calibration else None
            stabilizer = None
            if calibration and static_track and (reference := calibration.reference_frame()) is not None:
                stabilizer = TrackStabilizer(reference, static_track["points"])
            if static_track_mode and calibration is None:
                reference_count = min(track_reference_frames, max_frames)
                budget = min(reference_count, 8)
                candidates: list[SampledFrame] = []
                if frame_count > 0:
                    for index in reference_frame_indices(frame_count, fps, sample_fps, max_frames, budget):
                        if index and not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                            continue
                        ok, frame = capture.read()
                        if ok:
                            _validate_frame_size(frame)
                            candidates.append((index, frame))
                else:
                    frames = _sample_video_frames(capture, fps, sample_fps)
                    window = min(max_frames, max(track_reference_frames, 30))
                    stride = max(1, math.ceil(window / budget))
                    candidates = [(i, frame) for i, _, frame in islice(frames, 0, window, stride)]
                    frames.close()
                static_track, stabilizer = _select_static_track_payload(candidates, len(candidates))
                del candidates
                capture.release()
                capture = cv2.VideoCapture(source)

            results = []
            cached_masks = None
            stabilized_frames = 0
            truncated = False
            for index, (source_index, timestamp, frame) in enumerate(_sample_video_frames(capture, fps, sample_fps)):
                if index == max_frames:
                    truncated = True
                    break
                if static_track and (cached_masks is None or cached_masks[0].shape != frame.shape[:2]):
                    if calibration and not calibration.matches_frame(frame.shape[1], frame.shape[0]):
                        raise HTTPException(status_code=422, detail="Пропорции кадра отличаются от сохранённой границы камеры")
                    cached_masks = detector.track_masks(frame.shape[:2], static_track["points"])
                frame_track, frame_masks = static_track, cached_masks
                if static_track and stabilizer is not None and source_index != static_track.get("source_frame_index"):
                    base_points = static_track["points"]
                    adjusted = stabilizer.adjust(frame, base_points)
                    if adjusted is not base_points:
                        frame_track = {**static_track, "points": adjusted,
                                       "yolo_segmentation": _to_yolo_segmentation(0, adjusted)}
                        frame_masks = detector.track_masks(frame.shape[:2], adjusted)
                        stabilized_frames += 1
                analysis, masks, wheels, points = _analyze_frame(
                    frame, source_index, detector, frame_track, frame_masks, tracker, timestamp,
                    annotate=not compact,
                )
                for incident in incident_detector.update(analysis.cars, source_index, timestamp):
                    still = evidence_frame(frame, incident, camera_id, points, wheels)
                    incidents.append(write_incident(
                        Path(work), run_id, source, filename, fps, camera_id, incident, still,
                    ))
                results.append(_response(frame, index, analysis, masks, source_index, timestamp, compact))
        finally:
            capture.release()

        if not results:
            raise HTTPException(status_code=400, detail="No frames were decoded from the uploaded video")
        incident_detector.mark_gap_frames(results)
        if incidents:
            INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=INCIDENT_DIR) as staging:
                batch = Path(staging) / str(run_id)
                batch.mkdir()
                for incident in incidents:
                    package = f"{incident.id}.zip"
                    # The work directory and the configured output may be on different disks.
                    shutil.copyfile(Path(work) / package, batch / package)
                batch.rename(INCIDENT_DIR / str(run_id))

    if calibration:
        reason = f"saved track boundary for camera {camera_id}"
    elif static_track and static_track.get("reference_samples"):
        reason = f"automatic track boundary from {static_track['reference_samples']} frames"
    elif static_track:
        reason = f"fixed track mask selected from source frame {static_track['source_frame_index']}"
    else:
        reason = "no reference track mask found" if static_track_mode else "disabled"
    return VideoInferenceResponse(
        filename=filename, source_fps=source_fps, sample_fps=sample_fps, frames_processed=len(results),
        total_frames_estimate=frame_count if frame_count > 0 else None, truncated=truncated,
        static_track_enabled=bool(static_track),
        static_track_reference_frame=static_track.get("source_frame_index") if static_track else None,
        static_track_reference_samples=static_track.get("reference_samples") if static_track else None,
        stabilized_frames=stabilized_frames,
        static_track_reason=reason, calibration_camera_id=camera_id if calibration else None,
        results=results, incidents=incidents,
    )


@router.get("/incidents/{run_id}/{incident_id}")
def download_incident(run_id: UUID, incident_id: UUID) -> FileResponse:
    package = INCIDENT_DIR / str(run_id) / f"{incident_id}.zip"
    if not package.is_file():
        raise HTTPException(status_code=404, detail="Incident package was not found")
    return FileResponse(package, media_type="application/zip", filename=package.name)
