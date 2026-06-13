import base64
import math
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas import (
    FrameInferenceResponse,
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
) -> dict[str, Any] | None:
    """Choose one stable track mask for a fixed judge camera.

    The mask is selected from the first `reference_frames` sampled frames. We prefer
    frames with fewer dynamic objects (wheels/cars) and then the largest detected
    track polygon. This usually picks the least occluded frame; if the video has no
    empty frame, it still chooses the best available reference.
    """
    best_payload: dict[str, Any] | None = None
    best_dynamic_count: int | None = None
    best_area = -1.0

    for frame_index, frame_bgr in sampled_frames[:reference_frames]:
        track_payloads = track_service.build_track_mask_payload(frame_bgr, frame_index)
        track_payload = _select_track_payload(track_payloads)
        if not track_payload:
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

        area = _polygon_area(track_payload.get("points", []))
        is_better = (
            best_payload is None
            or dynamic_count < (best_dynamic_count if best_dynamic_count is not None else 10**9)
            or (dynamic_count == best_dynamic_count and area > best_area)
        )

        if is_better:
            best_payload = _make_static_track_payload(track_payload, frame_index, dynamic_count)
            best_dynamic_count = dynamic_count
            best_area = area

    return best_payload


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


def _dump_response(response: ViolationAnalysisResponse) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


def _build_violation_response(
    frame_bgr: np.ndarray,
    frame_index: int,
    offtrack_threshold: float,
    hard_violation_threshold: float,
    track_payload_override: dict[str, Any] | None = None,
) -> ViolationAnalysisResponse:
    if hard_violation_threshold < offtrack_threshold:
        raise HTTPException(
            status_code=422,
            detail=(
                "hard_violation_threshold must be greater than or equal "
                "to offtrack_threshold"
            ),
        )

    detector = OfftrackDetector(
        offtrack_threshold=offtrack_threshold,
        hard_violation_threshold=hard_violation_threshold,
    )

    all_masks: list[dict[str, Any]] = []
    try:
        if track_payload_override is None:
            track_payloads = track_service.build_track_mask_payload(frame_bgr, frame_index)
            track_payload = _select_track_payload(track_payloads)
        else:
            track_payloads = [_clone_payload(track_payload_override)]
            track_payload = track_payloads[0]

        wheel_payloads = wheel_service.build_wheel_mask_payload(frame_bgr, frame_index)
        car_payloads = car_service.build_car_mask_payload(frame_bgr, frame_index)
        all_masks = [
            *track_payloads,
            *wheel_payloads,
            *car_payloads,
        ]

        if track_payload and wheel_payloads:
            analysis = detector.analyze(
                frame_bgr=frame_bgr,
                track_points=track_payload["points"],
                wheel_masks=wheel_payloads,
            )

            violation_regions = _mask_to_violation_regions(
                violation_mask=analysis.violation_mask,
                violation_score=analysis.violation_score,
            )
            return ViolationAnalysisResponse(
                frame_index=frame_index,
                frame_width=int(frame_bgr.shape[1]),
                frame_height=int(frame_bgr.shape[0]),
                frame_data_url=encode_bgr_to_data_url(frame_bgr),
                annotated_frame_data_url=encode_bgr_to_data_url(analysis.annotated_frame_bgr),
                track_mask_data_url=encode_mask_to_data_url(analysis.track_mask),
                violation_mask_data_url=encode_mask_to_data_url(analysis.violation_mask),
                violation_detected=analysis.violation_detected,
                violation_score=analysis.violation_score,
                reason=analysis.reason,
                offtrack_wheels=analysis.offtrack_wheels,
                violation_regions=violation_regions,
                masks=all_masks,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model inference failed: {type(exc).__name__}: {exc}",
        ) from exc

    return ViolationAnalysisResponse(
        frame_index=frame_index,
        frame_width=int(frame_bgr.shape[1]),
        frame_height=int(frame_bgr.shape[0]),
        frame_data_url=encode_bgr_to_data_url(frame_bgr),
        annotated_frame_data_url="",
        track_mask_data_url="",
        violation_mask_data_url="",
        violation_detected=False,
        violation_score=0.0,
        reason="track or wheel masks were not found",
        offtrack_wheels=None,
        violation_regions=None,
        masks=all_masks,
    )


def _sample_video_frames(
    capture: cv2.VideoCapture,
    source_fps: float | None,
    sample_fps: float,
    max_frames: int,
) -> tuple[list[SampledFrame], bool]:
    target_interval = 1.0 / float(sample_fps)
    next_sample_time = 0.0
    raw_frame_index = 0
    sampled_index = 0
    sampled_frames: list[SampledFrame] = []
    truncated = False

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break

        if source_fps:
            current_time = raw_frame_index / source_fps
        else:
            current_time = sampled_index * target_interval

        if current_time + 1e-9 >= next_sample_time:
            _validate_frame_size(frame_bgr)
            sampled_frames.append((sampled_index, frame_bgr.copy()))
            sampled_index += 1
            next_sample_time += target_interval

            if len(sampled_frames) >= max_frames:
                truncated = True
                break

        raw_frame_index += 1

    return sampled_frames, truncated


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
    hard_violation_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.25,
) -> ViolationAnalysisResponse:
    frame_bgr = _decode_uploaded_frame(frame)
    return _build_violation_response(
        frame_bgr=frame_bgr,
        frame_index=frame_index,
        offtrack_threshold=offtrack_threshold,
        hard_violation_threshold=hard_violation_threshold,
    )


@router.post("/infer/video", response_model=VideoInferenceResponse)
async def infer_video(
    video: UploadFile = File(...),
    sample_fps: Annotated[float, Form(ge=0.1, le=30.0)] = 2.0,
    max_frames: Annotated[int, Form(ge=1, le=500)] = 120,
    offtrack_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.12,
    hard_violation_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = 0.25,
    static_track_mode: Annotated[bool, Form()] = True,
    track_reference_frames: Annotated[int, Form(ge=1, le=120)] = 30,
) -> VideoInferenceResponse:
    """Decode video on backend and optionally reuse a static track mask.

    For fixed judge cameras the road/track geometry should be stable. In static
    mode the backend chooses one least-occluded reference frame and applies its
    track mask to all sampled frames. This avoids frame-to-frame track mask jumps
    caused by cars covering parts of the road.
    """
    video_bytes = _read_upload_bytes(video, MAX_VIDEO_UPLOAD_BYTES)
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(video_bytes)
            tmp_path = tmp_file.name

        capture = cv2.VideoCapture(tmp_path)
        if not capture.isOpened():
            raise HTTPException(
                status_code=400,
                detail=(
                    "OpenCV could not open the uploaded video. "
                    "Try converting it to MP4/H.264 if the codec is unsupported."
                ),
            )

        source_fps_raw = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        source_fps = source_fps_raw if math.isfinite(source_fps_raw) and source_fps_raw > 0 else None
        frame_count_raw = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_frames_estimate = frame_count_raw if frame_count_raw > 0 else None

        sampled_frames, truncated = _sample_video_frames(
            capture=capture,
            source_fps=source_fps,
            sample_fps=float(sample_fps),
            max_frames=max_frames,
        )
        capture.release()

        if not sampled_frames:
            raise HTTPException(
                status_code=400,
                detail="No frames were decoded from the uploaded video",
            )

        static_track_payload = None
        static_track_reason = "disabled"
        if static_track_mode:
            static_track_payload = _select_static_track_payload(
                sampled_frames=sampled_frames,
                reference_frames=min(track_reference_frames, len(sampled_frames)),
            )
            if static_track_payload:
                static_track_reason = (
                    "fixed track mask selected from frame "
                    f"{static_track_payload.get('source_frame_index')} "
                    f"with {static_track_payload.get('reference_dynamic_masks')} dynamic masks"
                )
            else:
                static_track_reason = "static mode requested, but no track mask was found in reference frames"

        results: list[ViolationAnalysisResponse] = []
        for frame_index, frame_bgr in sampled_frames:
            response = _build_violation_response(
                frame_bgr=frame_bgr,
                frame_index=frame_index,
                offtrack_threshold=offtrack_threshold,
                hard_violation_threshold=hard_violation_threshold,
                track_payload_override=static_track_payload,
            )
            results.append(response)

        return VideoInferenceResponse(
            filename=video.filename or "video",
            source_fps=source_fps,
            sample_fps=float(sample_fps),
            frames_processed=len(results),
            total_frames_estimate=total_frames_estimate,
            truncated=truncated,
            static_track_enabled=bool(static_track_payload),
            static_track_reference_frame=(
                int(static_track_payload["source_frame_index"])
                if static_track_payload and "source_frame_index" in static_track_payload
                else None
            ),
            static_track_reason=static_track_reason,
            results=results,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
