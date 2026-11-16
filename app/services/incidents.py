import json
import math
import os
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4
from zipfile import ZIP_STORED, ZipFile

import cv2
import numpy as np

from app.schemas import CarTrackLimits, IncidentSummary, ViolationAnalysisResponse

INCIDENT_DIR = Path(os.getenv("INCIDENT_DIR", "./incidents")).resolve()
MAX_INCIDENT_GAP_SECONDS = 0.25


@dataclass
class TrackEvidence:
    trajectory: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=90))
    seen_at: float = 0.0
    candidate: tuple[int, float] | None = None
    last_outside: float | None = None
    returned_at: float | None = None
    active: bool = False
    last_verified_outside: tuple[int, float] | None = None


@dataclass
class Incident:
    id: UUID
    track_id: int
    start_frame: int
    start_time: float
    confirmed_frame: int
    confirmed_time: float
    car: CarTrackLimits
    trajectory: tuple[tuple[float, float], ...]


class IncidentDetector:
    def __init__(self, sample_fps: float) -> None:
        self.max_gap = max(0.15, 1.5 / sample_fps)
        self.tracks: dict[int, TrackEvidence] = {}
        self._gaps: list[tuple[int, int, int]] = []

    def update(self, cars: list[CarTrackLimits], frame: int, timestamp: float) -> list[Incident]:
        self.tracks = {
            key: state for key, state in self.tracks.items() if timestamp - state.seen_at <= 0.75
        }
        incidents = []
        for car in cars:
            if car.track_id is None:
                continue
            state = self.tracks.setdefault(car.track_id, TrackEvidence())
            state.seen_at = timestamp
            x1, y1, x2, y2 = car.bbox_xyxy
            state.trajectory.append(((x1 + x2) / 2, y2))
            if car.status != "outside":
                state.candidate = None
                state.last_outside = None
                if car.status in {"inside", "partial"}:
                    state.last_verified_outside = None
                    if state.returned_at is None:
                        state.returned_at = timestamp
                    if timestamp - state.returned_at >= 0.2:
                        state.active = False
                else:
                    state.returned_at = None
                continue

            state.returned_at = None
            if state.active:
                if state.last_verified_outside is not None:
                    previous_frame, previous_time = state.last_verified_outside
                    if (frame - previous_frame > 1
                            and timestamp - previous_time <= MAX_INCIDENT_GAP_SECONDS):
                        self._gaps.append((car.track_id, previous_frame, frame))
                state.last_verified_outside = frame, timestamp
                continue
            if (state.candidate is None or state.last_outside is None
                    or timestamp - state.last_outside > self.max_gap):
                state.candidate = frame, timestamp
                state.last_outside = timestamp
                continue
            start_frame, start_time = state.candidate
            incidents.append(Incident(
                uuid4(), car.track_id, start_frame, start_time, frame, timestamp,
                car, tuple(state.trajectory),
            ))
            state.active = True
            state.last_verified_outside = frame, timestamp
        # A missing detection breaks a pending confirmation but keeps active events latched.
        for state in self.tracks.values():
            if state.seen_at < timestamp:
                state.candidate = None
                state.last_outside = None
                state.returned_at = None
        return incidents

    def mark_gap_frames(self, results: list[ViolationAnalysisResponse]) -> None:
        """Show where adjacent evidence covers a detection gap in a confirmed incident."""
        source_indices = [
            frame.source_frame_index if frame.source_frame_index is not None else frame.frame_index
            for frame in results
        ]
        for track_id, previous, following in self._gaps:
            start = bisect_right(source_indices, previous)
            end = bisect_left(source_indices, following)
            for frame in results[start:end]:
                frame.incident_gap_track_ids.append(track_id)


def evidence_frame(
    frame: np.ndarray, incident: Incident, camera_id: str, track_points: list[list[float]],
    wheels: list[dict],
) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]
    size = np.array([width, height])
    path = np.rint(np.array(incident.trajectory) * size).astype(np.int32)
    cv2.polylines(image, [path], False, (255, 220, 30), 4, cv2.LINE_AA)
    if track_points:
        boundary = np.rint(np.array(track_points) * size).astype(np.int32)
        cv2.polylines(image, [boundary], True, (40, 220, 40), 3, cv2.LINE_AA)
    for index in incident.car.wheel_ids:
        wheel = np.rint(np.array(wheels[index]["points"]) * size).astype(np.int32)
        cv2.polylines(image, [wheel], True, (30, 65, 255), 3, cv2.LINE_AA)
    x1, y1, x2, y2 = (round(v * s) for v, s in zip(incident.car.bbox_xyxy, (width, height) * 2))
    cv2.rectangle(image, (x1, y1), (x2, y2), (30, 65, 255), 4)
    label = f"{camera_id}  |  Track {incident.track_id}  |  {incident.confirmed_time:.3f} s"
    cv2.putText(image, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def _write_clip(source: str, target: Path, first: int, last: int, fps: float) -> int:
    capture = cv2.VideoCapture(source)
    writer = None
    try:
        if first and not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise RuntimeError("Could not seek to the incident")
        count = 0
        for _ in range(first, last + 1):
            ok, image = capture.read()
            if not ok:
                break
            if writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("Could not open the MP4 writer")
            writer.write(image)
            count += 1
        if not count:
            raise RuntimeError("No source frames found for the incident")
        return count
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def write_incident(
    directory: Path, run_id: UUID, source: str, filename: str, fps: float, camera_id: str,
    incident: Incident, still: np.ndarray,
) -> IncidentSummary:
    first = max(0, incident.start_frame - math.ceil(2 * fps))
    last = incident.confirmed_frame + math.ceil(2 * fps)
    summary = IncidentSummary(
        id=str(incident.id), track_id=incident.track_id, camera_id=camera_id,
        start_frame=incident.start_frame, start_time=incident.start_time,
        confirmed_frame=incident.confirmed_frame, confirmed_time=incident.confirmed_time,
        download_url=f"/api/incidents/{run_id}/{incident.id}",
    )
    clip = directory / f"{incident.id}.mp4"
    frames = _write_clip(source, clip, first, last, fps)
    if first + frames <= incident.confirmed_frame:
        raise RuntimeError("Incident clip ends before the confirmation frame")
    ok, jpeg = cv2.imencode(".jpg", still)
    if not ok:
        raise RuntimeError("Could not encode the incident image")
    metadata = {
        **summary.model_dump(), "source_filename": filename, "source_fps": fps,
        "clip_first_frame": first, "clip_last_frame": first + frames - 1,
        "trajectory_normalized": incident.trajectory,
    }
    with ZipFile(directory / f"{incident.id}.zip", "w", compression=ZIP_STORED) as package:
        package.write(clip, "clip.mp4")
        package.writestr("frame.jpg", jpeg.tobytes())
        package.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    clip.unlink()
    return summary
