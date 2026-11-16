from typing import Any

import cv2
import numpy as np

from app.schemas import CarTrackLimits, OfftrackAnalysisResult, WheelOfftrackDetail
from app.services.car_tracking import assign_wheels, box_for


MAX_WHEELS_PER_CAR = 4


class OfftrackDetector:
    """Evaluate four distinct wheel masks belonging to the same car."""

    def __init__(
        self,
        offtrack_threshold: float = 0.12,
        hard_violation_threshold: float = 0.8,
        boundary_tolerance_px: int = 4,
    ) -> None:
        self.offtrack_threshold = offtrack_threshold
        self.hard_violation_threshold = hard_violation_threshold
        self.boundary_tolerance_px = boundary_tolerance_px

    @staticmethod
    def _to_abs_polygon(points: list[list[float]], width: int, height: int) -> np.ndarray:
        poly = np.array(points, dtype=np.float32).reshape(-1, 2)
        if poly.size and np.nanmax(poly) <= 1.5:
            poly *= [width, height]
        return np.round(poly).astype(np.int32)

    def track_masks(
        self, shape: tuple[int, int], points: list[list[float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = shape
        track = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(track, [self._to_abs_polygon(points, width, height)], 1)
        size = max(1, self.boundary_tolerance_px * 2 + 1)
        tolerant = cv2.dilate(track, np.ones((size, size), np.uint8))
        return track, tolerant

    def analyze(
        self,
        frame_bgr: np.ndarray,
        track_points: list[list[float]] | None,
        wheel_masks: list[dict[str, Any]],
        car_masks: list[dict[str, Any]],
        *,
        track_masks: tuple[np.ndarray, np.ndarray] | None = None,
        annotate: bool = True,
    ) -> OfftrackAnalysisResult:
        height, width = frame_bgr.shape[:2]
        track, tolerant = track_masks or (None, None)
        if tolerant is None and track_points and len(track_points) >= 3:
            track, tolerant = self.track_masks((height, width), track_points)

        candidates_by_car = assign_wheels(car_masks, wheel_masks)
        wheels_by_car = [
            sorted(indices, key=lambda index: wheel_masks[index].get("confidence") or 0, reverse=True)[:MAX_WHEELS_PER_CAR]
            for indices in candidates_by_car
        ]
        matched_wheels = sorted(index for indices in wheels_by_car for index in indices)
        details: dict[int, WheelOfftrackDetail] = {}
        outside_regions: dict[int, tuple[int, int, np.ndarray]] = {}
        if tolerant is not None:
            for index in matched_wheels:
                polygon = self._to_abs_polygon(wheel_masks[index]["points"], width, height)
                x, y, w, h = cv2.boundingRect(polygon)
                left, top = max(0, x), max(0, y)
                right, bottom = min(width, x + w), min(height, y + h)
                if right <= left or bottom <= top:
                    continue
                roi = np.zeros((bottom - top, right - left), dtype=np.uint8)
                cv2.fillPoly(roi, [polygon - [left, top]], 1)
                total = cv2.countNonZero(roi)
                if not total:
                    continue
                outside = roi & (1 - tolerant[top:bottom, left:right])
                count = cv2.countNonZero(outside)
                details[index] = WheelOfftrackDetail(
                    instance_id=index, outside_ratio=count / total,
                    outside_pixels=count, total_pixels=total,
                )
                outside_regions[index] = left, top, outside

        cars = []
        for car, candidates, indices in zip(car_masks, candidates_by_car, wheels_by_car, strict=True):
            scores = [details[index].outside_ratio for index in indices if index in details]
            wheels_beyond_edge = sum(score >= self.offtrack_threshold for score in scores)
            if len(candidates) > MAX_WHEELS_PER_CAR:
                status = "unknown"
            elif len(scores) == MAX_WHEELS_PER_CAR and min(scores) >= self.hard_violation_threshold:
                status = "outside"
            elif 0 < wheels_beyond_edge < len(scores):
                status = "partial"
            elif len(scores) == MAX_WHEELS_PER_CAR and wheels_beyond_edge == 0:
                status = "inside"
            else:
                status = "unknown"
            cars.append(CarTrackLimits(
                track_id=car.get("track_id"), bbox_xyxy=box_for(car), status=status,
                wheel_ids=indices, wheels_outside=wheels_beyond_edge, wheel_candidates=len(candidates),
            ))

        confirmed = [car for car in cars if car.status == "outside"]
        score = max(
            (min(details[index].outside_ratio for index in car.wheel_ids) for car in confirmed),
            default=0.0,
        )
        violation_mask = None
        annotated = None
        if track is not None and (confirmed or annotate):
            violation_mask = np.zeros((height, width), dtype=np.uint8)
            for car in confirmed:
                for index in car.wheel_ids:
                    x, y, region = outside_regions[index]
                    h, w = region.shape
                    violation_mask[y:y+h, x:x+w] |= region
        if annotate and track is not None:
            annotated = frame_bgr.copy()
            track_pixels = track.astype(bool)
            annotated[track_pixels] = annotated[track_pixels] * 0.82 + np.array([0, 180, 0]) * 0.18
            annotated[violation_mask.astype(bool)] = (0, 0, 255)
            if track_points:
                cv2.polylines(annotated, [self._to_abs_polygon(track_points, width, height)],
                              True, (0, 255, 255), 3)
            for index in matched_wheels:
                cv2.polylines(annotated, [self._to_abs_polygon(wheel_masks[index]["points"], width, height)],
                              True, (255, 255, 255), 2)

        if confirmed:
            reason = "four wheels of the same car are outside the track boundary"
        elif not cars or tolerant is None or not details or any(car.status == "unknown" for car in cars):
            reason = "insufficient evidence to determine the position of all wheels"
        else:
            reason = "no car has four wheels outside the track boundary"

        return OfftrackAnalysisResult(
            violation_detected=bool(confirmed), violation_score=score, reason=reason,
            offtrack_wheels=list(details.values()) or None,
            annotated_frame_bgr=annotated,
            track_mask=track.astype(bool) if annotate and track is not None else None,
            violation_mask=violation_mask.astype(bool) if violation_mask is not None else None,
            cars=cars,
        )
