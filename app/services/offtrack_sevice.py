import cv2
import numpy as np
from typing import Any
from app.schemas import OfftrackAnalysisResult, WheelOfftrackDetail

class OfftrackDetector:
    """
    Core logic:
    - build a binary track mask from the track polygon;
    - convert each wheel polygon to a binary mask;
    - compute how much of every wheel lies outside the track;
    - flag a violation if the ratio crosses a threshold.
    """

    def __init__(
        self,
        offtrack_threshold: float = 0.12,
        hard_violation_threshold: float = 0.25,
        boundary_tolerance_px: int = 4,
    ) -> None:
        self.offtrack_threshold = offtrack_threshold
        self.hard_violation_threshold = hard_violation_threshold
        self.boundary_tolerance_px = boundary_tolerance_px

    @staticmethod
    def _to_abs_polygon(points: list[list[float]], width: int, height: int) -> np.ndarray:
        poly = np.asarray(points, dtype=np.float32)
        if poly.size == 0:
            return poly
        # If points look normalized, convert to absolute pixels.
        if np.nanmax(poly) <= 1.5:
            poly[:, 0] *= width
            poly[:, 1] *= height
        return np.round(poly).astype(np.int32)

    @staticmethod
    def _polygon_mask(shape_hw: tuple[int, int], polygon_px: np.ndarray) -> np.ndarray:
        mask = np.zeros(shape_hw, dtype=np.uint8)
        if polygon_px.size == 0 or len(polygon_px) < 3:
            return mask.astype(bool)
        cv2.fillPoly(mask, [polygon_px.reshape(-1, 1, 2)], 1)
        return mask.astype(bool)

    @staticmethod
    def _mask_to_overlay(frame_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
        overlay = frame_bgr.copy()
        colored = np.zeros_like(frame_bgr)
        colored[:, :] = np.array(color_bgr, dtype=np.uint8)
        overlay[mask] = cv2.addWeighted(frame_bgr[mask], 1.0 - alpha, colored[mask], alpha, 0)
        return overlay

    def analyze(
        self,
        frame_bgr: np.ndarray,
        track_points: list[list[float]] | None,
        wheel_masks: list[dict[str, Any]],
    ) -> OfftrackAnalysisResult:
        if not track_points:
            return OfftrackAnalysisResult(
                violation_detected=False,
                violation_score=0.0,
                reason="",
                offtrack_wheels=None,
                annotated_frame_bgr=None,
                track_mask=None,
                violation_mask=None,
            )
        height, width = frame_bgr.shape[:2]
        track_poly = self._to_abs_polygon(track_points, width, height)
        track_mask = self._polygon_mask((height, width), track_poly)

        # Tolerance band to reduce false positives from boundary jaggedness.
        kernel_size = max(1, int(self.boundary_tolerance_px) * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        track_mask_tolerant = cv2.dilate(track_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        violation_mask = np.zeros((height, width), dtype=bool)
        details: list[WheelOfftrackDetail] = []
        violation_score = 0.0
        hard_violation = False

        for idx, wheel in enumerate(wheel_masks):
            poly_points = wheel.get("points", [])
            poly_px = self._to_abs_polygon(poly_points, width, height)
            wheel_mask = self._polygon_mask((height, width), poly_px)
            total_pixels = int(wheel_mask.sum())
            if total_pixels == 0:
                continue

            outside_mask = wheel_mask & (~track_mask_tolerant)
            outside_pixels = int(outside_mask.sum())
            outside_ratio = outside_pixels / float(total_pixels)

            details.append(
                WheelOfftrackDetail(
                    instance_id=int(wheel.get("instance_id", idx)),
                    outside_ratio=float(outside_ratio),
                    outside_pixels=outside_pixels,
                    total_pixels=total_pixels,
                )
            )

            violation_mask |= outside_mask
            violation_score = max(violation_score, outside_ratio)

            if outside_ratio >= self.hard_violation_threshold:
                hard_violation = True

        violation_detected = any(d.outside_ratio >= self.offtrack_threshold for d in details)
        if hard_violation:
            violation_detected = True

        if not details:
            reason = "wheel masks were not found"
        elif violation_detected:
            reason = "at least one wheel has crossed the track boundary"
        else:
            reason = "all wheels are inside the track boundary"

        annotated = frame_bgr.copy()

        # Track overlay in green.
        annotated = self._mask_to_overlay(annotated, track_mask, (0, 180, 0), alpha=0.18)

        # Highlight violation pixels in red.
        if violation_mask.any():
            annotated = self._mask_to_overlay(annotated, violation_mask, (0, 0, 255), alpha=0.65)

        # Draw track border.
        if len(track_poly) >= 3:
            cv2.polylines(annotated, [track_poly.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 255), thickness=3)

        # Draw wheel contours.
        for wheel in wheel_masks:
            poly_points = wheel.get("points", [])
            poly_px = self._to_abs_polygon(poly_points, width, height)
            if len(poly_px) >= 3:
                cv2.polylines(annotated, [poly_px.reshape(-1, 1, 2)], isClosed=True, color=(255, 255, 255), thickness=2)

        return OfftrackAnalysisResult(
            violation_detected=violation_detected,
            violation_score=float(violation_score),
            reason=reason,
            offtrack_wheels=details,
            annotated_frame_bgr=annotated,
            track_mask=track_mask,
            violation_mask=violation_mask,
        )
