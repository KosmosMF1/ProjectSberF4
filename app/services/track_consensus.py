from math import ceil

import cv2
import numpy as np

GRID_SIZE = 512
MIN_MASK_IOU = 0.65


def reference_frame_indices(frame_count: int, fps: float, sample_fps: float,
                            max_frames: int, budget: int) -> list[int]:
    """Spread inspection across the source frames covered by video analysis."""
    last = min(frame_count - 1, ceil((max_frames - 1) * fps / min(fps, sample_fps) - 1e-9))
    count = min(budget, last + 1)
    if count < 2:
        return [0] if count == 1 else []
    return [round(i * last / (count - 1)) for i in range(count)]


def consensus_track_polygon(
    anchor: list[list[float]], alternatives: list[list[list[float]]],
) -> tuple[list[list[float]], int] | None:
    """Fuse masks that describe the same track rather than disconnected detections."""
    def rasterize(points: list[list[float]]) -> np.ndarray:
        polygon = np.rint(np.asarray(points) * (GRID_SIZE - 1)).astype(np.int32)
        raster = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        cv2.fillPoly(raster, [polygon], 1)
        return raster

    reference = rasterize(anchor)
    votes = reference.copy()
    count = 1
    for points in alternatives:
        candidate = rasterize(points)
        intersection = cv2.countNonZero(reference & candidate)
        union = cv2.countNonZero(reference | candidate)
        if union and intersection / union >= MIN_MASK_IOU:
            votes += candidate
            count += 1
    if count < 3:
        return None

    consensus = (votes > count / 2).astype(np.uint8)
    if np.array_equal(consensus, reference):
        return anchor, count
    contours, _ = cv2.findContours(consensus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return None
    points = cv2.approxPolyDP(contours[0], 1, True).reshape(-1, 2)
    if len(points) < 3:
        return None
    return (points / (GRID_SIZE - 1)).tolist(), count
