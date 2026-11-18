import math

import cv2
import numpy as np


MAX_REFERENCE_SIDE = 960
MIN_BACKGROUND_POINTS = 8


class TrackStabilizer:
    """Estimate small camera motion from image features outside the reference track."""

    def __init__(self, reference: np.ndarray, track_points: list[list[float]]) -> None:
        height, width = reference.shape[:2]
        scale = min(1.0, MAX_REFERENCE_SIDE / max(width, height))
        self.size = (max(1, round(width * scale)), max(1, round(height * scale)))
        self.aspect_ratio = width / height
        self.reference = self._gray(reference)

        background = np.full(self.reference.shape, 255, dtype=np.uint8)
        polygon = np.rint(np.asarray(track_points) * self.size).astype(np.int32)
        cv2.fillPoly(background, [polygon], 0)
        background = cv2.erode(background, np.ones((7, 7), np.uint8))
        self.features = cv2.goodFeaturesToTrack(
            self.reference, maxCorners=160, qualityLevel=0.01, minDistance=7,
            mask=background, blockSize=5,
        )

    @property
    def available(self) -> bool:
        return self.features is not None and len(self.features) >= MIN_BACKGROUND_POINTS

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2][::-1] != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        height, width = frame.shape[:2]
        if not self.available or abs(width / height / self.aspect_ratio - 1) > 0.005:
            return None

        current = self._gray(frame)
        previous = self.features
        target, forward, _ = cv2.calcOpticalFlowPyrLK(self.reference, current, previous, None)
        if target is None or forward is None:
            return None
        keep = (forward.ravel() == 1) & np.isfinite(target).all(axis=(1, 2))
        previous, target = previous[keep], target[keep]
        if len(previous) < MIN_BACKGROUND_POINTS:
            return None

        returned, backward, _ = cv2.calcOpticalFlowPyrLK(current, self.reference, target, None)
        if returned is None or backward is None:
            return None
        error = np.linalg.norm((returned - previous).reshape(-1, 2), axis=1)
        keep = (backward.ravel() == 1) & (error < 1.0)
        previous, target = previous[keep], target[keep]
        if len(previous) < MIN_BACKGROUND_POINTS:
            return None

        matrix, inliers = cv2.estimateAffinePartial2D(
            previous, target, method=cv2.RANSAC, ransacReprojThreshold=1.5,
        )
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            return None
        support = previous[inliers.ravel().astype(bool)].reshape(-1, 2)
        if len(support) < max(MIN_BACKGROUND_POINTS, math.ceil(0.6 * len(previous))):
            return None
        if np.any(np.ptp(support, axis=0) < np.asarray(self.size) * (0.18, 0.10)):
            return None

        a, b = matrix[0, :2]
        if not 0.99 <= math.hypot(a, b) <= 1.01 or abs(math.degrees(math.atan2(b, a))) > 1.5:
            return None
        corners = np.asarray([[0, 0], [self.size[0], 0], [0, self.size[1]], self.size])
        motion = corners @ (matrix[:, :2] - np.eye(2)).T + matrix[:, 2]
        if np.linalg.norm(motion, axis=1).max() > max(self.size) * 0.025:
            return None
        return matrix

    def transform(self, points: list[list[float]], matrix: np.ndarray) -> list[list[float]]:
        pixel_points = np.asarray(points) * self.size
        return ((pixel_points @ matrix[:, :2].T + matrix[:, 2]) / self.size).tolist()

    def adjust(self, frame: np.ndarray, points: list[list[float]]) -> list[list[float]]:
        matrix = self.estimate(frame)
        if matrix is None:
            return points
        moved = self.transform(points, matrix)
        height, width = frame.shape[:2]
        displacement = (np.asarray(moved) - points) * (width, height)
        return moved if np.linalg.norm(displacement, axis=1).max() >= 0.75 else points
