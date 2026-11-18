import unittest

import cv2
import numpy as np

from app.services.track_stabilization import TrackStabilizer


class TrackStabilizationTests(unittest.TestCase):
    def setUp(self):
        self.height, self.width = 360, 640
        self.points = [[0.08, 0.55], [0.92, 0.55], [0.92, 0.90], [0.08, 0.90]]
        self.reference = np.full((self.height, self.width, 3), 42, dtype=np.uint8)
        rng = np.random.default_rng(21)
        for y in range(18, 180, 22):
            for x in range(16, self.width - 16, 25):
                color = int(rng.integers(65, 220))
                x += int(rng.integers(-5, 5))
                cv2.rectangle(self.reference, (x, y), (x + 7, y + 9), (color,) * 3, -1)
        cv2.line(self.reference, (51, 198), (589, 198), (230,) * 3, 2)

    def test_translation_and_small_rotation_keep_boundary_on_same_pixels(self):
        stabilizer = TrackStabilizer(self.reference, self.points)
        for angle, shift in ((0.0, (9, -4)), (1.0, (-4, 3))):
            with self.subTest(angle=angle, shift=shift):
                matrix = cv2.getRotationMatrix2D((self.width / 2, self.height / 2), angle, 1)
                matrix[:, 2] += shift
                moved = cv2.warpAffine(self.reference, matrix, (self.width, self.height))
                # A bright moving car occupies part of the track surface.
                cv2.rectangle(moved, (210, 222), (395, 290), (40, 130, 240), -1)
                expected = cv2.transform(
                    (np.asarray(self.points) * (self.width, self.height)).reshape(-1, 1, 2), matrix,
                ).reshape(-1, 2)
                actual = np.asarray(stabilizer.adjust(moved, self.points)) * (self.width, self.height)
                np.testing.assert_allclose(actual, expected, atol=0.9, rtol=0)

    def test_falls_back_for_large_motion_or_insufficient_features(self):
        stabilizer = TrackStabilizer(self.reference, self.points)
        matrix = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0]])
        large_move = cv2.warpAffine(self.reference, matrix, (self.width, self.height))
        blank = np.zeros_like(self.reference)
        self.assertIs(stabilizer.adjust(large_move, self.points), self.points)
        self.assertIs(stabilizer.adjust(self.reference, self.points), self.points)
        self.assertIs(TrackStabilizer(blank, self.points).adjust(blank, self.points), self.points)


if __name__ == "__main__":
    unittest.main()
