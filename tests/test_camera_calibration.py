import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from pydantic import ValidationError

from app.services.camera_calibration import (
    CameraCalibration, load_calibration, outline_from_mask, save_calibration,
)


class CameraCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.calibration = CameraCalibration(
            frame_width=2560,
            frame_height=1440,
            points=[(0.0, 0.2), (1.0, 0.2), (1.0, 1.0), (0.0, 1.0)],
        )

    def test_saved_boundaries_are_isolated_by_camera_and_can_be_replaced(self):
        with TemporaryDirectory() as directory, patch(
            "app.services.camera_calibration.CALIBRATION_DIR", Path(directory),
        ):
            save_calibration("camera-1", self.calibration)
            self.assertEqual(load_calibration("camera-1"), self.calibration)
            self.assertIsNone(load_calibration("camera-2"))

            second = CameraCalibration(
                frame_width=2560, frame_height=1440,
                points=[(0.1, 0.2), (0.9, 0.2), (0.3, 1.0)],
            )
            save_calibration("camera-1", second)
            self.assertEqual(load_calibration("camera-1"), second)
            self.assertEqual([path.name for path in Path(directory).iterdir()], ["camera-1.json"])

    def test_reuse_at_a_different_resolution_with_the_same_aspect_ratio(self):
        self.assertTrue(self.calibration.matches_frame(1280, 720))
        self.assertFalse(self.calibration.matches_frame(1920, 1440))

    def test_crossing_boundary_with_nonzero_area_is_rejected(self):
        with self.assertRaises(ValidationError):
            CameraCalibration(
                frame_width=2560, frame_height=1440,
                points=[(0.0, 0.0), (0.9, 0.9), (0.0, 1.0), (0.6, 0.0)],
            )

    def test_out_of_range_and_degenerate_boundaries_are_rejected(self):
        for points in (
            [(0.0, 0.0), (0.9, 0.9), (0.0, 1.1)],
            [(0.0, 0.0), (float("nan"), 0.9), (0.0, 1.0)],
            [(0.1, 0.5), (0.2, 0.5), (0.8, 0.5)],
            [(0.0, 0.0), (0.9, 0.9), (0.9, 0.9), (0.0, 1.0)],
        ):
            with self.subTest(points=points), self.assertRaises(ValidationError):
                CameraCalibration(frame_width=2560, frame_height=1440, points=points)

    def test_outline_keeps_a_brush_edit_as_a_valid_boundary(self):
        mask = np.zeros((121, 201), dtype=np.uint8)
        cv2.rectangle(mask, (20, 30), (180, 100), 255, -1)
        cv2.rectangle(mask, (80, 65), (120, 100), 0, -1)
        cv2.circle(mask, (50, 29), 12, 255, -1)

        points = outline_from_mask(mask)
        boundary = CameraCalibration(frame_width=201, frame_height=121, points=points)
        polygon = np.rint(np.asarray(boundary.points) * [200, 120]).astype(np.int32)
        self.assertGreater(cv2.pointPolygonTest(polygon, (50, 23), False), 0)
        self.assertLess(cv2.pointPolygonTest(polygon, (100, 80), False), 0)
        self.assertGreater(cv2.pointPolygonTest(polygon, (150, 70), False), 0)

    def test_outline_requires_one_connected_region_without_holes(self):
        empty = np.zeros((60, 100), dtype=np.uint8)
        separate = empty.copy()
        separate[10:30, 10:35] = 255
        separate[40:50, 60:90] = 255
        hole = empty.copy()
        hole[10:50, 10:90] = 255
        hole[20:40, 20:30] = 0
        for name, mask in (("empty", empty), ("separate", separate), ("hole", hole)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                outline_from_mask(mask)


if __name__ == "__main__":
    unittest.main()
