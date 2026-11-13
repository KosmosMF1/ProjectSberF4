import unittest

import numpy as np

from app.schemas import CarTrackLimits
from app.services import OfftrackDetector
from app.services.car_tracking import CarTracker, unique_masks
from app.services.incidents import IncidentDetector


def rectangle(x1, y1, x2, y2, confidence=1.0):
    return {
        "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "confidence": confidence,
    }


def wheels_at(x, y):
    return [rectangle(x + dx, y + dy, x + dx + 0.025, y + dy + 0.025)
            for dx in (0.0, 0.1) for dy in (0.0, 0.1)]


class TrackLimitsTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((300, 300, 3), np.uint8)
        self.track = rectangle(0.0, 0.0, 0.5, 0.99)["points"]
        self.detector = OfftrackDetector()

    def analyze(self, wheels, cars):
        return self.detector.analyze(self.frame, self.track, wheels, cars, annotate=False)

    def test_four_wheels_of_one_car_confirm_exit(self):
        cars = [rectangle(0.61, 0.19, 0.78, 0.36)]
        result = self.analyze(wheels_at(0.63, 0.21), cars)
        self.assertTrue(result.violation_detected)
        self.assertEqual(result.cars[0].wheels_outside, 4)

    def test_wheels_from_two_cars_do_not_confirm_exit(self):
        cars = [rectangle(0.61, 0.19, 0.78, 0.36), rectangle(0.61, 0.59, 0.78, 0.76)]
        result = self.analyze(wheels_at(0.63, 0.21)[:2] + wheels_at(0.63, 0.61)[:2], cars)
        self.assertFalse(result.violation_detected)
        self.assertEqual([car.status for car in result.cars], ["unknown", "unknown"])

    def test_partial_exit_does_not_confirm(self):
        result = self.analyze(wheels_at(0.46, 0.21), [rectangle(0.44, 0.19, 0.61, 0.36)])
        self.assertFalse(result.violation_detected)
        self.assertEqual(result.cars[0].status, "partial")

    def test_duplicate_wheel_cannot_complete_evidence(self):
        wheels = wheels_at(0.63, 0.21)[:3]
        wheels = unique_masks(wheels + [{**wheels[0], "confidence": 0.6}], iou=0.45, containment=0.75)
        result = self.analyze(wheels, [rectangle(0.61, 0.19, 0.78, 0.36)])
        self.assertFalse(result.violation_detected)

    def test_tracker_keeps_ids_after_reordered_detections_and_a_missing_frame(self):
        tracker = CarTracker()
        a = rectangle(0.10, 0.10, 0.25, 0.25)
        b = rectangle(0.50, 0.10, 0.65, 0.25)
        initial = tracker.update([a, b], 0.0)
        tracker.update([b], 0.1)
        resumed = tracker.update([b, a], 0.2)
        self.assertEqual([car["track_id"] for car in initial], [1, 2])
        self.assertEqual([car["track_id"] for car in resumed], [2, 1])

    def test_two_outside_observations_confirm_once_until_car_returns(self):
        detector = IncidentDetector(sample_fps=30)
        outside = CarTrackLimits(track_id=1, bbox_xyxy=(0.6, 0.1, 0.8, 0.3),
                                 status="outside", wheel_ids=[0, 1, 2, 3], wheels_outside=4)
        inside = outside.model_copy(update={"status": "inside", "wheels_outside": 0})
        self.assertEqual(detector.update([outside], 0, 0.0), [])
        self.assertEqual(detector.update([], 1, 0.033), [])
        self.assertEqual(detector.update([outside], 2, 0.066), [])
        first = detector.update([outside], 3, 0.1)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].start_frame, 2)
        self.assertEqual(detector.update([], 4, 0.133), [])
        self.assertEqual(detector.update([outside], 5, 0.166), [])
        self.assertEqual(detector.update([inside], 6, 0.2), [])
        self.assertEqual(detector.update([inside], 13, 0.433), [])
        self.assertEqual(detector.update([outside], 14, 0.466), [])
        self.assertEqual(len(detector.update([outside], 15, 0.5)), 1)


if __name__ == "__main__":
    unittest.main()
