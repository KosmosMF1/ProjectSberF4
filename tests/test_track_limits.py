import unittest

import numpy as np

from app.schemas import CarTrackLimits, ViolationAnalysisResponse
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

    def test_ambiguous_fifth_wheel_never_confirms_an_exit(self):
        car = rectangle(0.61, 0.19, 0.78, 0.36)
        wheels = wheels_at(0.63, 0.21) + [rectangle(0.685, 0.265, 0.71, 0.29, confidence=0.35)]
        result = self.analyze(wheels, [car])
        detection = result.cars[0]

        self.assertFalse(result.violation_detected)
        self.assertEqual(detection.status, "unknown")
        self.assertEqual(detection.wheel_candidates, 5)
        self.assertEqual(detection.wheel_ids, [0, 1, 2, 3])
        self.assertEqual({wheel.instance_id for wheel in result.offtrack_wheels}, {0, 1, 2, 3})

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

    def test_short_unknown_and_missing_frames_inherit_confirmed_incident_context(self):
        outside = CarTrackLimits(track_id=1, bbox_xyxy=(0.6, 0.1, 0.8, 0.3),
                                 status="outside", wheel_ids=[0, 1, 2, 3], wheels_outside=4)
        unknown = outside.model_copy(update={"status": "unknown", "wheel_ids": [0, 1, 2], "wheels_outside": 3})
        other_car = outside.model_copy(update={"track_id": 2})
        timeline = [
            (0.00, [outside]), (0.03, [outside]), (0.06, [unknown, other_car]),
            (0.10, []), (0.13, [outside]), (0.16, [unknown]),
        ]
        frames = self.incident_timeline(timeline)
        self.assertEqual([frame.incident_gap_track_ids for frame in frames],
                         [[], [], [1], [1], [], []])
        self.assertFalse(frames[3].violation_detected)
        self.assertEqual(frames[2].cars[0].status, "unknown")

    def test_long_unknown_or_partial_frames_do_not_extend_incidents(self):
        outside = CarTrackLimits(track_id=1, bbox_xyxy=(0.6, 0.1, 0.8, 0.3),
                                 status="outside", wheel_ids=[0, 1, 2, 3], wheels_outside=4)
        inside = outside.model_copy(update={"status": "inside", "wheels_outside": 0})
        partial = outside.model_copy(update={"status": "partial", "wheels_outside": 2})
        for blocker in (inside, partial):
            with self.subTest(status=blocker.status):
                frames = self.incident_timeline([
                    (0.00, [outside]), (0.03, [outside]), (0.06, []),
                    (0.10, [blocker]), (0.13, []), (0.16, [outside]),
                    (0.20, []), (0.43, [outside]),
                ])
                self.assertTrue(all(not frame.incident_gap_track_ids for frame in frames))

    def test_unconfirmed_observations_do_not_mark_gaps(self):
        outside = CarTrackLimits(track_id=1, bbox_xyxy=(0.6, 0.1, 0.8, 0.3),
                                 status="outside", wheel_ids=[0, 1, 2, 3], wheels_outside=4)
        frames = self.incident_timeline([(0.00, [outside]), (0.03, []), (0.06, [outside])])
        self.assertTrue(all(not frame.incident_gap_track_ids for frame in frames))

    @staticmethod
    def incident_timeline(timeline):
        detector = IncidentDetector(sample_fps=30)
        frames = []
        for index, (timestamp, cars) in enumerate(timeline):
            source_index = 2 * index + 10
            detector.update(cars, source_index, timestamp)
            frames.append(ViolationAnalysisResponse(
                frame_index=index, source_frame_index=source_index, timestamp=timestamp,
                frame_width=300, frame_height=300, frame_data_url="",
                annotated_frame_data_url=None, track_mask_data_url=None, violation_mask_data_url=None,
                violation_detected=any(car.status == "outside" for car in cars),
                violation_score=0, reason="", offtrack_wheels=None, violation_regions=None,
                masks=[], cars=cars,
            ))
        detector.mark_gap_frames(frames)
        return frames


if __name__ == "__main__":
    unittest.main()
