import unittest

from app.services.track_consensus import consensus_track_polygon, reference_frame_indices


def rectangle(left, right):
    return [[left, 0.2], [right, 0.2], [right, 0.8], [left, 0.8]]


class TrackConsensusTests(unittest.TestCase):
    def test_spreads_reference_frames_across_the_analysis_window(self):
        self.assertEqual(reference_frame_indices(300, 30, 15, 5, 4), [0, 3, 5, 8])
        self.assertEqual(reference_frame_indices(6, 30, 30, 100, 3), [0, 2, 5])
        self.assertEqual(reference_frame_indices(1, 30, 30, 100, 8), [0])

    def test_restores_the_region_supported_by_other_frames(self):
        anchor = rectangle(0.1, 0.7)
        wider = rectangle(0.1, 0.8)
        polygon, count = consensus_track_polygon(anchor, [wider, wider, rectangle(0.85, 0.99)])

        self.assertEqual(count, 3)
        self.assertAlmostEqual(max(point[0] for point in polygon), 0.8, places=2)

    def test_skips_fusion_when_masks_disagree(self):
        anchor = rectangle(0.1, 0.5)
        self.assertIsNone(consensus_track_polygon(anchor, [anchor, rectangle(0.6, 0.9)]))

    def test_keeps_exact_contour_if_all_masks_agree(self):
        anchor = rectangle(0.12, 0.67)
        self.assertEqual(consensus_track_polygon(anchor, [anchor, anchor]), (anchor, 3))


if __name__ == "__main__":
    unittest.main()
