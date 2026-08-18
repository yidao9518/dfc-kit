import unittest

import numpy as np

from dfckit import TimeSeriesRun
from dfckit.connectivity import SlidingWindowFC


class SlidingWindowFCTests(unittest.TestCase):
    def test_metadata_preserves_segment_boundaries(self):
        rng = np.random.default_rng(17)
        run = TimeSeriesRun(
            values=rng.normal(size=(12, 4)),
            original_indices=np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14]),
            roi_names=("a", "b", "c", "d"),
            subject="sub-001",
            session="off",
        )
        result = SlidingWindowFC(length=4, step=2).transform(run)
        np.testing.assert_array_equal(result.start_frames, [0, 8, 10])
        np.testing.assert_array_equal(result.end_frames, [3, 11, 13])
        np.testing.assert_array_equal(result.segment_ids, [0, 1, 1])
        self.assertEqual(result.features.shape, (3, 6))

    def test_raises_when_no_segment_is_long_enough(self):
        run = TimeSeriesRun(
            values=np.arange(12, dtype=float).reshape(6, 2),
            original_indices=np.array([0, 1, 4, 5, 8, 9]),
            roi_names=("a", "b"),
        )
        with self.assertRaisesRegex(ValueError, "no contiguous segment"):
            SlidingWindowFC(length=3, step=1).transform(run)


if __name__ == "__main__":
    unittest.main()
