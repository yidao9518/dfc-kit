import unittest

import numpy as np

from dfckit import TimeSeriesRun
from dfckit._preprocessing import _segment_standardized_samples


class SegmentStandardizationTests(unittest.TestCase):
    def test_standardizes_each_segment_and_preserves_frame_metadata(self):
        run = TimeSeriesRun(
            values=np.asarray(
                [
                    [1.0, 5.0],
                    [3.0, 5.0],
                    [10.0, 2.0],
                    [14.0, 6.0],
                ]
            ),
            original_indices=np.asarray([0, 1, 4, 5]),
            roi_names=("visual", "motor"),
        )

        values, original, segments = _segment_standardized_samples(run, method_name="test")

        np.testing.assert_allclose(values, [[-1.0, 0.0], [1.0, 0.0], [-1.0, -1.0], [1.0, 1.0]])
        np.testing.assert_array_equal(original, [0, 1, 4, 5])
        np.testing.assert_array_equal(segments, [0, 0, 1, 1])
        self.assertFalse(values.flags.writeable)


if __name__ == "__main__":
    unittest.main()
