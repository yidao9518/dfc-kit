import unittest

import numpy as np

from dfckit import TimeSeriesRun
from dfckit.connectivity import MTD, cross_block_mtd, within_block_mtd


class MTDTests(unittest.TestCase):
    def setUp(self):
        self.values = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, -1.0],
                [3.0, 1.0, 1.0],
                [100.0, -50.0, 20.0],
                [104.0, -47.0, 21.0],
                [109.0, -41.0, 25.0],
            ]
        )
        self.run = TimeSeriesRun(
            values=self.values,
            original_indices=np.asarray([0, 1, 2, 5, 6, 7]),
            roi_names=("visual", "motor", "putamen"),
            subject="sub-001",
            session="off",
        )

    def test_global_standardization_matches_direct_valid_derivatives(self):
        result = MTD().transform(self.run)
        valid = np.vstack(
            [
                np.diff(self.values[:3], axis=0),
                np.diff(self.values[3:], axis=0),
            ]
        )
        expected = (valid - valid.mean(axis=0)) / valid.std(axis=0, ddof=0)

        np.testing.assert_allclose(result.standardized_derivatives, expected)
        np.testing.assert_allclose(result.standardized_derivatives.mean(axis=0), 0.0, atol=1e-15)
        np.testing.assert_allclose(result.standardized_derivatives.std(axis=0), 1.0)

    def test_derivatives_never_cross_the_censor_gap(self):
        result = MTD().transform(self.run)

        np.testing.assert_array_equal(result.start_frames, [0, 1, 5, 6])
        np.testing.assert_array_equal(result.end_frames, [1, 2, 6, 7])
        np.testing.assert_array_equal(result.segment_ids, [0, 0, 1, 1])
        self.assertNotIn((2, 5), zip(result.start_frames, result.end_frames, strict=True))

    def test_edge_products_follow_upper_triangle_order(self):
        result = MTD().transform(self.run)
        derivatives = result.standardized_derivatives

        np.testing.assert_array_equal(result.edge_i, [0, 0, 1])
        np.testing.assert_array_equal(result.edge_j, [1, 2, 2])
        expected = np.column_stack(
            [derivatives[:, 0] * derivatives[:, 1],
             derivatives[:, 0] * derivatives[:, 2],
             derivatives[:, 1] * derivatives[:, 2]]
        )
        np.testing.assert_allclose(result.features, expected)

    def test_cross_and_within_block_summaries(self):
        result = MTD().transform(self.run)
        derivatives = result.standardized_derivatives

        cross = cross_block_mtd(derivatives, [0, 1], [2])
        within = within_block_mtd(derivatives, [0, 1, 2])

        expected_cross = (
            derivatives[:, 0] * derivatives[:, 2]
            + derivatives[:, 1] * derivatives[:, 2]
        ) / 2.0
        expected_within = result.features.mean(axis=1)
        np.testing.assert_allclose(cross, expected_cross)
        np.testing.assert_allclose(within, expected_within)

    def test_undefined_derivative_scale_is_rejected(self):
        run = TimeSeriesRun(
            values=np.column_stack([np.arange(5), np.arange(5) ** 2]),
            original_indices=np.arange(5),
            roi_names=("linear", "quadratic"),
        )
        with self.assertRaisesRegex(ValueError, r"ROI indices \[0\]"):
            MTD().transform(run)

    def test_cross_blocks_must_be_disjoint(self):
        derivatives = MTD().transform(self.run).standardized_derivatives
        with self.assertRaisesRegex(ValueError, "disjoint"):
            cross_block_mtd(derivatives, [0, 1], [1, 2])

    def test_segment_ids_preserve_single_frame_islands(self):
        run = TimeSeriesRun(
            values=np.asarray([[9.0, 4.0], [0.0, 0.0], [1.0, 2.0], [3.0, 1.0]]),
            original_indices=np.asarray([0, 3, 4, 5]),
            roi_names=("visual", "motor"),
        )

        result = MTD().transform(run)

        np.testing.assert_array_equal(result.segment_ids, [1, 1])


if __name__ == "__main__":
    unittest.main()
