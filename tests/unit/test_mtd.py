import unittest

import numpy as np

from dfckit import TimeSeriesRun
from dfckit.connectivity import (
    MTD,
    InstantaneousEdgeResult,
    cross_block_products,
    edge_products,
    within_block_products,
)
from dfckit.states import instantaneous_edge_sequences


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

    def standardized_derivatives(self):
        valid = np.vstack(
            [
                np.diff(self.values[:3], axis=0),
                np.diff(self.values[3:], axis=0),
            ]
        )
        return (valid - valid.mean(axis=0)) / valid.std(axis=0, ddof=0)

    def test_global_standardization_matches_direct_valid_derivatives(self):
        result = MTD().transform(self.run)
        expected = self.standardized_derivatives()

        np.testing.assert_allclose(result.features, edge_products(expected))

    def test_derivatives_never_cross_the_censor_gap(self):
        result = MTD().transform(self.run)

        self.assertIsInstance(result, InstantaneousEdgeResult)
        self.assertEqual(result.sample_kind, "interval")
        np.testing.assert_array_equal(result.sample_start_frames, [0, 1, 5, 6])
        np.testing.assert_array_equal(result.sample_end_frames, [1, 2, 6, 7])
        np.testing.assert_array_equal(result.segment_ids, [0, 0, 1, 1])
        self.assertNotIn(
            (2, 5),
            zip(result.sample_start_frames, result.sample_end_frames, strict=True),
        )

    def test_edge_products_follow_upper_triangle_order(self):
        result = MTD().transform(self.run)

        np.testing.assert_array_equal(result.edge_i, [0, 0, 1])
        np.testing.assert_array_equal(result.edge_j, [1, 2, 2])
        expected = self.standardized_derivatives()
        np.testing.assert_allclose(result.features, edge_products(expected))

    def test_cross_and_within_block_summaries(self):
        estimator = MTD()
        result = estimator.transform(self.run)

        cross = result.cross_block([0, 1], [2])
        within = result.within_block([0, 1, 2])

        expected_cross = (result.features[:, 1] + result.features[:, 2]) / 2.0
        expected_within = result.features.mean(axis=1)
        np.testing.assert_allclose(cross, expected_cross)
        np.testing.assert_allclose(within, expected_within)
        expected = self.standardized_derivatives()
        np.testing.assert_allclose(cross, cross_block_products(expected, [0, 1], [2]))
        np.testing.assert_allclose(within, within_block_products(expected, [0, 1, 2]))
        self.assertEqual(cross.shape, result.rss.shape)
        self.assertEqual(within.shape, result.rss.shape)

    def test_common_result_enters_the_common_state_sequence_pipeline(self):
        result = MTD().transform(self.run)
        sequences = instantaneous_edge_sequences((result,))

        self.assertEqual(sequences.source_contract, result.source_contract)
        self.assertEqual(sequences.feature_keys, result.feature_keys)
        self.assertEqual(sequences.n_samples, result.n_samples)
        np.testing.assert_array_equal(
            sequences.sequences[0].sample_start_indices,
            result.sample_start_frames[:2],
        )

    def test_undefined_derivative_scale_is_rejected(self):
        run = TimeSeriesRun(
            values=np.column_stack([np.arange(5), np.arange(5) ** 2]),
            original_indices=np.arange(5),
            roi_names=("linear", "quadratic"),
        )
        with self.assertRaisesRegex(ValueError, r"ROI indices \[0\]"):
            MTD().transform(run)

    def test_cross_blocks_must_be_disjoint(self):
        result = MTD().transform(self.run)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            result.cross_block([0, 1], [1, 2])

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
