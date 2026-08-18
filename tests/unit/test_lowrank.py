import unittest

import numpy as np

from dfckit import TimeSeriesRun
from dfckit.connectivity import (
    LowRankCovariance,
    bidirectional_heldout_r2,
    effective_rank,
    eigen_concentration,
    fit_standardized_pca,
    heldout_reconstruction_r2,
    mean_projector_basis,
    subspace_distance,
    subspace_similarity,
)


class LowRankKernelTests(unittest.TestCase):
    def test_standardized_pca_matches_direct_svd(self):
        rng = np.random.default_rng(41)
        values = rng.normal(size=(18, 5)) * np.asarray([1.0, 2.0, 0.5, 4.0, 0.8]) + 3.0

        fitted = fit_standardized_pca(values, rank=3)
        standardized = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
        _, singular, transposed = np.linalg.svd(standardized, full_matrices=False)
        expected_proportion = singular**2 / np.sum(singular**2)

        np.testing.assert_allclose(fitted.mean, values.mean(axis=0))
        np.testing.assert_allclose(fitted.scale, values.std(axis=0, ddof=0))
        np.testing.assert_allclose(fitted.variance_proportion, expected_proportion)
        self.assertAlmostEqual(
            subspace_similarity(fitted.basis, transposed[:3].T), 1.0
        )

    def test_entropy_rank_and_concentration_have_known_values(self):
        proportion = np.asarray([0.5, 0.5, 0.0])

        self.assertAlmostEqual(effective_rank(proportion), 2.0)
        self.assertAlmostEqual(eigen_concentration(proportion, 1), 0.5)
        self.assertAlmostEqual(eigen_concentration(proportion, 2), 1.0)

    def test_low_rank_signal_generalizes_to_holdout(self):
        rng = np.random.default_rng(7)
        loadings = rng.normal(size=(2, 12))
        train = rng.normal(size=(96, 2)) @ loadings + 0.01 * rng.normal(size=(96, 12))
        test = rng.normal(size=(96, 2)) @ loadings + 0.01 * rng.normal(size=(96, 12))

        rank_one = heldout_reconstruction_r2(train, test, rank=1)
        rank_two = heldout_reconstruction_r2(train, test, rank=2)
        self.assertGreater(rank_two, 0.99)
        self.assertLess(rank_one, rank_two)
        combined = np.vstack([train, test])
        self.assertGreater(bidirectional_heldout_r2(combined, rank=2, split=96), 0.99)

    def test_subspace_metrics_ignore_sign_and_within_subspace_rotation(self):
        rng = np.random.default_rng(8)
        basis, _ = np.linalg.qr(rng.normal(size=(12, 4)))
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        transformed = basis @ rotation

        self.assertAlmostEqual(subspace_similarity(basis, transformed), 1.0)
        self.assertAlmostEqual(subspace_distance(basis, transformed), 0.0)

    def test_mean_projector_basis_recovers_common_geometry(self):
        rng = np.random.default_rng(9)
        basis, _ = np.linalg.qr(rng.normal(size=(10, 3)))
        rotation_a, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotation_b, _ = np.linalg.qr(rng.normal(size=(3, 3)))

        aggregate = mean_projector_basis(
            [basis @ rotation_a, basis @ rotation_b], rank=3
        )

        self.assertAlmostEqual(subspace_similarity(basis, aggregate), 1.0)

    def test_invalid_scale_and_nonorthonormal_basis_are_rejected(self):
        values = np.column_stack([np.arange(8, dtype=float), np.ones(8), np.arange(8) ** 2])
        with self.assertRaisesRegex(ValueError, r"ROI indices \[1\]"):
            fit_standardized_pca(values, rank=1)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            subspace_similarity(np.ones((4, 1)), np.ones((4, 1)))


class LowRankEstimatorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.run = TimeSeriesRun(
            values=rng.normal(size=(32, 4)),
            original_indices=np.r_[np.arange(16), np.arange(20, 36)],
            roi_names=("visual", "motor", "putamen", "thalamus"),
            subject="sub-001",
            session="off",
            tr=0.8,
        )

    def test_transform_preserves_windows_and_never_compares_across_gap(self):
        result = LowRankCovariance(length=8, step=8, ranks=(1, 2)).transform(self.run)

        np.testing.assert_array_equal(result.start_frames, [0, 8, 20, 28])
        np.testing.assert_array_equal(result.end_frames, [7, 15, 27, 35])
        np.testing.assert_array_equal(result.segment_ids, [0, 0, 1, 1])
        np.testing.assert_array_equal(result.adjacent_left_windows, [0, 2])
        np.testing.assert_array_equal(result.adjacent_right_windows, [1, 3])
        self.assertEqual(result.effective_rank.shape, (4,))
        self.assertEqual(result.eigen_concentration.shape, (4, 2))
        self.assertEqual(result.heldout_r2.shape, (4, 2))
        self.assertEqual(result.split_similarity.shape, (4, 2))
        self.assertEqual(result.adjacent_similarity.shape, (2, 2))
        self.assertEqual(result.window_bases[0].shape, (4, 4, 1))
        self.assertEqual(result.window_bases[1].shape, (4, 4, 2))
        self.assertEqual(result.run_bases[0].shape, (4, 1))
        self.assertEqual(result.run_bases[1].shape, (4, 2))

    def test_transform_matches_audited_window_formulas(self):
        result = LowRankCovariance(length=8, step=8, ranks=(1, 2)).transform(self.run)
        first = self.run.values[:8]
        standardized = (first - first.mean(axis=0)) / first.std(axis=0, ddof=0)
        _, singular, transposed = np.linalg.svd(standardized, full_matrices=False)
        proportion = singular**2 / np.sum(singular**2)
        positive = proportion[proportion > 0]

        expected_effective_rank = np.exp(-np.sum(positive * np.log(positive)))
        self.assertAlmostEqual(result.effective_rank[0], expected_effective_rank)
        np.testing.assert_allclose(result.eigen_concentration[0], np.cumsum(proportion)[:2])
        self.assertAlmostEqual(
            result.heldout_r2[0, 1],
            bidirectional_heldout_r2(first, rank=2, split=4),
        )
        self.assertAlmostEqual(
            subspace_similarity(result.window_bases[1][0], transposed[:2].T),
            1.0,
        )

    def test_result_arrays_are_read_only_and_rank_lookup_is_explicit(self):
        result = LowRankCovariance(length=8, step=8, ranks=(1, 2)).transform(self.run)

        self.assertEqual(result.rank_index(2), 1)
        with self.assertRaisesRegex(KeyError, "was not fitted"):
            result.rank_index(3)
        with self.assertRaises(ValueError):
            result.effective_rank[0] = 0.0
        with self.assertRaises(ValueError):
            result.window_bases[0][0, 0, 0] = 0.0

    def test_no_valid_window_and_rank_above_roi_count_are_rejected(self):
        short = TimeSeriesRun(
            values=self.run.values[:6],
            original_indices=np.asarray([0, 1, 4, 5, 8, 9]),
            roi_names=self.run.roi_names,
        )
        with self.assertRaisesRegex(ValueError, "no contiguous segment"):
            LowRankCovariance(length=4, step=2, ranks=(1,)).transform(short)
        with self.assertRaisesRegex(ValueError, "exceeds.*ROIs"):
            LowRankCovariance(length=12, step=12, ranks=(1, 2, 3, 4, 5)).transform(
                self.run
            )


if __name__ == "__main__":
    unittest.main()
