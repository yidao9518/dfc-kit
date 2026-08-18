import importlib.util
import unittest

import numpy as np

from dfckit import TimeSeriesRun
from dfckit.connectivity import (
    LEiDA,
    analytic_phase,
    cross_block_phase_coherence,
    leading_phase_eigenvectors,
    within_block_phase_coherence,
)
from dfckit.states import leida_sequences

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


class LEiDAKernelTests(unittest.TestCase):
    def test_rank_two_solution_matches_direct_phase_coherence_eigendecomposition(self):
        phase = np.random.default_rng(51).uniform(-np.pi, np.pi, size=(12, 7))

        vectors, eigenvalues = leading_phase_eigenvectors(phase)

        expected_vectors = []
        expected_eigenvalues = []
        for row in phase:
            coherence = np.cos(row[:, None] - row[None, :])
            values, basis = np.linalg.eigh(coherence)
            vector = basis[:, -1]
            if vector.sum() < 0.0:
                vector *= -1.0
            expected_vectors.append(vector)
            expected_eigenvalues.append(values[-1])
        np.testing.assert_allclose(vectors, expected_vectors, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(eigenvalues, expected_eigenvalues, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
        self.assertTrue(np.all(vectors.sum(axis=1) >= -1e-12))

    def test_block_coherence_matches_explicit_cosine_differences(self):
        phase = np.random.default_rng(52).uniform(-np.pi, np.pi, size=(9, 6))

        cross = cross_block_phase_coherence(phase, [0, 1], [3, 4, 5])
        within = within_block_phase_coherence(phase, [0, 1, 2])

        expected_cross = np.cos(
            phase[:, [0, 1], None] - phase[:, None, [3, 4, 5]]
        ).mean(axis=(1, 2))
        expected_within = np.column_stack(
            [
                np.cos(phase[:, 0] - phase[:, 1]),
                np.cos(phase[:, 0] - phase[:, 2]),
                np.cos(phase[:, 1] - phase[:, 2]),
            ]
        ).mean(axis=1)
        np.testing.assert_allclose(cross, expected_cross)
        np.testing.assert_allclose(within, expected_within)

    def test_cross_blocks_must_be_disjoint(self):
        phase = np.zeros((5, 4))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            cross_block_phase_coherence(phase, [0, 1], [1, 2])

    @unittest.skipUnless(SCIPY_AVAILABLE, "requires dfc-kit[phase]")
    def test_analytic_phase_centers_each_roi_and_rejects_constant_signal(self):
        from scipy.signal import hilbert

        values = np.random.default_rng(53).normal(size=(25, 4)) + np.arange(4) * 10.0
        expected = np.angle(hilbert(values - values.mean(axis=0), axis=0))

        np.testing.assert_allclose(analytic_phase(values), expected)
        with self.assertRaisesRegex(ValueError, r"indices \[1\]"):
            analytic_phase(np.column_stack([values[:, 0], np.ones(25), values[:, 2]]))


@unittest.skipUnless(SCIPY_AVAILABLE, "requires dfc-kit[phase]")
class LEiDAEstimatorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(54)
        first = rng.normal(size=(20, 5))
        short = rng.normal(size=(19, 5))
        third = rng.normal(size=(21, 5))
        self.values = np.vstack([first, short, third])
        self.run = TimeSeriesRun(
            values=self.values,
            original_indices=np.r_[np.arange(20), np.arange(30, 49), np.arange(60, 81)],
            roi_names=("v1", "v2", "motor", "putamen", "thalamus"),
            subject="sub-001",
            session="off",
            tr=0.8,
        )

    def test_transform_applies_hilbert_independently_and_skips_short_segment(self):
        result = LEiDA(minimum_segment_length=20).transform(self.run)

        np.testing.assert_array_equal(
            result.original_indices,
            np.r_[np.arange(20), np.arange(60, 81)],
        )
        np.testing.assert_array_equal(result.segment_ids, np.r_[np.zeros(20), np.full(21, 2)])
        np.testing.assert_allclose(result.phase[:20], analytic_phase(self.values[:20]))
        np.testing.assert_allclose(result.phase[20:], analytic_phase(self.values[39:]))
        self.assertEqual(result.leading_vectors.shape, (41, 5))
        self.assertEqual(result.leading_eigenvalues.shape, (41,))
        self.assertEqual(result.orientation, "positive-vector-sum")

    def test_feature_sequences_preserve_segment_boundaries_and_metadata(self):
        result = LEiDA(minimum_segment_length=20).transform(self.run)
        dataset = leida_sequences([result])

        self.assertEqual(len(dataset.sequences), 2)
        self.assertEqual([sequence.segment_id for sequence in dataset.sequences], [0, 2])
        self.assertEqual([sequence.n_samples for sequence in dataset.sequences], [20, 21])
        self.assertEqual(dataset.feature_keys, tuple((name,) for name in self.run.roi_names))
        self.assertEqual(dataset.sample_interval_seconds, 0.8)
        self.assertIn("minimum-segment-length=20", dataset.source_contract)
        np.testing.assert_array_equal(
            dataset.sequences[1].sample_start_indices,
            np.arange(60, 81),
        )

    def test_outputs_are_read_only(self):
        result = LEiDA(minimum_segment_length=20).transform(self.run)

        with self.assertRaises(ValueError):
            result.phase[0, 0] = 0.0
        with self.assertRaises(ValueError):
            result.leading_vectors[0, 0] = 0.0

    def test_no_eligible_segment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no retained segment"):
            LEiDA(minimum_segment_length=22).transform(self.run)


if __name__ == "__main__":
    unittest.main()
