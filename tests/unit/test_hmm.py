import importlib.util
import unittest

import numpy as np

from dfckit import TimeSeriesDataset, TimeSeriesRun
from dfckit.states import (
    FeatureSequence,
    FeatureSequenceDataset,
    fit_gaussian_hmm_states,
    predict_gaussian_hmm_states,
    summarize_state_assignments,
    timeseries_sequences,
)

HMM_AVAILABLE = (
    importlib.util.find_spec("hmmlearn") is not None
    and importlib.util.find_spec("sklearn") is not None
)


def make_sequence(subject, segment_id, seed, *, n_samples=60):
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    state = np.r_[np.zeros(half), np.ones(n_samples - half)]
    means = np.asarray([[-3.0, -1.0, 0.5], [3.0, 1.0, -0.5]])
    values = means[state.astype(int)] + 0.25 * rng.normal(size=(n_samples, 3))
    start = segment_id * 100
    indices = np.arange(start, start + n_samples)
    return FeatureSequence(
        values=values,
        sample_start_indices=indices,
        sample_end_indices=indices,
        feature_keys=(("roi-a",), ("roi-b",), ("roi-c",)),
        subject=subject,
        session="off",
        segment_id=segment_id,
        source_contract="synthetic-roi-timeseries",
        sample_interval_seconds=0.8,
    )


class TimeSeriesSequenceTests(unittest.TestCase):
    def test_raw_sequences_preserve_censor_segments_and_original_indices(self):
        run = TimeSeriesRun(
            values=np.arange(12, dtype=float).reshape(4, 3),
            original_indices=[0, 1, 5, 6],
            roi_names=("a", "b", "c"),
            subject="sub-a",
            session="off",
            tr=0.8,
        )

        sequences = timeseries_sequences(
            TimeSeriesDataset([run]),
            minimum_segment_length=2,
        )

        self.assertEqual(len(sequences.sequences), 2)
        np.testing.assert_array_equal(sequences.sequences[0].sample_start_indices, [0, 1])
        np.testing.assert_array_equal(sequences.sequences[1].sample_start_indices, [5, 6])
        self.assertEqual([sequence.segment_id for sequence in sequences.sequences], [0, 1])
        self.assertEqual(sequences.sample_interval_seconds, 0.8)


@unittest.skipUnless(HMM_AVAILABLE, "requires dfc-kit[hmm]")
class GaussianHMMTests(unittest.TestCase):
    def setUp(self):
        self.dataset = FeatureSequenceDataset(
            [
                make_sequence("sub-a", 0, 71),
                make_sequence("sub-a", 1, 72),
                make_sequence("sub-b", 0, 73),
                make_sequence("sub-b", 1, 74),
            ]
        )

    def fit(self):
        return fit_gaussian_hmm_states(
            self.dataset,
            n_states=2,
            seed=75,
            n_pca_components=2,
            covariance_type="diag",
            n_init=2,
            n_iter=100,
            tol=1e-4,
        )

    def test_fit_recovers_emissions_and_returns_gap_bounded_posteriors(self):
        fitted = self.fit()
        model = fitted.model

        self.assertEqual(model.fit_subjects, ("sub-a", "sub-b"))
        self.assertEqual(model.fit_sequence_count, 4)
        self.assertEqual(model.fit_sample_count, 240)
        self.assertEqual(model.reduced_means.shape, (2, 2))
        self.assertEqual(model.reduced_covariances.shape, (2, 2, 2))
        self.assertEqual(model.emission_means.shape, (2, 3))
        self.assertEqual(model.emission_covariances.shape, (2, 3, 3))
        np.testing.assert_allclose(model.transition_matrix.sum(axis=1), 1.0)
        np.testing.assert_allclose(model.start_probabilities.sum(), 1.0)
        self.assertLess(model.emission_means[:, 0].min(), -2.5)
        self.assertGreater(model.emission_means[:, 0].max(), 2.5)

        self.assertEqual(len(fitted.states.assignments.sequences), 4)
        self.assertEqual(len(fitted.states.posterior_probabilities), 4)
        for posterior in fitted.states.posterior_probabilities:
            self.assertEqual(posterior.shape, (60, 2))
            np.testing.assert_allclose(posterior.sum(axis=1), 1.0)
        metrics = summarize_state_assignments(fitted.states.assignments)
        self.assertTrue(all(row.n_sequences == 2 for row in metrics))
        self.assertTrue(all(row.n_possible_transitions == 118 for row in metrics))

    def test_fixed_seed_reproduces_parameters_and_selected_initialization(self):
        first = self.fit().model
        second = self.fit().model

        np.testing.assert_allclose(first.transition_matrix, second.transition_matrix)
        np.testing.assert_allclose(first.reduced_means, second.reduced_means)
        np.testing.assert_allclose(
            first.initialization_log_likelihoods,
            second.initialization_log_likelihoods,
        )
        self.assertEqual(first.selected_initialization, second.selected_initialization)

    def test_prediction_rejects_fit_subject_overlap_and_scores_heldout_sequences(self):
        fitted = self.fit()
        with self.assertRaisesRegex(ValueError, "overlap"):
            predict_gaussian_hmm_states(fitted.model, self.dataset)

        heldout = FeatureSequenceDataset(
            [make_sequence("sub-heldout", 0, 76), make_sequence("sub-heldout", 1, 77)]
        )
        predicted = predict_gaussian_hmm_states(fitted.model, heldout)

        self.assertEqual(len(predicted.assignments.sequences), 2)
        self.assertTrue(np.isfinite(predicted.log_likelihood))

    def test_short_sequences_are_omitted_from_fit_and_decode(self):
        short = make_sequence("sub-short", 0, 78, n_samples=1)
        dataset = FeatureSequenceDataset([*self.dataset.sequences, short])

        fitted = fit_gaussian_hmm_states(
            dataset,
            n_states=2,
            seed=79,
            n_pca_components=2,
            n_init=1,
            minimum_sequence_length=2,
        )

        self.assertEqual(fitted.model.omitted_short_sequence_count, 1)
        self.assertNotIn("sub-short", fitted.model.fit_subjects)
        self.assertEqual(len(fitted.states.assignments.sequences), 4)

    def test_full_covariance_and_identity_pca_paths_are_supported(self):
        full = fit_gaussian_hmm_states(
            self.dataset,
            n_states=2,
            seed=80,
            n_pca_components=2,
            covariance_type="full",
            n_init=1,
            n_iter=60,
        ).model
        identity = fit_gaussian_hmm_states(
            self.dataset,
            n_states=2,
            seed=81,
            n_pca_components=None,
            covariance_type="diag",
            n_init=1,
            n_iter=60,
        ).model

        self.assertEqual(full.reduced_covariances.shape, (2, 2, 2))
        np.testing.assert_allclose(
            full.reduced_covariances,
            full.reduced_covariances.transpose(0, 2, 1),
        )
        self.assertEqual(identity.n_pca_components, 3)
        np.testing.assert_allclose(identity.pca_components, np.eye(3))
        np.testing.assert_allclose(identity.pca_mean, 0.0)

    def test_model_arrays_are_read_only_and_invalid_options_are_rejected(self):
        fitted = self.fit()
        with self.assertRaises(ValueError):
            fitted.model.transition_matrix[0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "covariance_type"):
            fit_gaussian_hmm_states(
                self.dataset,
                n_states=2,
                seed=1,
                covariance_type="spherical",
            )


if __name__ == "__main__":
    unittest.main()
