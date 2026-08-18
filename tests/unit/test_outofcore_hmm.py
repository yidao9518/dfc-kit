import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.outofcore_hmm import (
    fit_gaussian_hmm_store,
    predict_gaussian_hmm_store,
    score_gaussian_hmm_store,
)
from dfckit.states import FeatureSequence, FeatureSequenceDataset
from dfckit.states.hmm import _reconstruct_estimator, reconstruct_emission_covariance
from dfckit.storage import FeatureStore

HAS_HMM_EXTRA = (
    importlib.util.find_spec("hmmlearn") is not None
    and importlib.util.find_spec("sklearn") is not None
)


@unittest.skipUnless(HAS_HMM_EXTRA, "requires dfc-kit[hmm]")
class StreamingGaussianHMMTests(unittest.TestCase):
    keys = (("roi-0",), ("roi-1",), ("roi-2",), ("roi-3",))

    def _sequences(self):
        output = []
        state_means = np.asarray([[-2.0, -1.0, 0.5, 1.0], [2.0, 1.0, -0.5, -1.0]])
        latent = np.tile(np.repeat([0, 1], 15), 2)
        for subject_index in range(5):
            rng = np.random.default_rng(400 + subject_index)
            values = state_means[latent] + rng.normal(scale=0.18, size=(len(latent), 4))
            output.append(
                FeatureSequence(
                    values=values,
                    sample_start_indices=np.arange(len(values)),
                    sample_end_indices=np.arange(len(values)),
                    feature_keys=self.keys,
                    subject=f"sub-{subject_index:03d}",
                    session="off",
                    segment_id=0,
                    source_contract="synthetic-streaming-hmm:v1",
                    sample_interval_seconds=0.8,
                )
            )
        output.append(
            FeatureSequence(
                values=[[100.0, 100.0, 100.0, 100.0]],
                sample_start_indices=[100],
                sample_end_indices=[100],
                feature_keys=self.keys,
                subject="sub-000",
                session="off",
                segment_id=1,
                source_contract="synthetic-streaming-hmm:v1",
                sample_interval_seconds=0.8,
            )
        )
        return tuple(output)

    def _store(self, root: Path, chunk_size: int):
        store = FeatureStore.create(
            root,
            feature_keys=self.keys,
            source_contract="synthetic-streaming-hmm:v1",
            sample_interval_seconds=0.8,
        )
        store.append_dataset(FeatureSequenceDataset(self._sequences()), chunk_size=chunk_size)
        return store

    def _fit(self, store):
        return fit_gaussian_hmm_store(
            store,
            subjects=("sub-000", "sub-001", "sub-002"),
            n_states=2,
            seed=17,
            n_pca_components=2,
            covariance_type="diag",
            n_init=2,
            n_iter=60,
            pca_batch_size=17,
            minimum_sequence_length=2,
        )

    def test_fit_omits_short_sequence_and_preserves_gap_bounded_posteriors(self):
        with TemporaryDirectory() as temporary:
            store = self._store(Path(temporary) / "features", chunk_size=7)
            fit = self._fit(store)

            self.assertEqual(fit.model.fit_subjects, ("sub-000", "sub-001", "sub-002"))
            self.assertEqual(fit.model.fit_sample_count, 180)
            self.assertEqual(fit.model.fit_sequence_count, 3)
            self.assertEqual(fit.model.omitted_short_sequence_count, 1)
            self.assertEqual(fit.model.pca_batch_size, 17)
            self.assertEqual(
                fit.model.training_data_fingerprint,
                store.data_fingerprint(
                    subjects=("sub-000", "sub-001", "sub-002"),
                    minimum_sequence_length=2,
                ),
            )
            self.assertEqual(len(fit.states.assignments.sequences), 3)
            self.assertIn("IncrementalPCA", fit.model.implementation)
            self.assertIsNone(fit.model.emission_covariances)
            self.assertEqual(reconstruct_emission_covariance(fit.model, 0).shape, (4, 4))
            for labels, posterior in zip(
                fit.states.assignments.sequences,
                fit.states.posterior_probabilities,
                strict=True,
            ):
                self.assertEqual(labels.labels.shape, (60,))
                self.assertEqual(posterior.shape, (60, 2))
                np.testing.assert_allclose(posterior.sum(axis=1), 1.0)

    def test_heldout_prediction_matches_frozen_preprocessing_and_lengths(self):
        with TemporaryDirectory() as temporary:
            store = self._store(Path(temporary) / "features", chunk_size=5)
            fit = self._fit(store)
            with self.assertRaisesRegex(ValueError, "overlap"):
                predict_gaussian_hmm_store(
                    fit.model,
                    store,
                    subjects=("sub-000",),
                )
            observed = predict_gaussian_hmm_store(
                fit.model,
                store,
                subjects=("sub-003",),
            )

            sequence = store.read_dataset(subjects=("sub-003",)).sequences[0]
            standardized = (
                sequence.values - fit.model.feature_mean
            ) / fit.model.feature_scale
            reduced = (standardized - fit.model.pca_mean) @ fit.model.pca_components.T
            estimator = _reconstruct_estimator(fit.model)
            expected_log_likelihood, expected_posterior = estimator.score_samples(
                reduced,
                [len(reduced)],
            )
            expected_labels = estimator.predict(reduced, [len(reduced)])
            self.assertAlmostEqual(observed.log_likelihood, expected_log_likelihood)
            np.testing.assert_array_equal(
                observed.assignments.sequences[0].labels,
                expected_labels,
            )
            np.testing.assert_allclose(
                observed.posterior_probabilities[0],
                expected_posterior,
            )
            np.testing.assert_array_equal(
                observed.assignments.sequences[0].sample_start_indices,
                sequence.sample_start_indices,
            )

    def test_store_chunk_boundaries_do_not_change_pca_or_hmm_fit(self):
        with TemporaryDirectory() as temporary:
            left = self._store(Path(temporary) / "left", chunk_size=2)
            right = self._store(Path(temporary) / "right", chunk_size=19)
            left_fit = self._fit(left)
            right_fit = self._fit(right)

            for left_values, right_values in (
                (left_fit.model.feature_mean, right_fit.model.feature_mean),
                (left_fit.model.feature_scale, right_fit.model.feature_scale),
                (left_fit.model.pca_components, right_fit.model.pca_components),
                (left_fit.model.start_probabilities, right_fit.model.start_probabilities),
                (left_fit.model.transition_matrix, right_fit.model.transition_matrix),
                (left_fit.model.reduced_means, right_fit.model.reduced_means),
                (left_fit.model.reduced_covariances, right_fit.model.reduced_covariances),
            ):
                np.testing.assert_allclose(left_values, right_values, rtol=1e-11, atol=1e-11)
            self.assertAlmostEqual(
                left_fit.model.log_likelihood,
                right_fit.model.log_likelihood,
                places=9,
            )

    def test_heldout_score_restarts_hmm_at_every_censor_segment(self):
        with TemporaryDirectory() as temporary:
            store = self._store(Path(temporary) / "features", chunk_size=5)
            fit = self._fit(store)
            first = store.read_dataset(subjects=("sub-003",)).sequences[0]
            second = FeatureSequence(
                values=first.values[:12] + 0.03,
                sample_start_indices=np.arange(200, 212),
                sample_end_indices=np.arange(200, 212),
                feature_keys=self.keys,
                subject="sub-003",
                session="off",
                segment_id=1,
                source_contract="synthetic-streaming-hmm:v1",
                sample_interval_seconds=0.8,
            )
            store.append_sequence(second, chunk_size=4)
            with self.assertRaisesRegex(ValueError, "overlap"):
                score_gaussian_hmm_store(
                    fit.model,
                    store,
                    subjects=("sub-000",),
                )
            observed = score_gaussian_hmm_store(
                fit.model,
                store,
                subjects=("sub-003",),
            )
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0].n_sequences, 2)
            self.assertEqual(observed[0].n_samples, 72)

            estimator = _reconstruct_estimator(fit.model)
            expected = 0.0
            for sequence in store.read_dataset(subjects=("sub-003",)).sequences:
                standardized = (
                    sequence.values - fit.model.feature_mean
                ) / fit.model.feature_scale
                reduced = (
                    standardized - fit.model.pca_mean
                ) @ fit.model.pca_components.T
                expected += estimator.score(reduced, [len(reduced)])
            self.assertAlmostEqual(observed[0].log_likelihood, expected)
            self.assertAlmostEqual(
                observed[0].log_likelihood_per_sample,
                expected / 72,
            )

            concatenated = np.concatenate(
                [
                    (
                        (sequence.values - fit.model.feature_mean)
                        / fit.model.feature_scale
                        - fit.model.pca_mean
                    )
                    @ fit.model.pca_components.T
                    for sequence in store.read_dataset(subjects=("sub-003",)).sequences
                ]
            )
            crossed_gap = estimator.score(concatenated, [len(concatenated)])
            self.assertFalse(np.isclose(observed[0].log_likelihood, crossed_gap))


if __name__ == "__main__":
    unittest.main()
