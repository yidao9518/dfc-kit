import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.outofcore import (
    fit_incremental_pca_store,
    fit_minibatch_kmeans_store,
    iter_pca_store_chunks,
    predict_kmeans_store,
    score_kmeans_store,
)
from dfckit.states import FeatureSequence, FeatureSequenceDataset
from dfckit.storage import FeatureStore

HAS_STATES_EXTRA = importlib.util.find_spec("sklearn") is not None


class StreamingStoreMixin:
    keys = (("feature-0",), ("feature-1",), ("feature-2",))

    def _sequences(self):
        sequences = []
        for subject_index in range(5):
            rng = np.random.default_rng(100 + subject_index)
            center = np.asarray([-3.0, -2.0, -1.0]) if subject_index < 3 else np.asarray(
                [3.0, 2.0, 1.0]
            )
            values = np.vstack(
                [
                    center + rng.normal(scale=0.08, size=(9, 3)),
                    -center + rng.normal(scale=0.08, size=(9, 3)),
                ]
            )
            sequences.append(
                FeatureSequence(
                    values=values,
                    sample_start_indices=np.arange(18) * 2,
                    sample_end_indices=np.arange(18) * 2 + 1,
                    feature_keys=self.keys,
                    subject=f"sub-{subject_index:03d}",
                    session="off",
                    segment_id=0,
                    source_contract="synthetic-streaming-kmeans:v1",
                    sample_interval_seconds=1.6,
                )
            )
        return tuple(sequences)

    def _store(self, root: Path, chunk_size: int):
        store = FeatureStore.create(
            root,
            feature_keys=self.keys,
            source_contract="synthetic-streaming-kmeans:v1",
            sample_interval_seconds=1.6,
        )
        store.append_dataset(FeatureSequenceDataset(self._sequences()), chunk_size=chunk_size)
        return store


@unittest.skipUnless(HAS_STATES_EXTRA, "requires dfc-kit[states]")
class StreamingKMeansTests(StreamingStoreMixin, unittest.TestCase):

    def test_standardization_is_streaming_and_fit_is_deterministic(self):
        with TemporaryDirectory() as temporary:
            store = self._store(Path(temporary) / "features", chunk_size=4)
            first = fit_minibatch_kmeans_store(
                store,
                subjects=("sub-000", "sub-001", "sub-002"),
                n_states=2,
                seed=12,
                n_init=3,
                max_iter=4,
                batch_size=7,
            )
            second = fit_minibatch_kmeans_store(
                store,
                subjects=("sub-000", "sub-001", "sub-002"),
                n_states=2,
                seed=12,
                n_init=3,
                max_iter=4,
                batch_size=7,
            )
            expected = np.concatenate(
                [sequence.values for sequence in store.read_dataset(
                    subjects=("sub-000", "sub-001", "sub-002")
                ).sequences],
                axis=0,
            )
            np.testing.assert_allclose(first.model.feature_mean, expected.mean(axis=0))
            np.testing.assert_allclose(first.model.feature_scale, expected.std(axis=0))
            np.testing.assert_allclose(first.model.standardized_centers, second.model.standardized_centers)
            np.testing.assert_array_equal(
                np.concatenate([item.labels for item in first.assignments.sequences]),
                np.concatenate([item.labels for item in second.assignments.sequences]),
            )
            self.assertIn("out-of-core partial_fit", first.model.implementation)

    def test_chunk_boundaries_do_not_change_batches_or_result(self):
        with TemporaryDirectory() as temporary:
            left = self._store(Path(temporary) / "left", chunk_size=2)
            right = self._store(Path(temporary) / "right", chunk_size=11)
            left_fit = fit_minibatch_kmeans_store(
                left, n_states=2, seed=9, n_init=2, max_iter=3, batch_size=8
            )
            right_fit = fit_minibatch_kmeans_store(
                right, n_states=2, seed=9, n_init=2, max_iter=3, batch_size=8
            )
            np.testing.assert_allclose(
                left_fit.model.standardized_centers,
                right_fit.model.standardized_centers,
                rtol=1e-12,
                atol=1e-12,
            )
            self.assertAlmostEqual(left_fit.model.inertia, right_fit.model.inertia, places=10)

    def test_prediction_matches_direct_center_distance_and_rejects_overlap(self):
        with TemporaryDirectory() as temporary:
            store = self._store(Path(temporary) / "features", chunk_size=3)
            fit = fit_minibatch_kmeans_store(
                store,
                subjects=("sub-000", "sub-001", "sub-002"),
                n_states=2,
                seed=4,
                n_init=2,
                max_iter=3,
                batch_size=6,
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                predict_kmeans_store(fit.model, store, subjects=("sub-001",))
            observed = predict_kmeans_store(fit.model, store, subjects=("sub-003",))
            held_out = store.read_dataset(subjects=("sub-003",))
            expected = []
            for sequence in held_out.sequences:
                standardized = (
                    sequence.values - fit.model.feature_mean
                ) / fit.model.feature_scale
                distance = np.square(
                    standardized[:, None, :] - fit.model.standardized_centers[None, :, :]
                ).sum(axis=2)
                expected.append(np.argmin(distance, axis=1))
            np.testing.assert_array_equal(observed.sequences[0].labels, expected[0])
            np.testing.assert_array_equal(
                observed.sequences[0].sample_start_indices,
                held_out.sequences[0].sample_start_indices,
            )

    def test_heldout_score_matches_direct_distance_and_is_chunk_invariant(self):
        with TemporaryDirectory() as temporary:
            left = self._store(Path(temporary) / "left", chunk_size=2)
            right = self._store(Path(temporary) / "right", chunk_size=11)
            fit = fit_minibatch_kmeans_store(
                left,
                subjects=("sub-000", "sub-001", "sub-002"),
                n_states=2,
                seed=4,
                n_init=2,
                max_iter=3,
                batch_size=6,
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                score_kmeans_store(fit.model, left, subjects=("sub-001",))
            left_score = score_kmeans_store(
                fit.model,
                left,
                subjects=("sub-003", "sub-004"),
            )
            right_score = score_kmeans_store(
                fit.model,
                right,
                subjects=("sub-003", "sub-004"),
            )
            self.assertEqual(len(left_score), 2)
            self.assertEqual(left_score, right_score)
            held_out = left.read_dataset(subjects=("sub-003", "sub-004"))
            for observed, sequence in zip(left_score, held_out.sequences, strict=True):
                standardized = (
                    sequence.values - fit.model.feature_mean
                ) / fit.model.feature_scale
                squared_distances = np.square(
                    standardized[:, None, :]
                    - fit.model.standardized_centers[None, :, :]
                ).sum(axis=2)
                expected = float(np.min(squared_distances, axis=1).sum())
                self.assertEqual(observed.n_samples, len(sequence.values))
                self.assertEqual(observed.n_sequences, 1)
                self.assertAlmostEqual(observed.total_squared_distance, expected)
                self.assertAlmostEqual(
                    observed.mean_squared_distance,
                    expected / len(sequence.values),
                )


@unittest.skipUnless(HAS_STATES_EXTRA, "requires dfc-kit[states]")
class StreamingPCATests(StreamingStoreMixin, unittest.TestCase):
    def test_streaming_scaler_and_transform_match_direct_calculation(self):
        with TemporaryDirectory() as temporary:
            store = self._store(Path(temporary) / "features", chunk_size=4)
            model = fit_incremental_pca_store(
                store,
                subjects=("sub-000", "sub-001", "sub-002"),
                n_components=2,
                batch_size=13,
            )
            training = store.read_dataset(subjects=("sub-000", "sub-001", "sub-002"))
            pooled = np.concatenate([sequence.values for sequence in training.sequences])
            np.testing.assert_allclose(model.feature_mean, pooled.mean(axis=0))
            np.testing.assert_allclose(model.feature_scale, pooled.std(axis=0))
            self.assertEqual(model.fit_sample_count, len(pooled))
            self.assertEqual(model.fit_sequence_count, 3)

            with self.assertRaisesRegex(ValueError, "overlap"):
                list(iter_pca_store_chunks(model, store, subjects=("sub-000",)))
            chunks = list(iter_pca_store_chunks(model, store, subjects=("sub-003",)))
            observed = np.concatenate([chunk.values for chunk in chunks])
            held_out = store.read_dataset(subjects=("sub-003",)).sequences[0]
            standardized = (held_out.values - model.feature_mean) / model.feature_scale
            expected = (standardized - model.pca_mean) @ model.pca_components.T
            np.testing.assert_allclose(observed, expected)
            np.testing.assert_array_equal(
                np.concatenate([chunk.sample_start_indices for chunk in chunks]),
                held_out.sample_start_indices,
            )

    def test_chunk_boundaries_do_not_change_incremental_pca(self):
        with TemporaryDirectory() as temporary:
            left = self._store(Path(temporary) / "left", chunk_size=2)
            right = self._store(Path(temporary) / "right", chunk_size=11)
            left_model = fit_incremental_pca_store(
                left, n_components=2, batch_size=8
            )
            right_model = fit_incremental_pca_store(
                right, n_components=2, batch_size=8
            )
            np.testing.assert_allclose(left_model.feature_mean, right_model.feature_mean)
            np.testing.assert_allclose(left_model.feature_scale, right_model.feature_scale)
            np.testing.assert_allclose(
                left_model.pca_components,
                right_model.pca_components,
                rtol=1e-12,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                left_model.explained_variance_ratio,
                right_model.explained_variance_ratio,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_minimum_sequence_length_excludes_short_rows_from_scaler(self):
        with TemporaryDirectory() as temporary:
            store = self._store(Path(temporary) / "features", chunk_size=5)
            short = FeatureSequence(
                values=[[1000.0, 1000.0, 1000.0]],
                sample_start_indices=[100],
                sample_end_indices=[100],
                feature_keys=self.keys,
                subject="sub-short",
                session="off",
                segment_id=0,
                source_contract="synthetic-streaming-kmeans:v1",
                sample_interval_seconds=1.6,
            )
            store.append_sequence(short)
            model = fit_incremental_pca_store(
                store,
                n_components=2,
                batch_size=9,
                minimum_sequence_length=2,
            )
            expected = np.concatenate([sequence.values for sequence in self._sequences()])
            np.testing.assert_allclose(model.feature_mean, expected.mean(axis=0))
            self.assertEqual(model.fit_sequence_count, 5)

    def test_constant_feature_is_rejected_for_streaming_pca(self):
        with TemporaryDirectory() as temporary:
            sequence = FeatureSequence(
                values=[[0.0, 1.0, 3.0], [1.0, 1.0, 2.0], [2.0, 1.0, 1.0]],
                sample_start_indices=[0, 1, 2],
                sample_end_indices=[0, 1, 2],
                feature_keys=self.keys,
                subject="sub-constant",
                session="off",
                segment_id=0,
                source_contract="synthetic-streaming-kmeans:v1",
                sample_interval_seconds=1.6,
            )
            store = FeatureStore.create(
                Path(temporary) / "features",
                feature_keys=self.keys,
                source_contract="synthetic-streaming-kmeans:v1",
                sample_interval_seconds=1.6,
            )
            store.append_sequence(sequence, chunk_size=2)
            with self.assertRaisesRegex(ValueError, r"undefined for indices \[1\]"):
                fit_incremental_pca_store(store, n_components=2, batch_size=2)


if __name__ == "__main__":
    unittest.main()
