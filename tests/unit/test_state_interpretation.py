import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.states import (
    FeatureSequence,
    FeatureSequenceDataset,
    GaussianHMMStateModel,
    KMeansStateModel,
)
from dfckit.states.interpretation import (
    describe_gaussian_hmm_states,
    describe_kmeans_states,
)
from dfckit.storage import FeatureStore


class StateInterpretationTests(unittest.TestCase):
    def test_hmm_emission_means_are_ranked_in_original_feature_space(self):
        keys = (("visual", "putamen"), ("visual", "motor"))
        sequence = FeatureSequence(
            values=np.asarray([[-1.0, -2.0], [1.0, 2.0], [-1.0, -2.0], [1.0, 2.0]]),
            sample_start_indices=np.arange(4),
            sample_end_indices=np.arange(4),
            feature_keys=keys,
            subject="sub-001",
            session="off",
            segment_id=0,
            source_contract="window-fc:test",
            sample_interval_seconds=1.0,
        )
        model = GaussianHMMStateModel(
            start_probabilities=np.asarray([0.5, 0.5]),
            transition_matrix=np.asarray([[0.8, 0.2], [0.3, 0.7]]),
            reduced_means=np.asarray([[2.0, -1.0], [-2.0, 1.0]]),
            reduced_covariances=np.asarray([np.eye(2), np.eye(2)]),
            emission_means=np.asarray([[2.0, -1.0], [-2.0, 1.0]]),
            emission_covariances=None,
            feature_mean=np.zeros(2),
            feature_scale=np.ones(2),
            pca_mean=np.zeros(2),
            pca_components=np.eye(2),
            pca_explained_variance_ratio=np.asarray([0.6, 0.4]),
            feature_keys=keys,
            source_contract="window-fc:test",
            sample_interval_seconds=1.0,
            n_states=2,
            n_pca_components=2,
            covariance_type="diag",
            seed=1,
            n_init=1,
            n_iter=20,
            tol=1e-3,
            minimum_sequence_length=2,
            pca_batch_size=16,
            selected_initialization=0,
            initialization_seeds=(1,),
            initialization_log_likelihoods=np.asarray([-4.0]),
            iterations=3,
            converged=True,
            log_likelihood=-4.0,
            fit_subjects=("sub-001",),
            fit_sample_count=4,
            fit_sequence_count=1,
            omitted_short_sequence_count=0,
            implementation="synthetic IncrementalPCA",
        )
        with TemporaryDirectory() as temporary:
            store = FeatureStore.create(
                Path(temporary) / "store",
                feature_keys=keys,
                source_contract="window-fc:test",
                sample_interval_seconds=1.0,
            )
            store.append_dataset(FeatureSequenceDataset((sequence,)))
            payload = describe_gaussian_hmm_states(store, model, top_features=1)

        self.assertEqual(payload["model_kind"], "gaussian-hmm-state")
        self.assertIn("emission-mean", payload["ranking"])
        self.assertEqual(
            payload["states"][0]["top_positive_features"][0]["feature"],
            ["visual", "putamen"],
        )

    def test_centroids_are_ranked_against_fitted_store_distribution(self):
        keys = (("visual", "putamen"), ("visual", "motor"), ("putamen", "motor"))
        values = np.asarray(
            [
                [-1.0, -2.0, 0.0],
                [1.0, 2.0, 0.0],
                [-1.0, -2.0, 0.0],
                [1.0, 2.0, 0.0],
            ]
        )
        sequence = FeatureSequence(
            values=values,
            sample_start_indices=np.arange(4),
            sample_end_indices=np.arange(4),
            feature_keys=keys,
            subject="sub-001",
            session="off",
            segment_id=0,
            source_contract="window-fc:test",
            sample_interval_seconds=1.0,
        )
        model = KMeansStateModel(
            centers=np.asarray([[2.0, -1.0, 0.0], [-2.0, 1.0, 0.0]]),
            standardized_centers=np.asarray([[2.0, -1.0, 0.0], [-2.0, 1.0, 0.0]]),
            feature_mean=np.zeros(3),
            feature_scale=np.ones(3),
            feature_keys=keys,
            source_contract="window-fc:test",
            sample_interval_seconds=1.0,
            n_states=2,
            seed=1,
            n_init=2,
            max_iter=10,
            algorithm="lloyd",
            standardize_features=False,
            batch_size=None,
            reassignment_ratio=None,
            init_sample_size=None,
            iterations=2,
            inertia=1.0,
            fit_subjects=("sub-001",),
            fit_sample_count=4,
            implementation="synthetic",
        )
        with TemporaryDirectory() as temporary:
            store = FeatureStore.create(
                Path(temporary) / "store",
                feature_keys=keys,
                source_contract="window-fc:test",
                sample_interval_seconds=1.0,
            )
            store.append_dataset(FeatureSequenceDataset((sequence,)))
            payload = describe_kmeans_states(
                store,
                model,
                top_features=2,
                network_map={"visual": "VIS", "putamen": "STR", "motor": "SMN"},
            )

        self.assertEqual(payload["feature_type"], "edge")
        self.assertEqual(
            payload["states"][0]["top_positive_features"][0]["feature"],
            ["visual", "putamen"],
        )
        self.assertEqual(
            payload["states"][0]["top_negative_features"][0]["feature"],
            ["visual", "motor"],
        )
        self.assertEqual(payload["states"][0]["network_blocks"][0]["n_features"], 1)

    def test_missing_network_roi_is_rejected(self):
        keys = (("a",), ("b",))
        sequence = FeatureSequence(
            values=np.asarray([[-1.0, 0.0], [1.0, 0.0]]),
            sample_start_indices=[0, 1],
            sample_end_indices=[0, 1],
            feature_keys=keys,
            subject="sub-001",
            session="off",
            segment_id=0,
            source_contract="cap:test",
            sample_interval_seconds=1.0,
        )
        model = KMeansStateModel(
            centers=np.asarray([[-1.0, 0.0], [1.0, 0.0]]),
            standardized_centers=np.asarray([[-1.0, 0.0], [1.0, 0.0]]),
            feature_mean=np.zeros(2),
            feature_scale=np.ones(2),
            feature_keys=keys,
            source_contract="cap:test",
            sample_interval_seconds=1.0,
            n_states=2,
            seed=1,
            n_init=2,
            max_iter=10,
            algorithm="lloyd",
            standardize_features=False,
            batch_size=None,
            reassignment_ratio=None,
            init_sample_size=None,
            iterations=2,
            inertia=1.0,
            fit_subjects=("sub-001",),
            fit_sample_count=2,
            implementation="synthetic",
        )
        with TemporaryDirectory() as temporary:
            store = FeatureStore.create(
                Path(temporary) / "store",
                feature_keys=keys,
                source_contract="cap:test",
                sample_interval_seconds=1.0,
            )
            store.append_dataset(FeatureSequenceDataset((sequence,)))
            with self.assertRaisesRegex(ValueError, "missing model ROIs"):
                describe_kmeans_states(store, model, network_map={"a": "A"})


if __name__ == "__main__":
    unittest.main()
