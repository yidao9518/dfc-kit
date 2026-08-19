import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.state_interpretation import describe_kmeans_states
from dfckit.states import FeatureSequence, FeatureSequenceDataset, KMeansStateModel
from dfckit.storage import FeatureStore


class StateInterpretationTests(unittest.TestCase):
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
            training_data_fingerprint=None,
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
        self.assertEqual(len(payload["model_fingerprint"]), 64)
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
            training_data_fingerprint=None,
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
