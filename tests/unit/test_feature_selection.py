import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dfckit.states import FeatureSequence, FeatureSequenceDataset
from dfckit.states.streaming import fit_minibatch_kmeans_store
from dfckit.storage import FeatureStore, FeatureStoreView


def make_dataset() -> FeatureSequenceDataset:
    values = np.asarray(
        [
            [-2.0, 10.0, -1.8],
            [-2.2, 11.0, -2.0],
            [-1.8, 12.0, -2.1],
            [-2.1, 13.0, -1.9],
            [2.0, 20.0, 1.8],
            [2.2, 21.0, 2.0],
            [1.8, 22.0, 2.1],
            [2.1, 23.0, 1.9],
        ]
    )
    sequence = FeatureSequence(
        values=values,
        sample_start_indices=np.arange(len(values)),
        sample_end_indices=np.arange(len(values)),
        feature_keys=(("V1_L", "PUT_L"), ("unused",), ("V1_R", "PUT_R")),
        subject="sub-001",
        session="off",
        segment_id=0,
        source_contract="synthetic-window-fc",
        sample_interval_seconds=7.5,
    )
    return FeatureSequenceDataset((sequence,))


class InMemoryFeatureSelectionTests(unittest.TestCase):
    def test_named_selection_preserves_requested_order_and_metadata(self):
        dataset = make_dataset()
        selected = dataset.select_features(feature_keys=(("V1_R", "PUT_R"), ("V1_L", "PUT_L")))

        self.assertEqual(
            selected.feature_keys,
            (("V1_R", "PUT_R"), ("V1_L", "PUT_L")),
        )
        np.testing.assert_array_equal(
            selected.sequences[0].values,
            dataset.sequences[0].values[:, [2, 0]],
        )
        self.assertEqual(selected.sequences[0].subject, "sub-001")
        self.assertEqual(selected.sequences[0].sample_interval_seconds, 7.5)

    def test_boolean_mask_preserves_source_order(self):
        selected = make_dataset().select_features(feature_mask=[True, False, True])
        self.assertEqual(
            selected.feature_keys,
            (("V1_L", "PUT_L"), ("V1_R", "PUT_R")),
        )

    def test_invalid_or_ambiguous_selection_is_rejected(self):
        dataset = make_dataset()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            dataset.select_features()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            dataset.select_features(
                feature_keys=(("unused",),),
                feature_mask=[False, True, False],
            )
        with self.assertRaisesRegex(ValueError, "Boolean vector"):
            dataset.select_features(feature_mask=[1, 0, 1])
        with self.assertRaisesRegex(ValueError, "at least one"):
            dataset.select_features(feature_mask=[False, False, False])
        with self.assertRaisesRegex(KeyError, "absent"):
            dataset.select_features(feature_keys=(("missing",),))


@unittest.skipUnless(importlib.util.find_spec("sklearn") is not None, "requires dfc-kit[states]")
class StoredFeatureSelectionTests(unittest.TestCase):
    def test_store_view_supports_streaming_kmeans_without_copying_a_store(self):
        dataset = make_dataset()
        with tempfile.TemporaryDirectory() as temporary:
            store = FeatureStore.create(
                Path(temporary) / "features.store",
                feature_keys=dataset.feature_keys,
                source_contract=dataset.source_contract,
                sample_interval_seconds=dataset.sample_interval_seconds,
                dtype="float64",
            )
            store.append_dataset(dataset, chunk_size=3)
            selected = store.select_features(feature_mask=[True, False, True])

            self.assertIsInstance(selected, FeatureStoreView)
            self.assertEqual(selected.n_features, 2)
            self.assertEqual(selected.n_samples, store.n_samples)
            np.testing.assert_array_equal(
                selected.read_dataset().sequences[0].values,
                dataset.sequences[0].values[:, [0, 2]],
            )
            fit = fit_minibatch_kmeans_store(
                selected,
                n_states=2,
                seed=7,
                n_init=1,
                max_iter=2,
                batch_size=4,
                convergence_tol=0.0,
            )
            self.assertEqual(fit.model.feature_keys, selected.feature_keys)
            self.assertEqual(fit.model.fit_sample_count, dataset.n_samples)
            self.assertEqual(len(fit.assignments.sequences), 1)


if __name__ == "__main__":
    unittest.main()
