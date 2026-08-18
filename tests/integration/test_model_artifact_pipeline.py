import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.io import load_fitted_model, save_fitted_model
from dfckit.outofcore import (
    fit_incremental_pca_store,
    fit_minibatch_kmeans_store,
    iter_pca_store_chunks,
    predict_kmeans_store,
)
from dfckit.outofcore_hmm import fit_gaussian_hmm_store, predict_gaussian_hmm_store
from dfckit.states import FeatureSequence, FeatureSequenceDataset
from dfckit.storage import FeatureStore

HAS_MODEL_EXTRAS = (
    importlib.util.find_spec("hmmlearn") is not None
    and importlib.util.find_spec("sklearn") is not None
)


@unittest.skipUnless(HAS_MODEL_EXTRAS, "requires dfc-kit[hmm]")
class ModelArtifactPipelineTests(unittest.TestCase):
    def test_store_fitted_models_roundtrip_on_heldout_subject(self):
        keys = tuple((f"feature-{index}",) for index in range(4))
        latent = np.tile(np.repeat([0, 1], 15), 2)
        means = np.asarray([[-2.0, -1.0, 0.5, 1.0], [2.0, 1.0, -0.5, -1.0]])
        sequences = []
        for index in range(4):
            rng = np.random.default_rng(700 + index)
            values = means[latent] + rng.normal(scale=0.2, size=(len(latent), 4))
            sequences.append(
                FeatureSequence(
                    values=values,
                    sample_start_indices=np.arange(len(values)),
                    sample_end_indices=np.arange(len(values)),
                    feature_keys=keys,
                    subject=f"sub-{index:03d}",
                    session="off",
                    segment_id=0,
                    source_contract="model-artifact-integration:v1",
                    sample_interval_seconds=0.8,
                )
            )
        training = ("sub-000", "sub-001", "sub-002")
        heldout = ("sub-003",)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = FeatureStore.create(
                root / "features",
                feature_keys=keys,
                source_contract="model-artifact-integration:v1",
                sample_interval_seconds=0.8,
            )
            store.append_dataset(FeatureSequenceDataset(tuple(sequences)), chunk_size=7)

            kmeans = fit_minibatch_kmeans_store(
                store,
                subjects=training,
                n_states=2,
                seed=17,
                n_init=2,
                max_iter=2,
                batch_size=16,
            )
            restored_kmeans = load_fitted_model(
                save_fitted_model(kmeans.model, root / "kmeans.model")
            )
            expected_labels = predict_kmeans_store(
                kmeans.model, store, subjects=heldout
            ).sequences[0].labels
            observed_labels = predict_kmeans_store(
                restored_kmeans, store, subjects=heldout
            ).sequences[0].labels
            np.testing.assert_array_equal(observed_labels, expected_labels)

            pca = fit_incremental_pca_store(
                store,
                subjects=training,
                n_components=2,
                batch_size=16,
            )
            restored_pca = load_fitted_model(
                save_fitted_model(pca, root / "pca.model")
            )
            expected_chunks = list(iter_pca_store_chunks(pca, store, subjects=heldout))
            observed_chunks = list(
                iter_pca_store_chunks(restored_pca, store, subjects=heldout)
            )
            for observed, expected in zip(observed_chunks, expected_chunks, strict=True):
                np.testing.assert_allclose(observed.values, expected.values)
                np.testing.assert_array_equal(
                    observed.sample_start_indices,
                    expected.sample_start_indices,
                )

            hmm = fit_gaussian_hmm_store(
                store,
                subjects=training,
                n_states=2,
                seed=19,
                n_pca_components=2,
                n_init=1,
                n_iter=20,
                pca_batch_size=16,
            )
            restored_hmm = load_fitted_model(
                save_fitted_model(hmm.model, root / "hmm.model")
            )
            expected_hmm = predict_gaussian_hmm_store(
                hmm.model, store, subjects=heldout
            )
            observed_hmm = predict_gaussian_hmm_store(
                restored_hmm, store, subjects=heldout
            )
            np.testing.assert_array_equal(
                observed_hmm.assignments.sequences[0].labels,
                expected_hmm.assignments.sequences[0].labels,
            )
            np.testing.assert_allclose(
                observed_hmm.posterior_probabilities[0],
                expected_hmm.posterior_probabilities[0],
            )
            self.assertAlmostEqual(observed_hmm.log_likelihood, expected_hmm.log_likelihood)


if __name__ == "__main__":
    unittest.main()
