import importlib.util
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.artifacts import load_fitted_model, save_fitted_model
from dfckit.states import (
    FeatureSequence,
    FeatureSequenceDataset,
    GaussianHMMStateModel,
    KMeansStateModel,
    predict_gaussian_hmm_states,
    predict_kmeans_states,
)
from dfckit.states.streaming import StreamingPCAModel

HAS_HMM_EXTRA = importlib.util.find_spec("hmmlearn") is not None


def kmeans_model() -> KMeansStateModel:
    return KMeansStateModel(
        centers=np.asarray([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]),
        standardized_centers=np.asarray([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]),
        feature_mean=np.zeros(3),
        feature_scale=np.ones(3),
        feature_keys=(("a",), ("b",), ("c",)),
        source_contract="model-io-test:v1",
        sample_interval_seconds=0.8,
        n_states=2,
        seed=11,
        n_init=5,
        max_iter=100,
        algorithm="lloyd",
        standardize_features=True,
        batch_size=None,
        reassignment_ratio=None,
        init_sample_size=None,
        iterations=4,
        inertia=1.5,
        fit_subjects=("sub-train",),
        fit_sample_count=20,
        implementation="synthetic KMeans",
    )


def pca_model() -> StreamingPCAModel:
    return StreamingPCAModel(
        feature_mean=np.asarray([1.0, 2.0, 3.0]),
        feature_scale=np.asarray([2.0, 3.0, 4.0]),
        pca_mean=np.zeros(3),
        pca_components=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        explained_variance_ratio=np.asarray([0.6, 0.3]),
        feature_keys=(("a",), ("b",), ("c",)),
        source_contract="model-io-test:v1",
        sample_interval_seconds=0.8,
        n_components=2,
        standardize_features=True,
        batch_size=8,
        fit_subjects=("sub-train",),
        fit_sample_count=20,
        fit_sequence_count=2,
        implementation="synthetic IncrementalPCA",
    )


def hmm_model() -> GaussianHMMStateModel:
    return GaussianHMMStateModel(
        start_probabilities=np.asarray([0.6, 0.4]),
        transition_matrix=np.asarray([[0.9, 0.1], [0.2, 0.8]]),
        reduced_means=np.asarray([[-1.0, 0.5], [1.0, -0.5]]),
        reduced_covariances=np.asarray([np.eye(2), np.eye(2) * 1.5]),
        emission_means=np.asarray([[-1.0, 0.5, 0.0], [1.0, -0.5, 0.0]]),
        emission_covariances=None,
        feature_mean=np.zeros(3),
        feature_scale=np.ones(3),
        pca_mean=np.zeros(3),
        pca_components=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        pca_explained_variance_ratio=np.asarray([0.6, 0.3]),
        feature_keys=(("a",), ("b",), ("c",)),
        source_contract="model-io-test:v1",
        sample_interval_seconds=0.8,
        n_states=2,
        n_pca_components=2,
        covariance_type="diag",
        seed=13,
        n_init=2,
        n_iter=50,
        tol=1e-3,
        minimum_sequence_length=2,
        pca_batch_size=None,
        selected_initialization=0,
        initialization_seeds=(13, 14),
        initialization_log_likelihoods=np.asarray([-10.0, -11.0]),
        iterations=7,
        converged=True,
        log_likelihood=-10.0,
        fit_subjects=("sub-train",),
        fit_sample_count=20,
        fit_sequence_count=2,
        omitted_short_sequence_count=1,
        implementation="synthetic GaussianHMM",
    )


def heldout_dataset() -> FeatureSequenceDataset:
    sequence = FeatureSequence(
        values=np.asarray([[-1.1, 0.4, 0.0], [-0.9, 0.6, 0.0], [1.1, -0.4, 0.0]]),
        sample_start_indices=[0, 1, 2],
        sample_end_indices=[0, 1, 2],
        feature_keys=(("a",), ("b",), ("c",)),
        subject="sub-heldout",
        session="off",
        segment_id=0,
        source_contract="model-io-test:v1",
        sample_interval_seconds=0.8,
    )
    return FeatureSequenceDataset((sequence,))


class FittedModelIOTests(unittest.TestCase):
    def test_artifact_version_is_strict(self):
        with TemporaryDirectory() as temporary:
            target = save_fitted_model(kmeans_model(), Path(temporary) / "model")
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["format_version"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported.*format version"):
                load_fitted_model(target)

    def test_kmeans_roundtrip_preserves_predictions_and_readonly_arrays(self):
        with TemporaryDirectory() as temporary:
            model = kmeans_model()
            target = save_fitted_model(model, Path(temporary) / "kmeans.model")
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            loaded = load_fitted_model(target)

            self.assertEqual(manifest["format_version"], 4)
            self.assertIn("init_sample_size", manifest["metadata"])
            self.assertIsInstance(loaded, KMeansStateModel)
            self.assertEqual(loaded.fit_subjects, model.fit_subjects)
            self.assertFalse(loaded.centers.flags.writeable)
            before = predict_kmeans_states(model, heldout_dataset())
            after = predict_kmeans_states(loaded, heldout_dataset())
            np.testing.assert_array_equal(
                before.sequences[0].labels,
                after.sequences[0].labels,
            )

    def test_pca_kmeans_roundtrip_preserves_frozen_transform(self):
        model = replace(
            kmeans_model(),
            clustering_centers=np.asarray([[-1.0, 1.0], [1.0, -1.0]]),
            pca_mean=np.zeros(3),
            pca_components=np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            pca_explained_variance_ratio=np.asarray([0.6, 0.3]),
            n_pca_components=2,
        )
        with TemporaryDirectory() as temporary:
            loaded = load_fitted_model(
                save_fitted_model(model, Path(temporary) / "pca-kmeans.model")
            )

        self.assertEqual(loaded.n_pca_components, 2)
        np.testing.assert_array_equal(
            loaded.pca_components,
            model.pca_components,
        )
        before = predict_kmeans_states(model, heldout_dataset())
        after = predict_kmeans_states(loaded, heldout_dataset())
        np.testing.assert_array_equal(
            before.sequences[0].labels,
            after.sequences[0].labels,
        )

    def test_streaming_pca_roundtrip_preserves_all_arrays(self):
        with TemporaryDirectory() as temporary:
            model = pca_model()
            target = save_fitted_model(model, Path(temporary) / "pca.model")
            loaded = load_fitted_model(target)

            self.assertIsInstance(loaded, StreamingPCAModel)
            self.assertEqual(loaded.fit_sequence_count, 2)
            for name in (
                "feature_mean",
                "feature_scale",
                "pca_mean",
                "pca_components",
                "explained_variance_ratio",
            ):
                np.testing.assert_array_equal(getattr(loaded, name), getattr(model, name))
                self.assertFalse(getattr(loaded, name).flags.writeable)

    @unittest.skipUnless(HAS_HMM_EXTRA, "requires dfc-kit[hmm]")
    def test_hmm_roundtrip_preserves_frozen_decode_and_optional_covariance(self):
        with TemporaryDirectory() as temporary:
            model = hmm_model()
            target = save_fitted_model(model, Path(temporary) / "hmm.model")
            loaded = load_fitted_model(target)

            self.assertIsInstance(loaded, GaussianHMMStateModel)
            self.assertIsNone(loaded.emission_covariances)
            before = predict_gaussian_hmm_states(model, heldout_dataset())
            after = predict_gaussian_hmm_states(loaded, heldout_dataset())
            np.testing.assert_array_equal(
                before.assignments.sequences[0].labels,
                after.assignments.sequences[0].labels,
            )
            np.testing.assert_allclose(
                before.posterior_probabilities[0],
                after.posterior_probabilities[0],
            )
            self.assertAlmostEqual(before.log_likelihood, after.log_likelihood)

    def test_dense_hmm_covariances_roundtrip_when_present(self):
        model = replace(
            hmm_model(),
            covariance_type="full",
            emission_covariances=np.asarray([np.eye(3), np.eye(3) * 1.5]),
        )
        with TemporaryDirectory() as temporary:
            target = save_fitted_model(model, Path(temporary) / "hmm.model")
            loaded = load_fitted_model(target)

            self.assertIsInstance(loaded, GaussianHMMStateModel)
            np.testing.assert_array_equal(
                loaded.emission_covariances,
                model.emission_covariances,
            )
            self.assertFalse(loaded.emission_covariances.flags.writeable)

    def test_existing_target_is_not_overwritten(self):
        with TemporaryDirectory() as temporary:
            target = Path(temporary) / "model"
            save_fitted_model(kmeans_model(), target)
            manifest_before = (target / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                save_fitted_model(pca_model(), target)
            self.assertEqual((target / "manifest.json").read_bytes(), manifest_before)

    def test_unsupported_model_type_is_rejected(self):
        with TemporaryDirectory() as temporary, self.assertRaisesRegex(
            TypeError, "unsupported fitted model type"
        ):
            save_fitted_model(object(), Path(temporary) / "model")

    def test_unknown_version_and_manifest_archive_mismatch_are_rejected(self):
        with TemporaryDirectory() as temporary:
            target = save_fitted_model(kmeans_model(), Path(temporary) / "model")
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["format_version"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported.*format version"):
                load_fitted_model(target)

            manifest["format_version"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported.*format version"):
                load_fitted_model(target)

    def test_invalid_shape_and_missing_metadata_are_rejected(self):
        with TemporaryDirectory() as temporary:
            target = save_fitted_model(kmeans_model(), Path(temporary) / "model")
            arrays_path = target / "arrays.npz"
            with np.load(arrays_path, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
            arrays["centers"] = arrays["centers"][:, :2]
            np.savez(arrays_path, **arrays)
            with self.assertRaisesRegex(ValueError, "centers have an invalid shape"):
                load_fitted_model(target)

            arrays["centers"] = kmeans_model().centers
            np.savez(arrays_path, **arrays)
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["metadata"]["fit_subjects"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata fields"):
                load_fitted_model(target)

    def test_object_array_is_never_unpickled(self):
        with TemporaryDirectory() as temporary:
            target = save_fitted_model(kmeans_model(), Path(temporary) / "model")
            arrays_path = target / "arrays.npz"
            with np.load(arrays_path, allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
            arrays["centers"] = np.asarray([[object()]], dtype=object)
            np.savez(arrays_path, **arrays)

            with self.assertRaisesRegex(ValueError, "allow_pickle=False"):
                load_fitted_model(target)


if __name__ == "__main__":
    unittest.main()
