import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dfckit.io import (
    load_state_model_scores,
    state_model_scores_payload,
    write_state_model_scores,
)
from dfckit.states import RunGaussianHMMScore, RunKMeansScore


def _metadata(model_kind: str = "kmeans-state") -> dict[str, object]:
    metadata = {
        "model_kind": model_kind,
        "model_fingerprint": "a" * 64,
        "model_seed": 17,
        "n_states": 4,
        "fit_subjects": ("sub-001", "sub-002"),
        "training_data_fingerprint": "b" * 64,
        "evaluation_data_fingerprint": "c" * 64,
        "feature_contract_fingerprint": "d" * 64,
        "source_contract": "state-scoring-test:v1",
        "sample_interval_seconds": 0.8,
        "minimum_sequence_length": 1,
        "omitted_short_sequence_count": 0,
        "allow_fit_subjects": False,
    }
    metadata["model_specification"] = (
        {
            "algorithm": "minibatch",
            "batch_size": 4096,
            "implementation": "scikit-learn test MiniBatchKMeans",
            "init_sample_size": 1000,
            "max_iter": 10,
            "n_init": 10,
            "reassignment_ratio": 0.01,
            "standardize_features": True,
        }
        if model_kind == "kmeans-state"
        else {
            "covariance_type": "diag",
            "implementation": "hmmlearn test; scikit-learn test IncrementalPCA",
            "minimum_sequence_length": 1,
            "n_init": 3,
            "n_iter": 200,
            "n_pca_components": 2,
            "pca_batch_size": 4096,
            "tol": 0.001,
        }
    )
    return metadata


class StateScoringIOTests(unittest.TestCase):
    def test_kmeans_payload_uses_sample_weighted_summary(self):
        scores = (
            RunKMeansScore("sub-010", "off", "run-1", 4, 1, 8.0, 2.0),
            RunKMeansScore("sub-011", "off", "run-1", 6, 2, 6.0, 1.0),
        )
        payload = state_model_scores_payload(scores, **_metadata())
        self.assertEqual(payload["format"], "dfckit-state-model-scores")
        self.assertEqual(payload["selection_metric"], "mean_squared_distance")
        self.assertEqual(payload["n_samples"], 10)
        self.assertEqual(payload["n_sequences"], 3)
        self.assertEqual(payload["summary"]["total_squared_distance"], 14.0)
        self.assertEqual(payload["summary"]["mean_squared_distance"], 1.4)
        self.assertEqual(payload["evaluation_data_fingerprint"], "c" * 64)
        self.assertEqual(payload["minimum_sequence_length"], 1)
        self.assertEqual(payload["omitted_short_sequence_count"], 0)
        self.assertEqual(payload["format_version"], 2)
        self.assertEqual(payload["model_specification"]["n_init"], 10)

    def test_hmm_payload_preserves_finite_likelihoods(self):
        scores = (
            RunGaussianHMMScore("sub-010", None, None, 5, 2, -10.0, -2.0),
            RunGaussianHMMScore("sub-011", None, None, 5, 1, -15.0, -3.0),
        )
        payload = state_model_scores_payload(
            scores,
            **_metadata("gaussian-hmm-state"),
        )
        self.assertEqual(payload["selection_metric"], "log_likelihood_per_sample")
        self.assertEqual(payload["summary"]["log_likelihood"], -25.0)
        self.assertEqual(payload["summary"]["log_likelihood_per_sample"], -2.5)

    def test_writer_is_strict_atomic_and_refuses_overwrite(self):
        scores = (RunKMeansScore("sub-010", None, None, 4, 1, 8.0, 2.0),)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "scores.json"
            write_state_model_scores(scores, output, **_metadata())
            text = output.read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            self.assertEqual(json.loads(text)["n_runs"], 1)
            restored = load_state_model_scores(output)
            self.assertEqual(restored.format_version, 2)
            self.assertEqual(restored.subjects, ("sub-010",))
            self.assertEqual(restored.n_samples, 4)
            self.assertEqual(restored.model_specification["algorithm"], "minibatch")
            with self.assertRaises(FileExistsError):
                write_state_model_scores(scores, output, **_metadata())

    def test_invalid_score_type_fingerprint_and_duplicate_run_are_rejected(self):
        hmm = (RunGaussianHMMScore("sub-010", None, None, 4, 1, -8.0, -2.0),)
        with self.assertRaisesRegex(TypeError, "does not match"):
            state_model_scores_payload(hmm, **_metadata())

        invalid = _metadata()
        invalid["evaluation_data_fingerprint"] = "not-a-fingerprint"
        score = RunKMeansScore("sub-010", None, None, 4, 1, 8.0, 2.0)
        with self.assertRaisesRegex(ValueError, "evaluation_data_fingerprint"):
            state_model_scores_payload((score,), **invalid)
        with self.assertRaisesRegex(ValueError, "duplicate run"):
            state_model_scores_payload((score, score), **_metadata())

    def test_v1_remains_readable_but_has_no_model_specification(self):
        scores = (RunKMeansScore("sub-010", None, None, 4, 1, 8.0, 2.0),)
        metadata = _metadata()
        del metadata["model_specification"]
        del metadata["feature_contract_fingerprint"]
        payload = state_model_scores_payload(scores, **metadata)
        self.assertEqual(payload["format_version"], 1)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            restored = load_state_model_scores(path)
            self.assertEqual(restored.format_version, 1)
            self.assertIsNone(restored.model_specification)

    def test_loader_rejects_tampered_summary_unknown_fields_and_nan(self):
        scores = (RunKMeansScore("sub-010", None, None, 4, 1, 8.0, 2.0),)
        payload = state_model_scores_payload(scores, **_metadata())
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            changed = json.loads(json.dumps(payload))
            changed["summary"]["mean_squared_distance"] = 999.0
            path = root / "changed.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_state_model_scores(path)

            unexpected = json.loads(json.dumps(payload))
            unexpected["unexpected"] = True
            path = root / "unexpected.json"
            path.write_text(json.dumps(unexpected), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields"):
                load_state_model_scores(path)

            path = root / "nan.json"
            path.write_text(json.dumps(payload).replace("2.0", "NaN", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-standard JSON"):
                load_state_model_scores(path)

    def test_fit_evaluation_overlap_and_bad_model_specification_are_rejected(self):
        score = RunKMeansScore("sub-001", None, None, 4, 1, 8.0, 2.0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            state_model_scores_payload((score,), **_metadata())

        invalid = _metadata()
        del invalid["model_specification"]["n_init"]
        with self.assertRaisesRegex(ValueError, "fields"):
            state_model_scores_payload(
                (RunKMeansScore("sub-010", None, None, 4, 1, 8.0, 2.0),),
                **invalid,
            )


if __name__ == "__main__":
    unittest.main()
