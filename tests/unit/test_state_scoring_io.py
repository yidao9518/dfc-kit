import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dfckit.artifacts import (
    load_state_model_scores,
    state_model_scores_payload,
    write_state_model_scores,
)
from dfckit.states import RunGaussianHMMScore, RunKMeansScore


def _metadata(model_kind: str = "kmeans-state") -> dict[str, object]:
    return {
        "model_kind": model_kind,
        "model_seed": 17,
        "n_states": 4,
        "fit_subjects": ("sub-001", "sub-002"),
        "source_contract": "state-scoring-test:v1",
        "sample_interval_seconds": 0.8,
        "minimum_sequence_length": 1,
        "omitted_short_sequence_count": 0,
        "allow_fit_subjects": False,
        "model_specification": (
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
        ),
    }


class StateScoringIOTests(unittest.TestCase):
    def test_kmeans_payload_uses_sample_weighted_summary(self):
        scores = (
            RunKMeansScore("sub-010", "off", "run-1", 4, 1, 8.0, 2.0),
            RunKMeansScore("sub-011", "off", "run-1", 6, 2, 6.0, 1.0),
        )
        payload = state_model_scores_payload(scores, **_metadata())
        self.assertEqual(payload["summary"]["mean_squared_distance"], 1.4)

    def test_hmm_payload_preserves_finite_likelihoods(self):
        scores = (
            RunGaussianHMMScore("sub-010", None, None, 5, 2, -10.0, -2.0),
            RunGaussianHMMScore("sub-011", None, None, 5, 1, -15.0, -3.0),
        )
        payload = state_model_scores_payload(scores, **_metadata("gaussian-hmm-state"))
        self.assertEqual(payload["summary"]["log_likelihood_per_sample"], -2.5)

    def test_roundtrip_and_duplicate_run_rejection(self):
        score = RunKMeansScore("sub-010", None, None, 4, 1, 8.0, 2.0)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "scores.json"
            write_state_model_scores((score,), output, **_metadata())
            restored = load_state_model_scores(output)
            self.assertEqual(restored.subjects, ("sub-010",))
            with self.assertRaises(FileExistsError):
                write_state_model_scores((score,), output, **_metadata())
        with self.assertRaisesRegex(ValueError, "duplicate run"):
            state_model_scores_payload((score, score), **_metadata())

    def test_invalid_score_type_is_rejected(self):
        hmm = (RunGaussianHMMScore("sub-010", None, None, 4, 1, -8.0, -2.0),)
        with self.assertRaisesRegex(TypeError, "does not match"):
            state_model_scores_payload(hmm, **_metadata())


if __name__ == "__main__":
    unittest.main()
