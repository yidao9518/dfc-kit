import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.io import (
    StatePredictions,
    load_state_predictions,
    save_state_predictions,
    write_state_metrics,
)
from dfckit.states import StateAssignments, StateLabelSequence


def _assignments() -> StateAssignments:
    sequences = (
        StateLabelSequence(
            labels=[0, 0, 1, 1],
            sample_start_indices=[0, 1, 2, 3],
            sample_end_indices=[0, 1, 2, 3],
            subject="sub-001",
            session="off",
            acquisition_id="sub-001_ses-off_task-rest_run-1",
            segment_id=0,
        ),
        StateLabelSequence(
            labels=[1, 0, 0],
            sample_start_indices=[10, 11, 12],
            sample_end_indices=[10, 11, 12],
            subject="sub-002",
            session="on",
            acquisition_id="sub-002_ses-on_task-rest_run-1",
            segment_id=2,
        ),
    )
    return StateAssignments(
        sequences=sequences,
        n_states=2,
        source_contract="state-results-test:v1",
        sample_interval_seconds=0.8,
    )


class StateResultPersistenceTests(unittest.TestCase):
    def test_kmeans_roundtrip_and_json_metrics_have_no_nan(self):
        predictions = StatePredictions(
            assignments=_assignments(),
            model_kind="kmeans-state",
            model_seed=17,
            model_fingerprint="a" * 64,
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = save_state_predictions(predictions, root / "labels")
            restored = load_state_predictions(artifact)
            self.assertEqual(restored.model_kind, "kmeans-state")
            self.assertEqual(restored.assignments.sequences[1].acquisition_id, "sub-002_ses-on_task-rest_run-1")
            np.testing.assert_array_equal(
                restored.assignments.sequences[0].labels,
                [0, 0, 1, 1],
            )
            metrics_path = write_state_metrics(restored, root / "metrics.json")
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["n_runs"], 2)
            self.assertNotIn("NaN", metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["runs"][0]["n_switches"], 1)

    def test_hmm_roundtrip_validates_posterior_and_likelihood(self):
        assignments = _assignments()
        posterior = (
            np.asarray(
                [
                    [0.9, 0.1],
                    [0.8, 0.2],
                    [0.2, 0.8],
                    [0.1, 0.9],
                ]
            ),
            np.asarray([[0.2, 0.8], [0.7, 0.3], [0.6, 0.4]]),
        )
        predictions = StatePredictions(
            assignments=assignments,
            model_kind="gaussian-hmm-state",
            model_seed=19,
            model_fingerprint="b" * 64,
            posterior_probabilities=posterior,
            log_likelihood=-12.5,
        )
        with TemporaryDirectory() as temporary:
            restored = load_state_predictions(
                save_state_predictions(predictions, Path(temporary) / "hmm-labels")
            )
            self.assertEqual(restored.log_likelihood, -12.5)
            assert restored.posterior_probabilities is not None
            np.testing.assert_allclose(restored.posterior_probabilities[1], posterior[1])

    def test_invalid_posterior_and_overwrite_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "posterior"):
            StatePredictions(
                assignments=_assignments(),
                model_kind="gaussian-hmm-state",
                model_seed=19,
                model_fingerprint="b" * 64,
                posterior_probabilities=(np.ones((4, 2)), np.ones((3, 2))),
                log_likelihood=-1.0,
            )
        with TemporaryDirectory() as temporary:
            target = Path(temporary) / "labels"
            predictions = StatePredictions(
                assignments=_assignments(),
                model_kind="kmeans-state",
                model_seed=17,
                model_fingerprint="a" * 64,
            )
            save_state_predictions(predictions, target)
            with self.assertRaises(FileExistsError):
                save_state_predictions(predictions, target)

    def test_v1_artifact_without_model_seed_remains_readable(self):
        with TemporaryDirectory() as temporary:
            target = save_state_predictions(
                StatePredictions(
                    assignments=_assignments(),
                    model_kind="kmeans-state",
                    model_seed=17,
                    model_fingerprint="a" * 64,
                ),
                Path(temporary) / "labels",
            )
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["format_version"] = 1
            del manifest["model_seed"]
            del manifest["model_fingerprint"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            restored = load_state_predictions(target)
            self.assertIsNone(restored.model_seed)
            self.assertEqual(restored.subjects, ("sub-001", "sub-002"))

    def test_writer_uses_the_highest_supported_identity_version(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                (None, None, 1),
                (17, None, 2),
                (17, "a" * 64, 3),
            )
            for index, (seed, fingerprint, expected_version) in enumerate(cases):
                target = save_state_predictions(
                    StatePredictions(
                        assignments=_assignments(),
                        model_kind="kmeans-state",
                        model_seed=seed,
                        model_fingerprint=fingerprint,
                    ),
                    root / f"labels-{index}",
                )
                manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["format_version"], expected_version)
                self.assertEqual(load_state_predictions(target).model_fingerprint, fingerprint)


if __name__ == "__main__":
    unittest.main()
