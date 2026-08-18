import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.io import (
    StateModelScoreReport,
    selected_state_count_evaluation_payload,
    write_selected_state_count_evaluation,
)
from dfckit.states import RunKMeansScore


def _report(seed: int, subject_totals: tuple[tuple[float, float], ...]) -> StateModelScoreReport:
    scores = []
    for subject_index, (short_total, long_total) in enumerate(subject_totals, start=10):
        subject = f"sub-{subject_index:03d}"
        scores.extend(
            (
                RunKMeansScore(
                    subject,
                    "off",
                    f"{subject}_ses-off_task-rest_run-1",
                    2,
                    1,
                    short_total,
                    short_total / 2,
                ),
                RunKMeansScore(
                    subject,
                    "off",
                    f"{subject}_ses-off_task-rest_run-2",
                    8,
                    2,
                    long_total,
                    long_total / 8,
                ),
            )
        )
    return StateModelScoreReport(
        scores=tuple(scores),
        format_version=2,
        model_kind="kmeans-state",
        model_fingerprint=f"{seed + 1:064x}",
        model_seed=seed,
        n_states=4,
        fit_subjects=("sub-001", "sub-002", "sub-003"),
        training_data_fingerprint="a" * 64,
        evaluation_data_fingerprint="b" * 64,
        feature_contract_fingerprint="c" * 64,
        source_contract="state-evaluation-test:v1",
        sample_interval_seconds=0.8,
        minimum_sequence_length=1,
        omitted_short_sequence_count=0,
        allow_fit_subjects=False,
        model_specification={
            "algorithm": "minibatch",
            "batch_size": 4096,
            "implementation": "scikit-learn test MiniBatchKMeans",
            "init_sample_size": 1000,
            "max_iter": 10,
            "n_init": 10,
            "reassignment_ratio": 0.01,
            "standardize_features": True,
        },
    )


def _metadata() -> dict[str, object]:
    return {
        "model_artifacts": ("models/k-4_seed-17.model", "models/k-4_seed-29.model"),
        "score_artifacts": ("scores/k-4_seed-17.json", "scores/k-4_seed-29.json"),
        "method": "kmeans",
        "selection_policy": "one-standard-error",
        "selection_workflow_fingerprint": "d" * 64,
        "selected_n_states": 4,
        "development_data_fingerprint": "a" * 64,
        "development_subjects": ("sub-001", "sub-002", "sub-003"),
        "test_data_fingerprint": "b" * 64,
        "test_subjects": ("sub-010", "sub-011"),
        "fit_configuration": {
            "batch_size": 4096,
            "init_sample_size": None,
            "max_iter": 10,
            "n_init": 10,
            "reassignment_ratio": 0.01,
            "standardize_features": True,
        },
    }


class StateEvaluationIOTests(unittest.TestCase):
    def test_aggregates_runs_then_seeds_then_subjects(self):
        reports = (
            _report(17, ((2.0, 16.0), (6.0, 8.0))),
            _report(29, ((6.0, 24.0), (2.0, 32.0))),
        )
        payload = selected_state_count_evaluation_payload(reports, **_metadata())
        expected_by_seed = np.asarray([[1.8, 1.4], [3.0, 3.4]])
        expected_subjects = expected_by_seed.mean(axis=0)
        np.testing.assert_allclose(
            [
                [record["score"] for record in seed["subject_scores"]]
                for seed in payload["seeds"]
            ],
            expected_by_seed,
        )
        np.testing.assert_allclose(
            [record["score"] for record in payload["subject_scores"]],
            expected_subjects,
        )
        self.assertAlmostEqual(payload["cohort_score"], float(expected_subjects.mean()))
        self.assertNotEqual(
            payload["cohort_score"],
            float(
                np.mean(
                    [score.mean_squared_distance for report in reports for score in report.scores]
                )
            ),
        )

    def test_writer_is_atomic_and_refuses_overwrite(self):
        reports = (
            _report(17, ((2.0, 16.0), (6.0, 8.0))),
            _report(29, ((6.0, 24.0), (2.0, 32.0))),
        )
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation.json"
            write_selected_state_count_evaluation(reports, output, **_metadata())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["format_version"], 1)
            self.assertNotIn("NaN", output.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                write_selected_state_count_evaluation(reports, output, **_metadata())

    def test_rejects_overlap_contract_mismatch_and_wrong_artifact_path(self):
        reports = (
            _report(17, ((2.0, 16.0), (6.0, 8.0))),
            _report(29, ((6.0, 24.0), (2.0, 32.0))),
        )
        overlap = _metadata()
        overlap["test_subjects"] = ("sub-001", "sub-011")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            selected_state_count_evaluation_payload(reports, **overlap)

        mismatch = _metadata()
        mismatch["test_data_fingerprint"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "outer-test contract"):
            selected_state_count_evaluation_payload(reports, **mismatch)

        wrong_configuration = _metadata()
        wrong_configuration["fit_configuration"] = {
            **wrong_configuration["fit_configuration"],
            "max_iter": 11,
        }
        with self.assertRaisesRegex(ValueError, "fit_configuration disagrees"):
            selected_state_count_evaluation_payload(reports, **wrong_configuration)

        wrong_path = _metadata()
        wrong_path["model_artifacts"] = (
            "models/k-4_seed-29.model",
            "models/k-4_seed-17.model",
        )
        with self.assertRaisesRegex(ValueError, "path is misaligned"):
            selected_state_count_evaluation_payload(reports, **wrong_path)

        outside = _metadata()
        outside["score_artifacts"] = ("../score-17.json", "scores/k-4_seed-29.json")
        with self.assertRaisesRegex(ValueError, "inside the evaluation"):
            selected_state_count_evaluation_payload(reports, **outside)


if __name__ == "__main__":
    unittest.main()
