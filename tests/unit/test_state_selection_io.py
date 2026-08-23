import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dfckit.artifacts import StateModelScoreReport
from dfckit.states import (
    RunKMeansScore,
    compare_state_model_scores,
    state_count_comparison_payload,
    write_state_count_comparison,
)


def _report(fold: int, n_states: int, seed: int, score: float) -> StateModelScoreReport:
    evaluation_subjects = ("sub-001", "sub-002") if fold == 0 else ("sub-003", "sub-004")
    fit_subjects = ("sub-003", "sub-004") if fold == 0 else ("sub-001", "sub-002")
    scores = tuple(
        RunKMeansScore(subject, "off", f"{subject}_run-1", 10, 1, score * 10, score)
        for subject in evaluation_subjects
    )
    return StateModelScoreReport(
        scores=scores,
        format_version=3,
        model_kind="kmeans-state",
        model_seed=seed,
        n_states=n_states,
        fit_subjects=fit_subjects,
        source_contract="window-fc:test",
        sample_interval_seconds=0.8,
        minimum_sequence_length=1,
        omitted_short_sequence_count=0,
        allow_fit_subjects=False,
        model_specification={
            "algorithm": "minibatch",
            "batch_size": 128,
            "implementation": "test",
            "init_sample_size": 100,
            "max_iter": 10,
            "n_init": 5,
            "reassignment_ratio": 0.01,
            "standardize_features": True,
        },
    )


def _reports() -> tuple[StateModelScoreReport, ...]:
    return tuple(
        _report(fold, n_states, seed, base + offset)
        for fold, offset in ((0, 0.0), (1, 0.2))
        for n_states, base in ((2, 2.0), (4, 1.0))
        for seed in (11, 19)
    )


class StateSelectionTests(unittest.TestCase):
    def test_scores_are_averaged_by_seed_subject_and_fold(self):
        comparison = compare_state_model_scores(_reports())
        self.assertEqual(comparison.selection.best_n_states, 4)
        self.assertEqual(comparison.model_seeds, (11, 19))
        self.assertEqual(comparison.selection.fold_scores.tolist(), [[2.0, 1.0], [2.2, 1.2]])

    def test_payload_and_writer_are_compact(self):
        comparison = compare_state_model_scores(_reports())
        payload = state_count_comparison_payload(comparison)
        self.assertEqual(payload["best_n_states"], 4)
        with TemporaryDirectory() as temporary:
            path = write_state_count_comparison(comparison, Path(temporary) / "state-counts.json")
            self.assertEqual(json.loads(path.read_text())["best_n_states"], 4)

    def test_incomplete_seed_grid_is_rejected(self):
        reports = list(_reports())
        reports.pop()
        with self.assertRaisesRegex(ValueError, "same unique seeds|same candidate"):
            compare_state_model_scores(reports)

    def test_model_setting_mismatch_is_rejected(self):
        reports = list(_reports())
        changed = reports[-1]
        reports[-1] = StateModelScoreReport(
            scores=changed.scores,
            format_version=changed.format_version,
            model_kind=changed.model_kind,
            model_seed=changed.model_seed,
            n_states=changed.n_states,
            fit_subjects=changed.fit_subjects,
            source_contract=changed.source_contract,
            sample_interval_seconds=changed.sample_interval_seconds,
            minimum_sequence_length=changed.minimum_sequence_length,
            omitted_short_sequence_count=changed.omitted_short_sequence_count,
            allow_fit_subjects=changed.allow_fit_subjects,
            model_specification={**dict(changed.model_specification), "max_iter": 20},
        )
        with self.assertRaisesRegex(ValueError, "same model and feature settings"):
            compare_state_model_scores(reports)


if __name__ == "__main__":
    unittest.main()
