import contextlib
import io
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.cli import main
from dfckit.io import (
    StateModelScoreReport,
    compare_state_model_scores,
    state_count_comparison_payload,
    write_state_count_comparison,
    write_state_model_scores,
)
from dfckit.states import RunGaussianHMMScore, RunKMeansScore

KMEANS_SPECIFICATION = {
    "algorithm": "minibatch",
    "batch_size": 4096,
    "implementation": "scikit-learn test MiniBatchKMeans",
    "init_sample_size": 1000,
    "max_iter": 10,
    "n_init": 10,
    "reassignment_ratio": 0.01,
    "standardize_features": True,
}

HMM_SPECIFICATION = {
    "covariance_type": "diag",
    "implementation": "hmmlearn test; scikit-learn test IncrementalPCA",
    "minimum_sequence_length": 2,
    "n_init": 3,
    "n_iter": 200,
    "n_pca_components": 10,
    "pca_batch_size": 4096,
    "tol": 0.001,
}


def _fingerprint(index: int) -> str:
    return f"{index:064x}"


def _report(
    *,
    fold: int,
    n_states: int,
    values: tuple[float, float],
    kind: str = "kmeans-state",
    seed: int = 17,
    training_fingerprint: str | None = None,
    evaluation_fingerprint: str | None = None,
    evaluation_subjects: tuple[str, str] | None = None,
    fit_subjects: tuple[str, str] | None = None,
    model_fingerprint: str | None = None,
    specification: dict[str, object] | None = None,
    feature_contract_fingerprint: str = "f" * 64,
    format_version: int = 2,
) -> StateModelScoreReport:
    subjects = evaluation_subjects or (
        ("sub-003", "sub-004") if fold == 1 else ("sub-001", "sub-002")
    )
    fitted = fit_subjects or (
        ("sub-001", "sub-002") if fold == 1 else ("sub-003", "sub-004")
    )
    score_type = RunKMeansScore if kind == "kmeans-state" else RunGaussianHMMScore
    scores = []
    for subject, value in zip(subjects, values, strict=True):
        common = {
            "subject": subject,
            "session": "off",
            "acquisition_id": f"{subject}_ses-off_task-rest_run-1",
            "n_samples": 10,
            "n_sequences": 1,
        }
        if score_type is RunKMeansScore:
            scores.append(
                score_type(
                    **common,
                    total_squared_distance=value * 10,
                    mean_squared_distance=value,
                )
            )
        else:
            scores.append(
                score_type(
                    **common,
                    log_likelihood=value * 10,
                    log_likelihood_per_sample=value,
                )
            )
    if specification is None:
        specification = (
            dict(KMEANS_SPECIFICATION)
            if kind == "kmeans-state"
            else dict(HMM_SPECIFICATION)
        )
    if format_version == 1:
        specification = None
        feature_contract: str | None = None
    else:
        feature_contract = feature_contract_fingerprint
    return StateModelScoreReport(
        scores=tuple(scores),
        format_version=format_version,
        model_kind=kind,
        model_fingerprint=(
            model_fingerprint
            or _fingerprint(fold * 10_000 + n_states * 100 + seed)
        ),
        model_seed=seed,
        n_states=n_states,
        fit_subjects=fitted,
        training_data_fingerprint=training_fingerprint or _fingerprint(10 + fold),
        evaluation_data_fingerprint=evaluation_fingerprint or _fingerprint(20 + fold),
        feature_contract_fingerprint=feature_contract,
        source_contract="state-selection-test:v1",
        sample_interval_seconds=0.8,
        minimum_sequence_length=1 if kind == "kmeans-state" else 2,
        omitted_short_sequence_count=0,
        allow_fit_subjects=False,
        model_specification=specification,
    )


def _reports() -> tuple[StateModelScoreReport, ...]:
    return (
        _report(fold=2, n_states=6, values=(0.95, 0.97)),
        _report(fold=1, n_states=2, values=(1.0, 1.2)),
        _report(fold=2, n_states=2, values=(0.95, 1.05)),
        _report(fold=1, n_states=6, values=(0.88, 0.92)),
        _report(fold=1, n_states=4, values=(0.94, 0.96)),
        _report(fold=2, n_states=4, values=(0.93, 0.95)),
    )


def _multi_seed_reports() -> tuple[StateModelScoreReport, ...]:
    reports = []
    values = {
        (1, 2, 17): (1.0, 1.2),
        (1, 2, 29): (0.8, 1.0),
        (1, 4, 17): (0.8, 0.8),
        (1, 4, 29): (1.0, 1.0),
        (2, 2, 17): (0.9, 1.1),
        (2, 2, 29): (1.1, 1.3),
        (2, 4, 17): (0.85, 0.95),
        (2, 4, 29): (0.95, 1.05),
    }
    for (fold, n_states, seed), scores in values.items():
        reports.append(
            _report(
                fold=fold,
                n_states=n_states,
                seed=seed,
                values=scores,
            )
        )
    return tuple(reversed(reports))


class StateSelectionIOTests(unittest.TestCase):
    def test_comparison_groups_folds_and_uses_subject_balanced_scores(self):
        comparison = compare_state_model_scores(_reports())
        self.assertEqual(comparison.selection.candidate_n_states.tolist(), [2, 4, 6])
        self.assertEqual(comparison.selection.best_n_states, 6)
        self.assertEqual(comparison.selection.one_standard_error_n_states, 4)
        self.assertEqual(comparison.folds[0].candidate_n_states, (2, 4, 6))
        self.assertEqual(comparison.subjects, ("sub-003", "sub-004", "sub-001", "sub-002"))
        np.testing.assert_allclose(comparison.folds[0].fold_scores, (1.1, 0.95, 0.9))
        np.testing.assert_allclose(comparison.folds[1].fold_scores, (1.0, 0.94, 0.96))
        payload = state_count_comparison_payload(comparison)
        self.assertEqual(payload["n_folds"], 2)
        self.assertEqual(payload["n_subjects"], 4)
        np.testing.assert_allclose(
            payload["candidates"][2]["fold_scores"],
            [0.9, 0.96],
        )
        self.assertEqual(payload["feature_contract_fingerprint"], "f" * 64)

    def test_comparison_object_rejects_candidate_and_specification_misalignment(self):
        comparison = compare_state_model_scores(_reports())
        changed_candidate = replace(comparison.folds[0].candidates[2], n_states=8)
        changed_fold = replace(
            comparison.folds[0],
            candidates=(*comparison.folds[0].candidates[:2], changed_candidate),
        )
        with self.assertRaisesRegex(ValueError, "candidate state counts are misaligned"):
            replace(comparison, folds=(changed_fold, comparison.folds[1]))

        candidate = comparison.folds[0].candidates[0]
        specifications = [dict(value) for value in candidate.model_specifications]
        specifications[0]["n_init"] = 20
        changed_candidate = replace(
            candidate,
            model_specifications=tuple(specifications),
        )
        changed_fold = replace(
            comparison.folds[0],
            candidates=(changed_candidate, *comparison.folds[0].candidates[1:]),
        )
        with self.assertRaisesRegex(ValueError, "model specifications are inconsistent"):
            replace(comparison, folds=(changed_fold, comparison.folds[1]))

    def test_hmm_comparison_uses_higher_is_better(self):
        reports = (
            _report(fold=1, n_states=2, values=(-2.0, -2.2), kind="gaussian-hmm-state"),
            _report(fold=1, n_states=4, values=(-1.8, -1.9), kind="gaussian-hmm-state"),
            _report(fold=2, n_states=2, values=(-2.1, -2.0), kind="gaussian-hmm-state"),
            _report(fold=2, n_states=4, values=(-1.7, -1.8), kind="gaussian-hmm-state"),
        )
        comparison = compare_state_model_scores(reports)
        self.assertEqual(comparison.selection.best_n_states, 4)
        self.assertEqual(comparison.selection_metric, "subject-balanced log likelihood per sample")

    def test_repeated_seeds_are_equal_weighted_before_subjects_and_folds(self):
        comparison = compare_state_model_scores(_multi_seed_reports())
        self.assertEqual(comparison.model_seeds, (17, 29))
        self.assertEqual(comparison.selection.candidate_n_states.tolist(), [2, 4])
        np.testing.assert_allclose(comparison.folds[0].fold_scores, (1.0, 0.9))
        np.testing.assert_allclose(comparison.folds[1].fold_scores, (1.1, 0.95))
        candidate = comparison.folds[0].candidates[0]
        np.testing.assert_allclose(candidate.fold_scores_by_seed, (1.1, 0.9))
        np.testing.assert_allclose(candidate.subject_scores, (0.9, 1.1))
        payload = state_count_comparison_payload(comparison)
        self.assertEqual(payload["format_version"], 2)
        self.assertEqual(payload["n_seeds"], 2)
        self.assertEqual(payload["model_seeds"], [17, 29])
        self.assertEqual(
            [
                seed["model_seed"]
                for seed in payload["folds"][0]["candidates"][0]["seeds"]
            ],
            [17, 29],
        )

    def test_candidate_object_recomputes_every_repeated_seed_summary(self):
        comparison = compare_state_model_scores(_multi_seed_reports())
        candidate = comparison.folds[0].candidates[0]

        changed_seed_scores = [list(values) for values in candidate.subject_scores_by_seed]
        changed_seed_scores[0][0] += 0.25
        with self.assertRaisesRegex(ValueError, "subject scores do not equal their seed means"):
            replace(
                candidate,
                subject_scores_by_seed=tuple(tuple(values) for values in changed_seed_scores),
            )

        changed_subject_scores = list(candidate.subject_scores)
        changed_subject_scores[0] += 0.25
        with self.assertRaisesRegex(ValueError, "subject scores do not equal their seed means"):
            replace(candidate, subject_scores=tuple(changed_subject_scores))

        changed_seed_fold_scores = list(candidate.fold_scores_by_seed)
        changed_seed_fold_scores[0] += 0.25
        with self.assertRaisesRegex(ValueError, "seed fold scores are not subject balanced"):
            replace(candidate, fold_scores_by_seed=tuple(changed_seed_fold_scores))

        with self.assertRaisesRegex(ValueError, "fold score is not seed and subject balanced"):
            replace(candidate, fold_score=candidate.fold_score + 0.25)

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            replace(candidate, model_seeds=tuple(reversed(candidate.model_seeds)))

    def test_repeated_seed_grid_must_be_complete_in_every_fold_and_candidate(self):
        incomplete = list(_multi_seed_reports())
        incomplete.pop()
        with self.assertRaisesRegex(ValueError, "same model seeds|same state-count and seed grid"):
            compare_state_model_scores(incomplete)

        changed = list(_multi_seed_reports())
        reference = changed[0]
        changed[0] = _report(
            fold=2,
            n_states=reference.n_states,
            seed=41,
            values=tuple(score.mean_squared_distance for score in reference.scores),
        )
        with self.assertRaisesRegex(ValueError, "same model seeds|same state-count and seed grid"):
            compare_state_model_scores(changed)

    def test_writer_and_cli_roundtrip_real_score_artifacts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, report in enumerate(_reports()):
                path = root / f"score-{index}.json"
                write_state_model_scores(
                    report.scores,
                    path,
                    model_kind=report.model_kind,
                    model_fingerprint=report.model_fingerprint,
                    model_seed=report.model_seed,
                    n_states=report.n_states,
                    fit_subjects=report.fit_subjects,
                    training_data_fingerprint=report.training_data_fingerprint,
                    evaluation_data_fingerprint=report.evaluation_data_fingerprint,
                    feature_contract_fingerprint=report.feature_contract_fingerprint,
                    source_contract=report.source_contract,
                    sample_interval_seconds=report.sample_interval_seconds,
                    minimum_sequence_length=report.minimum_sequence_length,
                    omitted_short_sequence_count=report.omitted_short_sequence_count,
                    allow_fit_subjects=report.allow_fit_subjects,
                    model_specification=report.model_specification,
                )
                paths.append(path)
            output = root / "comparison.json"
            arguments = ["compare-state-counts", str(output)]
            for path in paths:
                arguments.extend(("--score", str(path)))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(arguments)
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["best_n_states"], 6)
            self.assertEqual(summary["one_standard_error_n_states"], 4)
            self.assertEqual(payload["format"], "dfckit-state-count-comparison")
            self.assertNotIn("NaN", output.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                write_state_count_comparison(compare_state_model_scores(_reports()), output)

    def test_cli_roundtrip_preserves_complete_repeated_seed_grid(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, report in enumerate(_multi_seed_reports()):
                path = root / f"score-{index}.json"
                write_state_model_scores(
                    report.scores,
                    path,
                    model_kind=report.model_kind,
                    model_fingerprint=report.model_fingerprint,
                    model_seed=report.model_seed,
                    n_states=report.n_states,
                    fit_subjects=report.fit_subjects,
                    training_data_fingerprint=report.training_data_fingerprint,
                    evaluation_data_fingerprint=report.evaluation_data_fingerprint,
                    feature_contract_fingerprint=report.feature_contract_fingerprint,
                    source_contract=report.source_contract,
                    sample_interval_seconds=report.sample_interval_seconds,
                    minimum_sequence_length=report.minimum_sequence_length,
                    omitted_short_sequence_count=report.omitted_short_sequence_count,
                    allow_fit_subjects=report.allow_fit_subjects,
                    model_specification=report.model_specification,
                )
                paths.append(path)

            output = root / "multi-seed-comparison.json"
            arguments = ["compare-state-counts", str(output)]
            for path in paths:
                arguments.extend(("--score", str(path)))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(arguments)

            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["n_seeds"], 2)
            self.assertEqual(summary["model_seeds"], [17, 29])
            self.assertEqual(payload["format_version"], 2)
            self.assertEqual(payload["model_seeds"], [17, 29])
            np.testing.assert_allclose(
                payload["folds"][0]["candidates"][0]["subject_scores"][0]["score"],
                0.9,
            )
            np.testing.assert_allclose(
                [
                    seed["fold_score"]
                    for seed in payload["folds"][0]["candidates"][0]["seeds"]
                ],
                [1.1, 0.9],
            )

    def test_incomplete_candidate_set_and_legacy_reports_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same state-count and seed grid"):
            compare_state_model_scores(_reports()[:-1])
        legacy = list(_reports())
        legacy[0] = _report(
            fold=2,
            n_states=6,
            values=(0.95, 0.97),
            format_version=1,
        )
        with self.assertRaisesRegex(ValueError, "v2"):
            compare_state_model_scores(legacy)

    def test_fold_training_seed_and_run_boundaries_must_match(self):
        changed = list(_reports())
        changed[0] = _report(
            fold=2,
            n_states=6,
            values=(0.95, 0.97),
            training_fingerprint=_fingerprint(99),
        )
        with self.assertRaisesRegex(ValueError, "same training data"):
            compare_state_model_scores(changed)

        changed = list(_reports())
        changed[0] = _report(fold=2, n_states=6, values=(0.95, 0.97), seed=29)
        with self.assertRaisesRegex(ValueError, "same model seeds"):
            compare_state_model_scores(changed)

        changed = list(_reports())
        boundary = _report(fold=2, n_states=6, values=(0.95, 0.97))
        first = boundary.scores[0]
        changed_score = RunKMeansScore(
            first.subject,
            first.session,
            first.acquisition_id,
            20,
            first.n_sequences,
            first.total_squared_distance * 2,
            first.mean_squared_distance,
        )
        changed[0] = StateModelScoreReport(
            **{
                **boundary.__dict__,
                "scores": (changed_score, boundary.scores[1]),
            }
        )
        with self.assertRaisesRegex(ValueError, "run boundaries"):
            compare_state_model_scores(changed)

    def test_outer_subject_feature_contract_and_specification_must_match(self):
        changed = list(_reports())
        for index, n_states, values in (
            (0, 6, (0.95, 0.97)),
            (2, 2, (0.95, 1.05)),
            (5, 4, (0.93, 0.95)),
        ):
            changed[index] = _report(
                fold=2,
                n_states=n_states,
                values=values,
                evaluation_subjects=("sub-003", "sub-005"),
                fit_subjects=("sub-001", "sub-002"),
            )
        with self.assertRaisesRegex(ValueError, "evaluation subjects overlap"):
            compare_state_model_scores(changed)

        changed = list(_reports())
        for index, n_states, values in (
            (0, 6, (0.95, 0.97)),
            (2, 2, (0.95, 1.05)),
            (5, 4, (0.93, 0.95)),
        ):
            changed[index] = _report(
                fold=2,
                n_states=n_states,
                values=values,
                evaluation_subjects=("sub-005", "sub-006"),
                fit_subjects=("sub-003", "sub-004"),
            )
        with self.assertRaisesRegex(ValueError, "one development cohort"):
            compare_state_model_scores(changed)

        incomplete_validation = []
        for report in _reports():
            if report.evaluation_data_fingerprint == _fingerprint(22):
                incomplete_validation.append(
                    _report(
                        fold=2,
                        n_states=report.n_states,
                        values=(
                            report.scores[0].mean_squared_distance,
                            report.scores[1].mean_squared_distance,
                        ),
                        evaluation_subjects=("sub-003", "sub-004"),
                        fit_subjects=("sub-001", "sub-002", "sub-005"),
                    )
                )
            else:
                incomplete_validation.append(
                    _report(
                        fold=1,
                        n_states=report.n_states,
                        values=(
                            report.scores[0].mean_squared_distance,
                            report.scores[1].mean_squared_distance,
                        ),
                        evaluation_subjects=("sub-001", "sub-002"),
                        fit_subjects=("sub-003", "sub-004", "sub-005"),
                    )
                )
        with self.assertRaisesRegex(ValueError, "evaluate every development subject"):
            compare_state_model_scores(incomplete_validation)

        changed = list(_reports())
        changed[0] = _report(
            fold=2,
            n_states=6,
            values=(0.95, 0.97),
            feature_contract_fingerprint="e" * 64,
        )
        with self.assertRaisesRegex(ValueError, "feature contract"):
            compare_state_model_scores(changed)

        specification = dict(KMEANS_SPECIFICATION)
        specification["n_init"] = 20
        changed = list(_reports())
        changed[0] = _report(
            fold=2,
            n_states=6,
            values=(0.95, 0.97),
            specification=specification,
        )
        with self.assertRaisesRegex(ValueError, "model specifications"):
            compare_state_model_scores(changed)

        varying_initialization = dict(KMEANS_SPECIFICATION)
        varying_initialization["init_sample_size"] = 800
        varying_initialization["implementation"] = (
            "scikit-learn test MiniBatchKMeans; kmeans++ sample=800"
        )
        changed = list(_reports())
        changed[0] = _report(
            fold=2,
            n_states=6,
            values=(0.95, 0.97),
            specification=varying_initialization,
        )
        comparison = compare_state_model_scores(changed)
        payload = state_count_comparison_payload(comparison)
        self.assertNotIn("init_sample_size", payload["comparison_specification"])
        self.assertEqual(
            payload["folds"][1]["candidates"][2]["seeds"][0][
                "model_specification"
            ]["init_sample_size"],
            800,
        )

    def test_cli_refuses_existing_output_before_reading_scores(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "comparison.json"
            output.write_text("preserve", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "compare-state-counts",
                        str(output),
                        "--score",
                        str(Path(temporary) / "missing.json"),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("already exists", stderr.getvalue())
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
