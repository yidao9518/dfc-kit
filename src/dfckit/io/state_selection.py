"""Cross-fold comparison of held-out state-model score artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..states.scoring import RunKMeansScore
from ..states.selection import StateCountSelection, select_state_count
from .state_scoring import StateModelScoreReport

FORMAT_NAME = "dfckit-state-count-comparison"
FORMAT_VERSION = 2


def _is_fingerprint(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class CandidateStateModelScores:
    """Repeated-seed held-out scores for one candidate state count in one fold."""

    n_states: int
    model_seeds: tuple[int, ...]
    model_fingerprints: tuple[str, ...]
    model_specifications: tuple[Mapping[str, object], ...]
    score_artifacts: tuple[str, ...]
    subject_scores_by_seed: tuple[tuple[float, ...], ...]
    subject_scores: tuple[float, ...]
    fold_scores_by_seed: tuple[float, ...]
    fold_score: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_states, (bool, np.bool_))
            or not isinstance(self.n_states, (int, np.integer))
            or self.n_states < 2
        ):
            raise ValueError("candidate n_states must be an integer of at least two")
        raw_seeds = tuple(self.model_seeds)
        if (
            not raw_seeds
            or any(
                isinstance(seed, (bool, np.bool_))
                or not isinstance(seed, (int, np.integer))
                or seed < 0
                for seed in raw_seeds
            )
            or any(right <= left for left, right in pairwise(raw_seeds))
        ):
            raise ValueError("candidate model seeds must be strictly increasing non-negative integers")
        seeds = tuple(int(seed) for seed in raw_seeds)
        fingerprints = tuple(self.model_fingerprints)
        artifacts = tuple(self.score_artifacts)
        specifications = tuple(dict(value) for value in self.model_specifications)
        seed_subject_scores = tuple(
            tuple(float(item) for item in values)
            for values in self.subject_scores_by_seed
        )
        subject_scores = tuple(float(value) for value in self.subject_scores)
        seed_fold_scores = tuple(float(value) for value in self.fold_scores_by_seed)
        fold_score = float(self.fold_score)
        n_seeds = len(seeds)
        if len(set(fingerprints)) != n_seeds or not all(
            _is_fingerprint(value) for value in fingerprints
        ):
            raise ValueError("candidate model fingerprints must be distinct SHA-256 values")
        if not (
            len(artifacts)
            == len(specifications)
            == len(seed_subject_scores)
            == len(seed_fold_scores)
            == n_seeds
        ):
            raise ValueError("candidate seed arrays must have matching lengths")
        if any(not isinstance(value, str) for value in artifacts):
            raise TypeError("candidate score artifact paths must be strings")
        if not subject_scores or any(
            len(values) != len(subject_scores) for values in seed_subject_scores
        ):
            raise ValueError("candidate seed scores must share a non-empty subject axis")
        if not all(np.isfinite(values).all() for values in seed_subject_scores):
            raise ValueError("candidate seed subject scores must be finite")
        if not np.isfinite(subject_scores).all() or not np.isfinite(seed_fold_scores).all():
            raise ValueError("candidate aggregated scores must be finite")
        if not np.isfinite(fold_score):
            raise ValueError("candidate fold score must be finite")
        expected_subject_scores = np.mean(seed_subject_scores, axis=0)
        expected_seed_fold_scores = np.mean(seed_subject_scores, axis=1)
        expected_fold_score = float(np.mean(expected_subject_scores))
        if not np.allclose(
            subject_scores,
            expected_subject_scores,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("candidate subject scores do not equal their seed means")
        if not np.allclose(
            seed_fold_scores,
            expected_seed_fold_scores,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("candidate seed fold scores are not subject balanced")
        if not np.isclose(fold_score, expected_fold_score, rtol=1e-12, atol=1e-12):
            raise ValueError("candidate fold score is not seed and subject balanced")
        for specification in specifications:
            try:
                json.dumps(specification, allow_nan=False, sort_keys=True)
            except (TypeError, ValueError) as error:
                raise ValueError("candidate model specifications must be finite JSON") from error
        object.__setattr__(self, "n_states", int(self.n_states))
        object.__setattr__(self, "model_seeds", seeds)
        object.__setattr__(self, "model_fingerprints", fingerprints)
        object.__setattr__(
            self,
            "model_specifications",
            tuple(MappingProxyType(value) for value in specifications),
        )
        object.__setattr__(self, "score_artifacts", artifacts)
        object.__setattr__(self, "subject_scores_by_seed", seed_subject_scores)
        object.__setattr__(self, "subject_scores", subject_scores)
        object.__setattr__(self, "fold_scores_by_seed", seed_fold_scores)
        object.__setattr__(self, "fold_score", fold_score)

    @property
    def n_seeds(self) -> int:
        return len(self.model_seeds)


@dataclass(frozen=True)
class FoldStateModelScores:
    """One subject-disjoint fold with complete candidate and seed grids."""

    evaluation_data_fingerprint: str
    training_data_fingerprint: str
    fit_subjects: tuple[str, ...]
    evaluation_subjects: tuple[str, ...]
    n_runs: int
    n_sequences: int
    n_samples: int
    omitted_short_sequence_count: int
    candidates: tuple[CandidateStateModelScores, ...]

    def __post_init__(self) -> None:
        if not _is_fingerprint(self.evaluation_data_fingerprint) or not _is_fingerprint(
            self.training_data_fingerprint
        ):
            raise ValueError("fold data fingerprints must be lowercase SHA-256 values")
        fit_subjects = tuple(self.fit_subjects)
        evaluation_subjects = tuple(self.evaluation_subjects)
        if (
            not fit_subjects
            or not evaluation_subjects
            or len(set(fit_subjects)) != len(fit_subjects)
            or len(set(evaluation_subjects)) != len(evaluation_subjects)
            or any(
                not isinstance(subject, str) or not subject.strip()
                for subject in (*fit_subjects, *evaluation_subjects)
            )
        ):
            raise ValueError("fold subject identities must be unique and non-empty")
        if set(fit_subjects).intersection(evaluation_subjects):
            raise ValueError("fold fit and evaluation subjects must be disjoint")
        counts = (self.n_runs, self.n_sequences, self.n_samples)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in counts
        ):
            raise ValueError("fold run, sequence, and sample counts must be positive integers")
        if (
            self.n_runs < len(evaluation_subjects)
            or self.n_sequences < self.n_runs
            or self.n_samples < self.n_sequences
        ):
            raise ValueError("fold run, sequence, and sample counts are inconsistent")
        if (
            isinstance(self.omitted_short_sequence_count, bool)
            or not isinstance(self.omitted_short_sequence_count, int)
            or self.omitted_short_sequence_count < 0
        ):
            raise ValueError("fold omitted sequence count must be a non-negative integer")
        candidates = tuple(self.candidates)
        if len(candidates) < 2 or any(
            not isinstance(candidate, CandidateStateModelScores)
            for candidate in candidates
        ):
            raise ValueError("fold requires at least two candidate score objects")
        candidate_n_states = tuple(candidate.n_states for candidate in candidates)
        if any(right <= left for left, right in pairwise(candidate_n_states)):
            raise ValueError("fold candidate state counts must be strictly increasing")
        model_seeds = candidates[0].model_seeds
        if any(candidate.model_seeds != model_seeds for candidate in candidates[1:]):
            raise ValueError("fold candidates must use the same model seeds")
        if any(
            len(candidate.subject_scores) != len(evaluation_subjects)
            for candidate in candidates
        ):
            raise ValueError("fold candidate scores do not match evaluation subjects")
        object.__setattr__(self, "fit_subjects", fit_subjects)
        object.__setattr__(self, "evaluation_subjects", evaluation_subjects)
        object.__setattr__(self, "candidates", candidates)

    @property
    def candidate_n_states(self) -> tuple[int, ...]:
        return tuple(candidate.n_states for candidate in self.candidates)

    @property
    def model_seeds(self) -> tuple[int, ...]:
        return self.candidates[0].model_seeds

    @property
    def fold_scores(self) -> tuple[float, ...]:
        return tuple(candidate.fold_score for candidate in self.candidates)


@dataclass(frozen=True)
class StateCountComparison:
    """Validated cross-fold state-count comparison and selection."""

    model_kind: str
    selection_metric: str
    selection_direction: str
    source_contract: str
    sample_interval_seconds: float | None
    minimum_sequence_length: int
    feature_contract_fingerprint: str
    model_specification: Mapping[str, object]
    selection: StateCountSelection
    folds: tuple[FoldStateModelScores, ...]

    def __post_init__(self) -> None:
        if self.model_kind not in {"kmeans-state", "gaussian-hmm-state"}:
            raise ValueError("state-count comparison model_kind is invalid")
        expected_metric = (
            "subject-balanced mean squared distance"
            if self.model_kind == "kmeans-state"
            else "subject-balanced log likelihood per sample"
        )
        expected_direction = (
            "lower is better" if self.model_kind == "kmeans-state" else "higher is better"
        )
        if self.selection_metric != expected_metric or self.selection_direction != expected_direction:
            raise ValueError("state-count comparison metric metadata is inconsistent")
        if not isinstance(self.source_contract, str) or not self.source_contract.strip():
            raise ValueError("state-count comparison source_contract must be non-empty")
        if self.sample_interval_seconds is not None and (
            not np.isfinite(self.sample_interval_seconds) or self.sample_interval_seconds <= 0.0
        ):
            raise ValueError("state-count comparison sample interval is invalid")
        if (
            isinstance(self.minimum_sequence_length, bool)
            or not isinstance(self.minimum_sequence_length, int)
            or self.minimum_sequence_length < 1
        ):
            raise ValueError("state-count comparison minimum sequence length is invalid")
        if not _is_fingerprint(self.feature_contract_fingerprint):
            raise ValueError("state-count comparison feature contract fingerprint is invalid")
        if not isinstance(self.selection, StateCountSelection):
            raise TypeError("state-count comparison selection has an invalid type")
        expected_higher_is_better = self.model_kind == "gaussian-hmm-state"
        if self.selection.higher_is_better != expected_higher_is_better:
            raise ValueError("state-count comparison selection direction is inconsistent")
        folds = tuple(self.folds)
        if len(folds) != self.selection.n_folds or any(
            not isinstance(fold, FoldStateModelScores) for fold in folds
        ):
            raise ValueError("state-count comparison folds do not match selection")
        for index, fold in enumerate(folds):
            if fold.candidate_n_states != tuple(self.selection.candidate_n_states):
                raise ValueError("state-count comparison candidate state counts are misaligned")
            if len(fold.fold_scores) != self.selection.n_candidates or not np.allclose(
                fold.fold_scores,
                self.selection.fold_scores[index],
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError("state-count comparison fold scores are misaligned")
        model_seeds = folds[0].model_seeds
        if any(fold.model_seeds != model_seeds for fold in folds[1:]):
            raise ValueError("state-count comparison folds must use the same model seeds")
        subjects = tuple(subject for fold in folds for subject in fold.evaluation_subjects)
        if len(set(subjects)) != len(subjects):
            raise ValueError("state-count comparison evaluation subjects overlap")
        development_subjects = set(folds[0].fit_subjects).union(
            folds[0].evaluation_subjects
        )
        if any(
            set(fold.fit_subjects).union(fold.evaluation_subjects)
            != development_subjects
            for fold in folds[1:]
        ):
            raise ValueError(
                "state-count comparison folds must partition one development cohort"
            )
        if set(subjects) != development_subjects:
            missing = sorted(development_subjects.difference(subjects))
            raise ValueError(
                "state-count comparison must evaluate every development subject "
                f"exactly once; missing: {missing}"
            )
        fingerprints = tuple(
            fingerprint
            for fold in folds
            for candidate in fold.candidates
            for fingerprint in candidate.model_fingerprints
        )
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("state-count comparison fitted models are not unique")
        comparison_specification = dict(self.model_specification)
        try:
            json.dumps(comparison_specification, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("comparison specification must be finite JSON") from error
        if not comparison_specification:
            raise ValueError("comparison specification must be non-empty")
        for fold in folds:
            if any(
                _comparison_specification_values(
                    candidate_specification,
                    self.model_kind,
                )
                != comparison_specification
                for candidate in fold.candidates
                for candidate_specification in candidate.model_specifications
            ):
                raise ValueError(
                    "state-count comparison candidate model specifications are inconsistent"
                )
        object.__setattr__(
            self,
            "model_specification",
            MappingProxyType(comparison_specification),
        )
        object.__setattr__(self, "folds", folds)

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(subject for fold in self.folds for subject in fold.evaluation_subjects)

    @property
    def model_seeds(self) -> tuple[int, ...]:
        return self.folds[0].model_seeds


def _sample_interval_matches(
    left: float | None,
    right: float | None,
) -> bool:
    return (left is None and right is None) or (
        left is not None
        and right is not None
        and np.isclose(left, right, rtol=0.0, atol=1e-9)
    )


def _run_boundaries(report: StateModelScoreReport) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            score.subject,
            score.session,
            score.acquisition_id,
            score.n_samples,
            score.n_sequences,
        )
        for score in report.scores
    )


def _subject_scores(report: StateModelScoreReport) -> tuple[float, ...]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for score in report.scores:
        total = (
            score.total_squared_distance
            if isinstance(score, RunKMeansScore)
            else score.log_likelihood
        )
        totals[score.subject] = totals.get(score.subject, 0.0) + total
        counts[score.subject] = counts.get(score.subject, 0) + score.n_samples
    return tuple(totals[subject] / counts[subject] for subject in report.subjects)


def _comparison_specification_values(
    specification: Mapping[str, object],
    model_kind: str,
) -> dict[str, object]:
    output = dict(specification)
    if model_kind == "kmeans-state":
        # The effective initialization pool may be capped by fold sample count
        # or scale with K. Preserve its actual value per fit, but compare the
        # shared optimizer and dependency contract independently of that cap.
        if "init_sample_size" not in output or "implementation" not in output:
            raise ValueError("KMeans comparison specification is incomplete")
        output.pop("init_sample_size")
        implementation = str(output["implementation"])
        output["implementation"] = implementation.split(
            "; kmeans++ sample=",
            maxsplit=1,
        )[0]
    return output


def _comparison_specification(report: StateModelScoreReport) -> dict[str, object]:
    return _comparison_specification_values(report.model_specification, report.model_kind)


def compare_state_model_scores(
    reports: Sequence[StateModelScoreReport],
    *,
    score_artifacts: Sequence[str | Path] | None = None,
) -> StateCountComparison:
    """Compare a complete candidate-state set across subject-disjoint folds."""
    reports = tuple(reports)
    if len(reports) < 4:
        raise ValueError(
            "state-count comparison requires at least two candidates in two folds"
        )
    if any(not isinstance(report, StateModelScoreReport) for report in reports):
        raise TypeError("reports must contain StateModelScoreReport objects")
    if any(report.format_version < 2 or report.model_specification is None for report in reports):
        raise ValueError("state-count comparison requires v2 score artifacts")
    if any(report.allow_fit_subjects for report in reports):
        raise ValueError("state-count comparison does not accept fit-subject scoring")
    if any(report.training_data_fingerprint is None for report in reports):
        raise ValueError("state-count comparison requires training-data fingerprints")

    if score_artifacts is None:
        artifacts = tuple("" for _ in reports)
    else:
        artifacts = tuple(str(Path(path)) for path in score_artifacts)
        if len(artifacts) != len(reports):
            raise ValueError("score_artifacts must match the report count")
        if len(set(artifacts)) != len(artifacts):
            raise ValueError("score_artifacts must be distinct")

    reference = reports[0]
    for report in reports[1:]:
        if report.model_kind != reference.model_kind:
            raise ValueError("state-count comparison requires one model family")
        if report.source_contract != reference.source_contract:
            raise ValueError("state-count comparison requires one source contract")
        if report.feature_contract_fingerprint != reference.feature_contract_fingerprint:
            raise ValueError("state-count comparison requires one exact feature contract")
        if not _sample_interval_matches(
            report.sample_interval_seconds,
            reference.sample_interval_seconds,
        ):
            raise ValueError("state-count comparison requires one sample interval")
        if report.minimum_sequence_length != reference.minimum_sequence_length:
            raise ValueError("state-count comparison requires one minimum sequence length")
        if _comparison_specification(report) != _comparison_specification(reference):
            raise ValueError(
                "state-count comparison requires identical non-state-count model specifications"
            )
    fingerprints = tuple(report.model_fingerprint for report in reports)
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("state-count comparison requires distinct fitted models")

    grouped: dict[str, list[tuple[StateModelScoreReport, str]]] = {}
    for report, artifact in zip(reports, artifacts, strict=True):
        grouped.setdefault(report.evaluation_data_fingerprint, []).append((report, artifact))
    if len(grouped) < 2:
        raise ValueError("state-count comparison requires at least two folds")

    candidate_grids = []
    for group in grouped.values():
        by_state: dict[int, list[StateModelScoreReport]] = {}
        for report, _ in group:
            by_state.setdefault(report.n_states, []).append(report)
        states = tuple(sorted(by_state))
        if len(states) < 2:
            raise ValueError(
                "each validation fold requires scores for at least two state counts"
            )
        seed_sets = []
        for reports_for_state in by_state.values():
            seeds = tuple(sorted(report.model_seed for report in reports_for_state))
            if len(set(seeds)) != len(seeds):
                raise ValueError(
                    "each validation-fold state count requires unique model seeds"
                )
            seed_sets.append(seeds)
        model_seeds = seed_sets[0]
        if any(seeds != model_seeds for seeds in seed_sets[1:]):
            raise ValueError("every state count in a fold must use the same model seeds")
        candidate_grids.append((states, model_seeds))
    candidates, model_seeds = candidate_grids[0]
    if any(grid != candidate_grids[0] for grid in candidate_grids[1:]):
        raise ValueError(
            "every validation fold must contain the same state-count and seed grid"
        )

    folds: list[FoldStateModelScores] = []
    seen_evaluation_subjects: set[str] = set()
    for evaluation_fingerprint in sorted(grouped):
        ordered = sorted(
            grouped[evaluation_fingerprint],
            key=lambda item: (item[0].n_states, item[0].model_seed),
        )
        fold_reference = ordered[0][0]
        for report, _ in ordered[1:]:
            if report.training_data_fingerprint != fold_reference.training_data_fingerprint:
                raise ValueError("candidate models in one fold must use the same training data")
            if report.fit_subjects != fold_reference.fit_subjects:
                raise ValueError("candidate models in one fold must use the same training subjects")
            if report.subjects != fold_reference.subjects:
                raise ValueError("candidate models in one fold must score the same subjects")
            if _run_boundaries(report) != _run_boundaries(fold_reference):
                raise ValueError("candidate models in one fold must score identical run boundaries")
            if (
                report.omitted_short_sequence_count
                != fold_reference.omitted_short_sequence_count
            ):
                raise ValueError("candidate models in one fold disagree on omitted sequences")
        overlap = sorted(seen_evaluation_subjects.intersection(fold_reference.subjects))
        if overlap:
            raise ValueError(f"fold evaluation subjects overlap: {overlap}")
        seen_evaluation_subjects.update(fold_reference.subjects)
        fold_candidates = []
        for n_states in candidates:
            seed_reports = tuple(
                (report, artifact)
                for report, artifact in ordered
                if report.n_states == n_states
            )
            if tuple(report.model_seed for report, _ in seed_reports) != model_seeds:
                raise ValueError("fold candidate seed order is inconsistent")
            seed_subject_scores = tuple(
                _subject_scores(report) for report, _ in seed_reports
            )
            subject_scores = tuple(
                float(value) for value in np.mean(seed_subject_scores, axis=0)
            )
            seed_fold_scores = tuple(
                float(np.mean(values)) for values in seed_subject_scores
            )
            fold_candidates.append(
                CandidateStateModelScores(
                    n_states=n_states,
                    model_seeds=model_seeds,
                    model_fingerprints=tuple(
                        report.model_fingerprint for report, _ in seed_reports
                    ),
                    model_specifications=tuple(
                        dict(report.model_specification) for report, _ in seed_reports
                    ),
                    score_artifacts=tuple(
                        artifact for _, artifact in seed_reports
                    ),
                    subject_scores_by_seed=seed_subject_scores,
                    subject_scores=subject_scores,
                    fold_scores_by_seed=seed_fold_scores,
                    fold_score=float(np.mean(subject_scores)),
                )
            )
        folds.append(
            FoldStateModelScores(
                evaluation_data_fingerprint=evaluation_fingerprint,
                training_data_fingerprint=str(fold_reference.training_data_fingerprint),
                fit_subjects=fold_reference.fit_subjects,
                evaluation_subjects=fold_reference.subjects,
                n_runs=fold_reference.n_runs,
                n_sequences=fold_reference.n_sequences,
                n_samples=fold_reference.n_samples,
                omitted_short_sequence_count=fold_reference.omitted_short_sequence_count,
                candidates=tuple(fold_candidates),
            )
        )

    higher_is_better = reference.model_kind == "gaussian-hmm-state"
    selection = select_state_count(
        np.asarray([fold.fold_scores for fold in folds], dtype=float),
        list(candidates),
        higher_is_better=higher_is_better,
    )
    return StateCountComparison(
        model_kind=reference.model_kind,
        selection_metric=(
            "subject-balanced mean squared distance"
            if reference.model_kind == "kmeans-state"
            else "subject-balanced log likelihood per sample"
        ),
        selection_direction="higher is better" if higher_is_better else "lower is better",
        source_contract=reference.source_contract,
        sample_interval_seconds=reference.sample_interval_seconds,
        minimum_sequence_length=reference.minimum_sequence_length,
        feature_contract_fingerprint=str(reference.feature_contract_fingerprint),
        model_specification=_comparison_specification(reference),
        selection=selection,
        folds=tuple(folds),
    )


def state_count_comparison_payload(comparison: StateCountComparison) -> dict[str, object]:
    """Convert a validated comparison to strict JSON-compatible values."""
    if not isinstance(comparison, StateCountComparison):
        raise TypeError("comparison must be a StateCountComparison")
    selection = comparison.selection
    candidates = []
    for index, n_states in enumerate(selection.candidate_n_states):
        candidates.append(
            {
                "n_states": int(n_states),
                "mean_score": float(selection.mean_scores[index]),
                "fold_standard_deviation": float(
                    selection.fold_standard_deviations[index]
                ),
                "fold_standard_error": float(selection.fold_standard_errors[index]),
                "rank": int(selection.ranks[index]),
                "within_one_standard_error": bool(
                    selection.within_one_standard_error[index]
                ),
                "fold_scores": selection.fold_scores[:, index].tolist(),
            }
        )
    folds = []
    for fold_index, fold in enumerate(comparison.folds):
        fold_candidates = []
        for candidate in fold.candidates:
            seed_payloads = []
            for seed_index, model_seed in enumerate(candidate.model_seeds):
                seed_payloads.append(
                    {
                        "model_seed": model_seed,
                        "model_fingerprint": candidate.model_fingerprints[seed_index],
                        "model_specification": dict(
                            candidate.model_specifications[seed_index]
                        ),
                        "score_artifact": (
                            candidate.score_artifacts[seed_index] or None
                        ),
                        "fold_score": candidate.fold_scores_by_seed[seed_index],
                        "subject_scores": [
                            {"subject": subject, "score": score}
                            for subject, score in zip(
                                fold.evaluation_subjects,
                                candidate.subject_scores_by_seed[seed_index],
                                strict=True,
                            )
                        ],
                    }
                )
            fold_candidates.append(
                {
                    "n_states": candidate.n_states,
                    "fold_score": candidate.fold_score,
                    "subject_scores": [
                        {"subject": subject, "score": score}
                        for subject, score in zip(
                            fold.evaluation_subjects,
                            candidate.subject_scores,
                            strict=True,
                        )
                    ],
                    "seeds": seed_payloads,
                }
            )
        folds.append(
            {
                "fold_index": fold_index,
                "evaluation_data_fingerprint": fold.evaluation_data_fingerprint,
                "training_data_fingerprint": fold.training_data_fingerprint,
                "model_seeds": list(fold.model_seeds),
                "fit_subjects": list(fold.fit_subjects),
                "evaluation_subjects": list(fold.evaluation_subjects),
                "n_runs": fold.n_runs,
                "n_sequences": fold.n_sequences,
                "n_samples": fold.n_samples,
                "omitted_short_sequence_count": fold.omitted_short_sequence_count,
                "candidates": fold_candidates,
            }
        )
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model_kind": comparison.model_kind,
        "selection_metric": comparison.selection_metric,
        "selection_direction": comparison.selection_direction,
        "score_weighting": (
            "runs sample-weighted within subject and seed; seeds equal within "
            "subject; subjects equal within fold; folds equal"
        ),
        "uncertainty": "sample SD and SE across fold scores (ddof=1)",
        "source_contract": comparison.source_contract,
        "sample_interval_seconds": comparison.sample_interval_seconds,
        "minimum_sequence_length": comparison.minimum_sequence_length,
        "feature_contract_fingerprint": comparison.feature_contract_fingerprint,
        "comparison_specification": dict(comparison.model_specification),
        "n_folds": selection.n_folds,
        "n_candidates": selection.n_candidates,
        "n_seeds": len(comparison.model_seeds),
        "model_seeds": list(comparison.model_seeds),
        "n_subjects": len(comparison.subjects),
        "subjects": list(comparison.subjects),
        "candidate_n_states": selection.candidate_n_states.tolist(),
        "best_n_states": selection.best_n_states,
        "one_standard_error_n_states": selection.one_standard_error_n_states,
        "one_standard_error_threshold": selection.one_standard_error_threshold,
        "candidates": candidates,
        "folds": folds,
    }


def write_state_count_comparison(
    comparison: StateCountComparison,
    path: str | Path,
) -> Path:
    """Atomically write a strict state-count comparison JSON artifact."""
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"state-count comparison output path already exists: {target}")
    payload = state_count_comparison_payload(comparison)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"state-count comparison output path already exists: {target}"
            )
        os.rename(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "CandidateStateModelScores",
    "FoldStateModelScores",
    "StateCountComparison",
    "compare_state_model_scores",
    "state_count_comparison_payload",
    "write_state_count_comparison",
]
