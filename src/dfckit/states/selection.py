"""State-count selection from held-out scores.

This module contains both the small numerical selector and the subject-disjoint
aggregation used to compare complete state-model score grids.  Keeping these
operations together makes the state-count decision part of the state API rather
than a separate validation workflow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .._arrays import readonly_copy as _readonly
from ..artifacts._fields import sample_intervals_match
from ..artifacts._json import write_json_atomic
from ..artifacts.state_scoring import StateModelScoreReport, _run_boundaries
from .scoring import RunKMeansScore


def _selection_statistics(
    scores: NDArray[np.float64],
    states: NDArray[np.int64],
    *,
    higher_is_better: bool,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.bool_],
    int,
    int,
    float,
]:
    means = scores.mean(axis=0)
    deviations = scores.std(axis=0, ddof=1)
    errors = deviations / np.sqrt(scores.shape[0])
    optimization_values = -means if higher_is_better else means
    ranking_order = np.lexsort((states, optimization_values))
    ranks = np.empty(len(states), dtype=np.int64)
    ranks[ranking_order] = np.arange(1, len(states) + 1)
    best_index = int(ranking_order[0])
    threshold = float(
        means[best_index] - errors[best_index]
        if higher_is_better
        else means[best_index] + errors[best_index]
    )
    within = means >= threshold if higher_is_better else means <= threshold
    one_standard_error_index = int(np.flatnonzero(within)[0])
    return (
        means,
        deviations,
        errors,
        ranks,
        within,
        int(states[best_index]),
        int(states[one_standard_error_index]),
        threshold,
    )


@dataclass(frozen=True)
class StateCountSelection:
    """Fold-level scores with summaries derived once from those observations."""

    candidate_n_states: NDArray[np.int64]
    fold_scores: NDArray[np.float64]
    higher_is_better: bool
    mean_scores: NDArray[np.float64] = field(init=False)
    fold_standard_deviations: NDArray[np.float64] = field(init=False)
    fold_standard_errors: NDArray[np.float64] = field(init=False)
    ranks: NDArray[np.int64] = field(init=False)
    within_one_standard_error: NDArray[np.bool_] = field(init=False)
    best_n_states: int = field(init=False)
    one_standard_error_n_states: int = field(init=False)
    one_standard_error_threshold: float = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.higher_is_better, (bool, np.bool_)):
            raise TypeError("higher_is_better must be boolean")
        scores = np.asarray(self.fold_scores, dtype=float)
        raw_states = np.asarray(self.candidate_n_states)
        if raw_states.dtype.kind not in "iu":
            raise TypeError("candidate_n_states must contain integers")
        states = raw_states.astype(np.int64, copy=False)
        if states.ndim != 1 or len(states) < 2:
            raise ValueError("candidate_n_states must contain at least two values")
        if len(set(states.tolist())) != len(states) or np.any(states < 2):
            raise ValueError("candidate_n_states must be unique integers of at least two")
        if scores.ndim != 2 or scores.shape[1] != len(states):
            raise ValueError("fold_scores must be a folds-by-candidates matrix")
        if scores.shape[0] < 2:
            raise ValueError("state-count selection requires at least two folds")
        if not np.isfinite(scores).all():
            raise ValueError("fold_scores must be finite")
        order = np.argsort(states)
        states, scores = states[order], scores[:, order]
        object.__setattr__(self, "candidate_n_states", _readonly(states))
        object.__setattr__(self, "fold_scores", _readonly(scores))
        object.__setattr__(self, "higher_is_better", bool(self.higher_is_better))
        statistics = _selection_statistics(scores, states, higher_is_better=self.higher_is_better)
        names = (
            "mean_scores", "fold_standard_deviations", "fold_standard_errors",
            "ranks", "within_one_standard_error", "best_n_states",
            "one_standard_error_n_states", "one_standard_error_threshold",
        )
        for name, value in zip(names, statistics, strict=True):
            object.__setattr__(self, name, _readonly(value) if isinstance(value, np.ndarray) else value)

    @property
    def n_folds(self) -> int:
        return int(self.fold_scores.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.fold_scores.shape[1])


def select_state_count(
    fold_scores: NDArray[np.floating] | list[list[float]],
    candidate_n_states: NDArray[np.integer] | list[int],
    *,
    higher_is_better: bool,
) -> StateCountSelection:
    """Rank candidates and apply the fold-level one-standard-error rule.

    Rows are subject-disjoint validation folds and columns are candidate state
    counts. Folds have equal weight, and smaller K breaks exact score ties.
    Select the smallest K within one standard error of the best candidate.
    """
    return StateCountSelection(candidate_n_states, fold_scores, higher_is_better)


def _subject_scores(report: StateModelScoreReport) -> tuple[float, ...]:
    totals: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    for score in report.scores:
        total = (
            score.total_squared_distance
            if isinstance(score, RunKMeansScore)
            else score.log_likelihood
        )
        totals[score.subject] = totals.get(score.subject, 0.0) + total
        sample_counts[score.subject] = sample_counts.get(score.subject, 0) + score.n_samples
    return tuple(totals[subject] / sample_counts[subject] for subject in report.subjects)


def _comparison_specification(report: StateModelScoreReport) -> dict[str, object]:
    specification = dict(report.model_specification)
    specification.pop("init_sample_size", None)
    return specification


@dataclass(frozen=True)
class CandidateStateModelScores:
    """Repeated-seed validation scores for one candidate K in one fold."""

    n_states: int
    model_seeds: tuple[int, ...]
    subject_scores_by_seed: tuple[tuple[float, ...], ...]
    subject_scores: tuple[float, ...]
    fold_scores_by_seed: tuple[float, ...]
    fold_score: float


@dataclass(frozen=True)
class FoldStateModelScores:
    """Scores for all candidate state counts in one validation fold."""

    fit_subjects: tuple[str, ...]
    evaluation_subjects: tuple[str, ...]
    n_runs: int
    n_sequences: int
    n_samples: int
    omitted_short_sequence_count: int
    candidates: tuple[CandidateStateModelScores, ...]

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
    """Cross-fold scores and the resulting state-count choice."""

    model_kind: str
    selection_metric: str
    selection_direction: str
    source_contract: str
    sample_interval_seconds: float | None
    minimum_sequence_length: int
    model_specification: Mapping[str, object]
    selection: StateCountSelection
    folds: tuple[FoldStateModelScores, ...]

    def __post_init__(self) -> None:
        if self.model_kind not in {"kmeans-state", "gaussian-hmm-state"}:
            raise ValueError("unsupported state model kind")
        if len(self.folds) < 2:
            raise ValueError("state-count comparison requires at least two folds")
        candidates = self.folds[0].candidate_n_states
        seeds = self.folds[0].model_seeds
        if any(
            fold.candidate_n_states != candidates or fold.model_seeds != seeds
            for fold in self.folds[1:]
        ):
            raise ValueError("all folds must use the same candidate K and seed grid")
        seen: set[str] = set()
        for fold in self.folds:
            if set(fold.fit_subjects).intersection(fold.evaluation_subjects):
                raise ValueError("fit and validation subjects must be disjoint")
            overlap = seen.intersection(fold.evaluation_subjects)
            if overlap:
                raise ValueError(f"validation subjects appear in multiple folds: {sorted(overlap)}")
            seen.update(fold.evaluation_subjects)
        object.__setattr__(
            self,
            "model_specification",
            MappingProxyType(dict(self.model_specification)),
        )

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(subject for fold in self.folds for subject in fold.evaluation_subjects)

    @property
    def model_seeds(self) -> tuple[int, ...]:
        return self.folds[0].model_seeds


def compare_state_model_scores(
    reports: Sequence[StateModelScoreReport],
) -> StateCountComparison:
    """Compare candidate state counts across subject-disjoint folds."""
    reports = tuple(reports)
    if len(reports) < 4 or any(
        not isinstance(report, StateModelScoreReport) for report in reports
    ):
        raise ValueError("at least two candidates in two folds are required")
    if any(report.allow_fit_subjects for report in reports):
        raise ValueError("state-count selection requires held-out scores")

    reference = reports[0]
    reference_specification = _comparison_specification(reference)
    for report in reports[1:]:
        if (
            report.model_kind != reference.model_kind
            or report.source_contract != reference.source_contract
            or not sample_intervals_match(
                report.sample_interval_seconds,
                reference.sample_interval_seconds,
            )
            or report.minimum_sequence_length != reference.minimum_sequence_length
            or _comparison_specification(report) != reference_specification
        ):
            raise ValueError("all score reports must use the same model and feature settings")

    grouped: dict[tuple[str, ...], list[StateModelScoreReport]] = {}
    for report in reports:
        grouped.setdefault(report.subjects, []).append(report)
    if len(grouped) < 2:
        raise ValueError("at least two validation folds are required")

    folds: list[FoldStateModelScores] = []
    expected_grid: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    for evaluation_subjects in sorted(grouped):
        group = sorted(grouped[evaluation_subjects], key=lambda item: (item.n_states, item.model_seed))
        fold_reference = group[0]
        for report in group[1:]:
            if (
                report.fit_subjects != fold_reference.fit_subjects
                or report.subjects != fold_reference.subjects
                or _run_boundaries(report) != _run_boundaries(fold_reference)
                or report.omitted_short_sequence_count
                != fold_reference.omitted_short_sequence_count
            ):
                raise ValueError("reports within a fold must score the same held-out data")

        candidates = tuple(sorted({report.n_states for report in group}))
        if len(candidates) < 2:
            raise ValueError("each fold requires at least two candidate state counts")
        seed_sets = {
            n_states: tuple(
                sorted(report.model_seed for report in group if report.n_states == n_states)
            )
            for n_states in candidates
        }
        model_seeds = seed_sets[candidates[0]]
        if (
            len(set(model_seeds)) != len(model_seeds)
            or any(seeds != model_seeds for seeds in seed_sets.values())
        ):
            raise ValueError("each fold and candidate must use the same unique seeds")
        grid = (candidates, model_seeds)
        if expected_grid is None:
            expected_grid = grid
        elif grid != expected_grid:
            raise ValueError("all folds must use the same candidate K and seed grid")

        fold_candidates = []
        for n_states in candidates:
            seed_reports = tuple(report for report in group if report.n_states == n_states)
            seed_subject_scores = tuple(_subject_scores(report) for report in seed_reports)
            subject_scores = tuple(np.mean(seed_subject_scores, axis=0).tolist())
            seed_fold_scores = tuple(float(np.mean(values)) for values in seed_subject_scores)
            fold_candidates.append(
                CandidateStateModelScores(
                    n_states=n_states,
                    model_seeds=model_seeds,
                    subject_scores_by_seed=seed_subject_scores,
                    subject_scores=subject_scores,
                    fold_scores_by_seed=seed_fold_scores,
                    fold_score=float(np.mean(subject_scores)),
                )
            )
        folds.append(
            FoldStateModelScores(
                fit_subjects=fold_reference.fit_subjects,
                evaluation_subjects=evaluation_subjects,
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
        list(expected_grid[0]),
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
        model_specification=reference_specification,
        selection=selection,
        folds=tuple(folds),
    )


def _state_count_comparison_payload(comparison: StateCountComparison) -> dict[str, object]:
    """Convert a state-count comparison to JSON-compatible values."""
    selection = comparison.selection
    candidates = [
        {
            "n_states": int(n_states),
            "mean_score": float(selection.mean_scores[index]),
            "standard_error": float(selection.fold_standard_errors[index]),
            "rank": int(selection.ranks[index]),
            "within_one_standard_error": bool(selection.within_one_standard_error[index]),
            "fold_scores": selection.fold_scores[:, index].tolist(),
        }
        for index, n_states in enumerate(selection.candidate_n_states)
    ]
    return {
        "format": "dfckit-state-count-comparison",
        "format_version": 3,
        "model_kind": comparison.model_kind,
        "selection_metric": comparison.selection_metric,
        "selection_direction": comparison.selection_direction,
        "source_contract": comparison.source_contract,
        "sample_interval_seconds": comparison.sample_interval_seconds,
        "minimum_sequence_length": comparison.minimum_sequence_length,
        "model_specification": dict(comparison.model_specification),
        "n_folds": selection.n_folds,
        "model_seeds": list(comparison.model_seeds),
        "subjects": list(comparison.subjects),
        "best_n_states": selection.best_n_states,
        "one_standard_error_n_states": selection.one_standard_error_n_states,
        "candidates": candidates,
    }


def _write_state_count_comparison(
    comparison: StateCountComparison,
    path: str | Path,
) -> Path:
    """Write a state-count comparison as JSON."""
    return write_json_atomic(path, _state_count_comparison_payload(comparison))


__all__ = [
    "CandidateStateModelScores",
    "FoldStateModelScores",
    "StateCountComparison",
    "StateCountSelection",
    "compare_state_model_scores",
    "select_state_count",
]
