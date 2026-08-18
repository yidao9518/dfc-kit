"""Fold-level selection of a state count from held-out scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import _readonly


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
    """Fold-level candidate scores and the one-standard-error decision."""

    candidate_n_states: NDArray[np.int64]
    fold_scores: NDArray[np.float64]
    mean_scores: NDArray[np.float64]
    fold_standard_deviations: NDArray[np.float64]
    fold_standard_errors: NDArray[np.float64]
    ranks: NDArray[np.int64]
    within_one_standard_error: NDArray[np.bool_]
    higher_is_better: bool
    best_n_states: int
    one_standard_error_n_states: int
    one_standard_error_threshold: float

    def __post_init__(self) -> None:
        raw_states = np.asarray(self.candidate_n_states)
        if raw_states.dtype.kind not in "iu" or raw_states.dtype.kind == "b":
            raise TypeError("candidate state counts must be integers")
        states = raw_states.astype(np.int64, copy=False)
        scores = np.asarray(self.fold_scores, dtype=float)
        means = np.asarray(self.mean_scores, dtype=float)
        deviations = np.asarray(self.fold_standard_deviations, dtype=float)
        errors = np.asarray(self.fold_standard_errors, dtype=float)
        raw_ranks = np.asarray(self.ranks)
        if raw_ranks.dtype.kind not in "iu" or raw_ranks.dtype.kind == "b":
            raise TypeError("state-count selection ranks must be integers")
        ranks = raw_ranks.astype(np.int64, copy=False)
        raw_within = np.asarray(self.within_one_standard_error)
        if raw_within.dtype.kind != "b":
            raise TypeError("within_one_standard_error must be boolean")
        within = raw_within.astype(bool, copy=False)
        if states.ndim != 1 or len(states) < 2 or len(set(states.tolist())) != len(states):
            raise ValueError("state-count selection requires at least two unique candidates")
        if np.any(states < 2) or np.any(states[1:] <= states[:-1]):
            raise ValueError("candidate state counts must be strictly increasing and at least two")
        if scores.ndim != 2 or scores.shape[0] < 2 or scores.shape[1] != len(states):
            raise ValueError("state-count selection requires folds-by-candidates scores")
        expected_shape = (len(states),)
        if any(
            values.shape != expected_shape
            for values in (means, deviations, errors, ranks, within)
        ):
            raise ValueError("state-count selection summaries do not match candidates")
        if not isinstance(self.higher_is_better, (bool, np.bool_)):
            raise TypeError("higher_is_better must be boolean")
        higher_is_better = bool(self.higher_is_better)
        if not np.isfinite(scores).all() or not np.isfinite(means).all():
            raise ValueError("state-count selection scores must be finite")
        if not np.isfinite(deviations).all() or not np.isfinite(errors).all():
            raise ValueError("state-count selection uncertainty must be finite")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in (self.best_n_states, self.one_standard_error_n_states)
        ):
            raise TypeError("selected state counts must be integers")
        best_n_states = int(self.best_n_states)
        one_standard_error_n_states = int(self.one_standard_error_n_states)
        try:
            threshold = float(self.one_standard_error_threshold)
        except (TypeError, ValueError) as error:
            raise TypeError("one-standard-error threshold must be numeric") from error
        if not np.isfinite(threshold):
            raise ValueError("one-standard-error threshold must be finite")
        expected = _selection_statistics(
            scores,
            states,
            higher_is_better=higher_is_better,
        )
        expected_means, expected_deviations, expected_errors = expected[:3]
        expected_ranks, expected_within = expected[3:5]
        expected_best, expected_one_se, expected_threshold = expected[5:]
        if not np.allclose(means, expected_means, rtol=1e-12, atol=1e-12):
            raise ValueError("state-count selection means disagree with fold scores")
        if not np.allclose(deviations, expected_deviations, rtol=1e-12, atol=1e-12):
            raise ValueError("state-count selection deviations disagree with fold scores")
        if not np.allclose(errors, expected_errors, rtol=1e-12, atol=1e-12):
            raise ValueError("state-count selection errors disagree with fold scores")
        if not np.array_equal(ranks, expected_ranks):
            raise ValueError("state-count selection ranks disagree with fold scores")
        if not np.array_equal(within, expected_within):
            raise ValueError("one-standard-error mask disagrees with fold scores")
        if best_n_states != expected_best:
            raise ValueError("best state count disagrees with fold scores")
        if one_standard_error_n_states != expected_one_se:
            raise ValueError("one-standard-error state count disagrees with fold scores")
        if not np.isclose(threshold, expected_threshold, rtol=1e-12, atol=1e-12):
            raise ValueError("one-standard-error threshold disagrees with fold scores")
        for name, values in {
            "candidate_n_states": states,
            "fold_scores": scores,
            "mean_scores": means,
            "fold_standard_deviations": deviations,
            "fold_standard_errors": errors,
            "ranks": ranks,
            "within_one_standard_error": within,
        }.items():
            object.__setattr__(self, name, _readonly(values))
        object.__setattr__(self, "higher_is_better", higher_is_better)
        object.__setattr__(self, "best_n_states", best_n_states)
        object.__setattr__(self, "one_standard_error_n_states", one_standard_error_n_states)
        object.__setattr__(self, "one_standard_error_threshold", threshold)

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

    Rows are subject-disjoint validation folds and columns are candidate state counts. Every fold is
    weighted equally. The best candidate optimizes the mean score, with the
    smaller state count breaking exact ties. The one-standard-error rule then
    chooses the smallest candidate no worse than one standard error of the
    best candidate, where uncertainty is estimated across fold scores.
    """
    if not isinstance(higher_is_better, (bool, np.bool_)):
        raise TypeError("higher_is_better must be boolean")
    scores = np.asarray(fold_scores, dtype=float)
    raw_states = np.asarray(candidate_n_states)
    if raw_states.dtype.kind not in "iu" or raw_states.dtype.kind == "b":
        raise TypeError("candidate_n_states must contain integers")
    states = raw_states.astype(np.int64, copy=False)
    if states.ndim != 1 or len(states) < 2:
        raise ValueError("candidate_n_states must contain at least two values")
    if len(set(states.tolist())) != len(states) or np.any(states < 2):
        raise ValueError("candidate_n_states must be unique integers of at least two")
    if scores.ndim != 2 or scores.shape != (scores.shape[0], len(states)):
        raise ValueError("fold_scores must be a folds-by-candidates matrix")
    if scores.shape[0] < 2:
        raise ValueError("state-count selection requires at least two folds")
    if not np.isfinite(scores).all():
        raise ValueError("fold_scores must be finite")

    order = np.argsort(states)
    states = states[order]
    scores = scores[:, order]
    (
        means,
        deviations,
        errors,
        ranks,
        within,
        best_n_states,
        one_standard_error_n_states,
        threshold,
    ) = _selection_statistics(scores, states, higher_is_better=bool(higher_is_better))
    return StateCountSelection(
        candidate_n_states=states,
        fold_scores=scores,
        mean_scores=means,
        fold_standard_deviations=deviations,
        fold_standard_errors=errors,
        ranks=ranks,
        within_one_standard_error=within,
        higher_is_better=bool(higher_is_better),
        best_n_states=best_n_states,
        one_standard_error_n_states=one_standard_error_n_states,
        one_standard_error_threshold=threshold,
    )


__all__ = ["StateCountSelection", "select_state_count"]
