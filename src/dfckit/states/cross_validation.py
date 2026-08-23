"""Deterministic participant-level validation folds for state-model selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np


def _subject_label(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("cross-validation subjects must be non-empty strings")
    return value


def _nonnegative_seed(value: object) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError("cross-validation split seed must be a non-negative integer")
    return int(value)


@dataclass(frozen=True)
class SubjectValidationFold:
    """One training/validation partition of a complete development cohort."""

    fold_index: int
    fit_subjects: tuple[str, ...]
    evaluation_subjects: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.fold_index, (bool, np.bool_))
            or not isinstance(self.fold_index, (int, np.integer))
            or self.fold_index < 0
        ):
            raise ValueError("fold_index must be a non-negative integer")
        fit_subjects = tuple(_subject_label(value) for value in self.fit_subjects)
        evaluation_subjects = tuple(
            _subject_label(value) for value in self.evaluation_subjects
        )
        if not fit_subjects or not evaluation_subjects:
            raise ValueError("every validation fold requires fit and evaluation subjects")
        if len(set(fit_subjects)) != len(fit_subjects) or len(
            set(evaluation_subjects)
        ) != len(evaluation_subjects):
            raise ValueError("fold subject identities must be unique")
        if set(fit_subjects).intersection(evaluation_subjects):
            raise ValueError("fold fit and evaluation subjects must be disjoint")
        object.__setattr__(self, "fold_index", int(self.fold_index))
        object.__setattr__(self, "fit_subjects", fit_subjects)
        object.__setattr__(self, "evaluation_subjects", evaluation_subjects)


def make_subject_validation_folds(
    subjects: Sequence[str],
    *,
    n_folds: int,
    seed: int,
) -> tuple[SubjectValidationFold, ...]:
    """Create balanced, reproducible, participant-exclusive validation folds.

    Subjects are randomly permuted with the supplied seed and divided into
    contiguous balanced groups.
    """
    cohort = tuple(sorted(_subject_label(value) for value in subjects))
    if len(set(cohort)) != len(cohort):
        raise ValueError("cross-validation subjects must be unique")
    if (
        isinstance(n_folds, (bool, np.bool_))
        or not isinstance(n_folds, (int, np.integer))
        or n_folds < 2
    ):
        raise ValueError("n_folds must be an integer of at least two")
    n_folds = int(n_folds)
    if len(cohort) < n_folds:
        raise ValueError("n_folds cannot exceed the number of subjects")
    split_seed = _nonnegative_seed(seed)
    rng = np.random.default_rng(split_seed)
    ranked = tuple(cohort[index] for index in rng.permutation(len(cohort)))
    base_size, remainder = divmod(len(ranked), n_folds)
    evaluation_groups = []
    offset = 0
    for fold_index in range(n_folds):
        fold_size = base_size + int(fold_index < remainder)
        selected = set(ranked[offset : offset + fold_size])
        evaluation_groups.append(tuple(subject for subject in cohort if subject in selected))
        offset += fold_size
    folds = tuple(
        SubjectValidationFold(
            fold_index=fold_index,
            fit_subjects=tuple(
                subject for subject in cohort if subject not in set(evaluation_subjects)
            ),
            evaluation_subjects=evaluation_subjects,
        )
        for fold_index, evaluation_subjects in enumerate(evaluation_groups)
    )
    if tuple(fold.fold_index for fold in folds) != tuple(range(n_folds)) or any(
        right.fold_index <= left.fold_index for left, right in pairwise(folds)
    ):
        raise RuntimeError("cross-validation fold indices are inconsistent")
    observed = tuple(subject for fold in folds for subject in fold.evaluation_subjects)
    if len(observed) != len(cohort) or set(observed) != set(cohort):
        raise RuntimeError("cross-validation folds do not evaluate every subject exactly once")
    return folds


__all__ = ["SubjectValidationFold", "make_subject_validation_folds"]
