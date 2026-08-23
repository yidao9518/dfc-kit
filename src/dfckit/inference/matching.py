"""Within-participant matching on continuous covariates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._arrays import readonly_copy as _readonly
from .._validation import validated_subject_labels


def _validated_covariates(values: ArrayLike, *, label: str) -> NDArray[np.float64]:
    output = np.asarray(values, dtype=float)
    if output.ndim == 1:
        output = output[:, None]
    if output.ndim != 2 or not len(output) or not output.shape[1]:
        raise ValueError(f"{label} must be a non-empty observations-by-covariates array")
    if not np.isfinite(output).all():
        raise ValueError(f"{label} contain non-finite values")
    return output


@dataclass(frozen=True)
class WindowMotionSummary:
    """Raw framewise-displacement summaries for fixed original-frame windows."""

    mean: NDArray[np.float64]
    p95: NDArray[np.float64]
    maximum: NDArray[np.float64]
    start_frames: NDArray[np.int64]
    end_frames: NDArray[np.int64]
    finite_frame_counts: NDArray[np.int64]

    @property
    def covariates(self) -> NDArray[np.float64]:
        return _readonly(np.column_stack([self.mean, self.p95]))


@dataclass(frozen=True)
class WithinSubjectMatchResult:
    """One-to-one matched observation indices aligned by row."""

    left_indices: NDArray[np.int64]
    right_indices: NDArray[np.int64]
    subjects: tuple[str, ...]
    absolute_differences: NDArray[np.float64]
    scaled_costs: NDArray[np.float64]
    covariate_names: tuple[str, ...]
    calipers: NDArray[np.float64]
    ceilings: NDArray[np.float64]
    minimum_pairs: int
    n_left_observations: int
    n_right_observations: int
    implementation: str

    @property
    def n_pairs(self) -> int:
        return len(self.left_indices)


@dataclass(frozen=True)
class MatchedSubjectDifferences:
    """Right-minus-left endpoint means over matched pairs within participant."""

    subjects: tuple[str, ...]
    differences: NDArray[np.float64]
    pair_counts: NDArray[np.int64]
    endpoint_names: tuple[str, ...]


def summarize_window_motion(
    framewise_displacement: ArrayLike,
    start_frames: ArrayLike,
    end_frames: ArrayLike,
    *,
    maximum_nonfinite: int = 1,
) -> WindowMotionSummary:
    """Summarize full-length FD over inclusive original-frame windows."""
    fd = np.asarray(framewise_displacement, dtype=float)
    starts = np.asarray(start_frames, dtype=np.int64)
    ends = np.asarray(end_frames, dtype=np.int64)
    if fd.ndim != 1 or not len(fd):
        raise ValueError("framewise_displacement must be a non-empty one-dimensional array")
    if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape or not len(starts):
        raise ValueError("start_frames and end_frames must be aligned non-empty vectors")
    if np.any(starts < 0) or np.any(ends < starts) or np.any(ends >= len(fd)):
        raise ValueError("window frame bounds are invalid for framewise_displacement")
    if isinstance(maximum_nonfinite, (bool, np.bool_)) or not isinstance(
        maximum_nonfinite, (int, np.integer)
    ):
        raise TypeError("maximum_nonfinite must be an integer")
    if maximum_nonfinite < 0:
        raise ValueError("maximum_nonfinite cannot be negative")

    mean = np.empty(len(starts), dtype=float)
    p95 = np.empty(len(starts), dtype=float)
    maximum = np.empty(len(starts), dtype=float)
    counts = np.empty(len(starts), dtype=np.int64)
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        raw = fd[start : end + 1]
        finite = raw[np.isfinite(raw)]
        missing = len(raw) - len(finite)
        if missing > maximum_nonfinite or not len(finite):
            raise ValueError(
                f"window {index} contains {missing} non-finite FD values; "
                f"maximum allowed is {maximum_nonfinite}"
            )
        mean[index] = finite.mean()
        p95[index] = np.percentile(finite, 95)
        maximum[index] = finite.max()
        counts[index] = len(finite)
    return WindowMotionSummary(
        mean=_readonly(mean),
        p95=_readonly(p95),
        maximum=_readonly(maximum),
        start_frames=_readonly(starts),
        end_frames=_readonly(ends),
        finite_frame_counts=_readonly(counts),
    )


def match_within_subject(
    left_covariates: ArrayLike,
    left_subjects: Iterable[str],
    right_covariates: ArrayLike,
    right_subjects: Iterable[str],
    *,
    calipers: ArrayLike,
    covariate_names: Iterable[str] | None = None,
    ceilings: ArrayLike | None = None,
    minimum_pairs: int = 1,
) -> WithinSubjectMatchResult:
    """Match left and right observations one to one within each participant.

    Assignment first maximizes the number of pairs satisfying every caliper,
    then minimizes the sum of absolute differences divided by their calipers.
    """
    left = _validated_covariates(left_covariates, label="left_covariates")
    right = _validated_covariates(right_covariates, label="right_covariates")
    if left.shape[1] != right.shape[1]:
        raise ValueError("left and right must use the same number of covariates")
    left_ids = validated_subject_labels(left_subjects, n_observations=len(left))
    right_ids = validated_subject_labels(right_subjects, n_observations=len(right))
    widths = np.asarray(calipers, dtype=float)
    if widths.ndim == 0:
        widths = widths[None]
    if widths.ndim != 1 or len(widths) != left.shape[1]:
        raise ValueError("calipers must contain one value per covariate")
    if not np.isfinite(widths).all() or np.any(widths <= 0.0):
        raise ValueError("calipers must be finite and positive")
    if ceilings is None:
        limits = np.full(left.shape[1], np.inf, dtype=float)
    else:
        limits = np.asarray(ceilings, dtype=float)
        if limits.ndim == 0:
            limits = limits[None]
        if limits.ndim != 1 or len(limits) != left.shape[1]:
            raise ValueError("ceilings must contain one value per covariate")
        if np.any(np.isnan(limits)):
            raise ValueError("ceilings cannot contain NaN")
    if covariate_names is None:
        names = tuple(f"covariate-{index}" for index in range(left.shape[1]))
    else:
        names = tuple(str(name) for name in covariate_names)
        if len(names) != left.shape[1] or any(not name.strip() for name in names):
            raise ValueError("covariate_names must identify every covariate")
        if len(set(names)) != len(names):
            raise ValueError("covariate_names must be unique")
    if isinstance(minimum_pairs, (bool, np.bool_)) or not isinstance(
        minimum_pairs, (int, np.integer)
    ):
        raise TypeError("minimum_pairs must be an integer")
    if minimum_pairs < 1:
        raise ValueError("minimum_pairs must be at least one")
    try:
        import scipy
        from scipy.optimize import linear_sum_assignment
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "within-subject matching requires the 'inference' extra: "
            "pip install 'dfc-kit[inference]'"
        ) from error

    left_subject_array = np.asarray(left_ids, dtype=object)
    right_subject_array = np.asarray(right_ids, dtype=object)
    left_eligible = np.all(left <= limits, axis=1)
    right_eligible = np.all(right <= limits, axis=1)
    common_subjects = tuple(
        subject
        for subject in dict.fromkeys(left_ids)
        if subject in set(right_ids)
    )
    matched_left: list[int] = []
    matched_right: list[int] = []
    matched_subjects: list[str] = []
    matched_differences: list[NDArray[np.float64]] = []
    matched_costs: list[float] = []

    for subject in common_subjects:
        left_indices = np.flatnonzero(
            (left_subject_array == subject) & left_eligible
        )
        right_indices = np.flatnonzero(
            (right_subject_array == subject) & right_eligible
        )
        if not len(left_indices) or not len(right_indices):
            continue
        differences = np.abs(
            left[left_indices, None, :] - right[right_indices, :][None, :, :]
        )
        scaled = np.sum(differences / widths, axis=2)
        invalid = np.any(differences > widths, axis=2)
        assignment_size = min(len(left_indices), len(right_indices))
        maximum_valid_total = assignment_size * left.shape[1]
        invalid_penalty = float(maximum_valid_total + 1.0)
        assignment_cost = np.where(invalid, invalid_penalty, scaled)
        left_position, right_position = linear_sum_assignment(assignment_cost)
        valid = ~invalid[left_position, right_position]
        left_position = left_position[valid]
        right_position = right_position[valid]
        if len(left_position) < minimum_pairs:
            continue
        for left_item, right_item in zip(left_position, right_position, strict=True):
            matched_left.append(int(left_indices[left_item]))
            matched_right.append(int(right_indices[right_item]))
            matched_subjects.append(subject)
            matched_differences.append(differences[left_item, right_item])
            matched_costs.append(float(scaled[left_item, right_item]))

    difference_array = (
        np.stack(matched_differences)
        if matched_differences
        else np.empty((0, left.shape[1]), dtype=float)
    )
    return WithinSubjectMatchResult(
        left_indices=_readonly(np.asarray(matched_left, dtype=np.int64)),
        right_indices=_readonly(np.asarray(matched_right, dtype=np.int64)),
        subjects=tuple(matched_subjects),
        absolute_differences=_readonly(difference_array),
        scaled_costs=_readonly(np.asarray(matched_costs, dtype=float)),
        covariate_names=names,
        calipers=_readonly(widths),
        ceilings=_readonly(limits),
        minimum_pairs=int(minimum_pairs),
        n_left_observations=len(left),
        n_right_observations=len(right),
        implementation=f"scipy {scipy.__version__} linear_sum_assignment",
    )


def matched_subject_differences(
    matches: WithinSubjectMatchResult,
    left_endpoints: ArrayLike,
    right_endpoints: ArrayLike,
    *,
    endpoint_names: Iterable[str] | None = None,
) -> MatchedSubjectDifferences:
    """Average right-minus-left endpoint differences within each subject."""
    left = np.asarray(left_endpoints, dtype=float)
    right = np.asarray(right_endpoints, dtype=float)
    if left.ndim == 1:
        left = left[:, None]
    if right.ndim == 1:
        right = right[:, None]
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("left and right endpoints must be aligned two-dimensional arrays")
    if len(left) != matches.n_left_observations or len(right) != matches.n_right_observations:
        raise ValueError("endpoint row counts must match the observations used for matching")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("endpoint values contain non-finite samples")
    if endpoint_names is None:
        names = tuple(f"endpoint-{index}" for index in range(left.shape[1]))
    else:
        names = tuple(str(name) for name in endpoint_names)
        if len(names) != left.shape[1] or any(not name.strip() for name in names):
            raise ValueError("endpoint_names must identify every endpoint")
        if len(set(names)) != len(names):
            raise ValueError("endpoint_names must be unique")

    subjects = tuple(dict.fromkeys(matches.subjects))
    differences = np.empty((len(subjects), left.shape[1]), dtype=float)
    counts = np.empty(len(subjects), dtype=np.int64)
    match_subject_array = np.asarray(matches.subjects, dtype=object)
    for index, subject in enumerate(subjects):
        selected = np.flatnonzero(match_subject_array == subject)
        pairwise = (
            right[matches.right_indices[selected]]
            - left[matches.left_indices[selected]]
        )
        differences[index] = pairwise.mean(axis=0)
        counts[index] = len(selected)
    return MatchedSubjectDifferences(
        subjects=subjects,
        differences=_readonly(differences),
        pair_counts=_readonly(counts),
        endpoint_names=names,
    )
