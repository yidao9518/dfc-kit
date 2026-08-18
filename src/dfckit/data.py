"""Validated data structures for retained ROI time series."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .segments import contiguous_segments, window_positions


@dataclass(frozen=True)
class TimeWindow:
    """One fixed-length window contained within a retained-frame segment."""

    values: NDArray[np.float64]
    original_indices: NDArray[np.int64]
    segment_id: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class TimeSeriesRun:
    """ROI time series with their positions in the original acquisition."""

    values: ArrayLike
    original_indices: ArrayLike
    roi_names: tuple[str, ...]
    subject: str | None = None
    session: str | None = None
    tr: float | None = None
    acquisition_id: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        original_indices = np.asarray(self.original_indices, dtype=np.int64)
        roi_names = tuple(str(name) for name in self.roi_names)

        if values.ndim != 2:
            raise ValueError("values must be a two-dimensional frames-by-ROI array")
        if original_indices.ndim != 1:
            raise ValueError("original_indices must be one-dimensional")
        if len(values) != len(original_indices):
            raise ValueError("values and original_indices must contain the same frame count")
        if values.shape[1] != len(roi_names):
            raise ValueError("roi_names must match the number of time-series columns")
        if len(set(roi_names)) != len(roi_names):
            raise ValueError("roi_names must be unique")
        if len(original_indices) and np.any(np.diff(original_indices) <= 0):
            raise ValueError("original_indices must be strictly increasing")
        if len(original_indices) and original_indices[0] < 0:
            raise ValueError("original_indices cannot be negative")
        if not np.isfinite(values).all():
            raise ValueError("values contain non-finite samples")
        if self.tr is not None and (not np.isfinite(self.tr) or self.tr <= 0):
            raise ValueError("tr must be finite and positive")
        if self.acquisition_id is not None and not str(self.acquisition_id).strip():
            raise ValueError("acquisition_id cannot be empty")

        values = values.copy()
        original_indices = original_indices.copy()
        values.setflags(write=False)
        original_indices.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "original_indices", original_indices)
        object.__setattr__(self, "roi_names", roi_names)
        object.__setattr__(
            self,
            "acquisition_id",
            None if self.acquisition_id is None else str(self.acquisition_id),
        )

    @property
    def n_frames(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_rois(self) -> int:
        return int(self.values.shape[1])

    def segments(self, min_length: int = 1) -> tuple[NDArray[np.int64], ...]:
        """Return retained row positions for every contiguous original-frame run."""
        return contiguous_segments(self.original_indices, min_length=min_length)

    def windows(self, length: int, step: int) -> tuple[TimeWindow, ...]:
        """Generate fixed-length windows without crossing a censor gap."""
        windows = []
        for segment_id, positions in window_positions(self.original_indices, length, step):
            original = self.original_indices[positions]
            windows.append(
                TimeWindow(
                    values=self.values[positions],
                    original_indices=original,
                    segment_id=segment_id,
                    start_frame=int(original[0]),
                    end_frame=int(original[-1]),
                )
            )
        return tuple(windows)


@dataclass(frozen=True)
class TimeSeriesDataset:
    """Runs that share one ROI axis and temporal sampling contract.

    The dataset groups acquisitions by participant without assigning clinical
    meaning to sessions. Methods that learn parameters can use
    :meth:`require_subject_ids` and :func:`validate_subject_disjoint` to enforce
    participant-level train/test separation.
    """

    runs: Sequence[TimeSeriesRun]

    def __post_init__(self) -> None:
        runs = tuple(self.runs)
        if not runs:
            raise ValueError("a time-series dataset must contain at least one run")
        if any(not isinstance(run, TimeSeriesRun) for run in runs):
            raise TypeError("runs must contain only TimeSeriesRun objects")

        roi_names = runs[0].roi_names
        for index, run in enumerate(runs[1:], start=1):
            if run.roi_names != roi_names:
                raise ValueError(
                    f"run {index} has a different ROI identity or order from run 0"
                )

        trs = tuple(run.tr for run in runs)
        known_trs = tuple(tr for tr in trs if tr is not None)
        if known_trs and len(known_trs) != len(trs):
            raise ValueError("tr must be set for every run or omitted for every run")
        if known_trs and any(
            not np.isclose(tr, known_trs[0], rtol=0.0, atol=1e-9)
            for tr in known_trs[1:]
        ):
            raise ValueError("all runs in a dataset must use the same tr")

        acquisition_keys: set[tuple[str, str | None, str | None]] = set()
        for index, run in enumerate(runs):
            if run.subject is not None and not str(run.subject).strip():
                raise ValueError(f"run {index} has an empty subject identifier")
            if run.session is not None and not str(run.session).strip():
                raise ValueError(f"run {index} has an empty session identifier")
            if run.session is not None and run.subject is None:
                raise ValueError(f"run {index} has a session but no subject identifier")
            if run.acquisition_id is not None and run.subject is None:
                raise ValueError(
                    f"run {index} has an acquisition_id but no subject identifier"
                )
            if run.subject is None:
                continue
            key = (
                str(run.subject),
                None if run.session is None else str(run.session),
                run.acquisition_id,
            )
            if key in acquisition_keys:
                raise ValueError(
                    "duplicate subject/session acquisition identity: "
                    f"subject={key[0]!r}, session={key[1]!r}, "
                    f"acquisition_id={key[2]!r}"
                )
            acquisition_keys.add(key)

        object.__setattr__(self, "runs", runs)

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def roi_names(self) -> tuple[str, ...]:
        return self.runs[0].roi_names

    @property
    def tr(self) -> float | None:
        return self.runs[0].tr

    @property
    def has_complete_subject_ids(self) -> bool:
        return all(run.subject is not None for run in self.runs)

    @property
    def subjects(self) -> tuple[str, ...]:
        """Return unique known subject identifiers in first-seen order."""
        return tuple(dict.fromkeys(run.subject for run in self.runs if run.subject is not None))

    def require_subject_ids(self, operation: str = "learned operation") -> tuple[str, ...]:
        """Require every run to identify its participant and return those IDs."""
        missing = tuple(index for index, run in enumerate(self.runs) if run.subject is None)
        if missing:
            raise ValueError(f"{operation} requires subject IDs; missing for run indices {missing}")
        return self.subjects

    def runs_by_subject(self) -> Mapping[str, tuple[TimeSeriesRun, ...]]:
        """Group runs by participant while preserving dataset order."""
        self.require_subject_ids("subject grouping")
        grouped: dict[str, list[TimeSeriesRun]] = {}
        for run in self.runs:
            assert run.subject is not None
            grouped.setdefault(run.subject, []).append(run)
        frozen = {subject: tuple(subject_runs) for subject, subject_runs in grouped.items()}
        return MappingProxyType(frozen)


def validate_subject_disjoint(
    train_runs: Iterable[TimeSeriesRun],
    test_runs: Iterable[TimeSeriesRun],
) -> None:
    """Reject a split that places any participant in both run collections."""

    def subject_set(runs: Iterable[TimeSeriesRun], label: str) -> set[str]:
        output: set[str] = set()
        for index, run in enumerate(runs):
            if not isinstance(run, TimeSeriesRun):
                raise TypeError(f"{label} runs must contain only TimeSeriesRun objects")
            if run.subject is None:
                raise ValueError(f"{label} run {index} has no subject identifier")
            output.add(run.subject)
        if not output:
            raise ValueError(f"{label} runs cannot be empty")
        return output

    train_subjects = subject_set(train_runs, "training")
    test_subjects = subject_set(test_runs, "test")
    overlap = sorted(train_subjects.intersection(test_subjects))
    if overlap:
        raise ValueError(f"training and test subjects overlap: {overlap}")
