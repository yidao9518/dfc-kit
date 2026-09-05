"""Portable fixed-length MI/CMI analysis artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .._arrays import readonly_copy as _readonly
from .._validation import validated_integer as _integer
from ..data import TimeSeriesDataset, TimeSeriesRun
from .estimators import (
    FixedWindowSamples,
    _eligible_fixed_window_count,
    _estimate_information_blocks,
    _information_dependencies,
    sample_fixed_windows,
)

FORMAT_NAME = "dfckit-fixed-information"
FORMAT_VERSION = 1
GROUP_FIELDS = ("conditioning", "left", "right")
SCHEDULE_FIELDS = ("acquisition_id", "length", "draw", "start_frame", "end_frame")
BASE_ARRAY_NAMES = frozenset(
    {
        "acquisition_index",
        "draw",
        "eligible_starts",
        "end_frame",
        "length",
        "mean_mutual_information",
        "mutual_information",
        "segment_id",
        "start_frame",
        "start_within_segment",
    }
)
CONDITIONAL_ARRAY_NAMES = frozenset(
    {"conditional_mutual_information", "mean_conditional_mutual_information"}
)


def _strict_json(path: Path, *, label: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} contains duplicate field {key!r}")
            output[key] = value
        return output

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error


def _finite(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be numeric")
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if not np.isfinite(output) or (minimum is not None and output < minimum):
        suffix = "" if minimum is None else f" and at least {minimum}"
        raise ValueError(f"{label} must be finite{suffix}")
    return output


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _roi_group(value: object, *, label: str, optional: bool = False) -> tuple[str, ...] | None:
    if optional and value is None:
        return None
    if isinstance(value, str) or not isinstance(value, list) or not value:
        suffix = " or null" if optional else ""
        raise ValueError(f"{label} must be a non-empty JSON array{suffix}")
    names = tuple(_identifier(name, label=f"{label} ROI") for name in value)
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate ROI names")
    return names


@dataclass(frozen=True)
class InformationGroups:
    """Named left, right, and optional conditioning ROI groups."""

    left: tuple[str, ...]
    right: tuple[str, ...]
    conditioning: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if isinstance(self.left, str) or not isinstance(self.left, Sequence):
            raise TypeError("left must be a sequence of ROI names")
        if isinstance(self.right, str) or not isinstance(self.right, Sequence):
            raise TypeError("right must be a sequence of ROI names")
        if self.conditioning is not None and (
            isinstance(self.conditioning, str) or not isinstance(self.conditioning, Sequence)
        ):
            raise TypeError("conditioning must be a sequence of ROI names or null")
        left = _roi_group(list(self.left), label="left")
        right = _roi_group(list(self.right), label="right")
        conditioning = _roi_group(
            None if self.conditioning is None else list(self.conditioning),
            label="conditioning",
            optional=True,
        )
        assert left is not None and right is not None
        groups = {"left": left, "right": right, "conditioning": conditioning or ()}
        seen: dict[str, str] = {}
        for group, names in groups.items():
            for name in names:
                if name in seen:
                    raise ValueError(
                        f"ROI groups must be disjoint; {name!r} occurs in {seen[name]} and {group}"
                    )
                seen[name] = group
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "conditioning", conditioning)

    def indices(self, roi_names: Sequence[str]) -> tuple[NDArray[np.int64], ...]:
        """Resolve all group names against one exact ROI axis."""
        axis = tuple(str(name) for name in roi_names)
        if len(set(axis)) != len(axis):
            raise ValueError("ROI axis contains duplicate names")
        positions = {name: index for index, name in enumerate(axis)}
        missing = sorted(
            set(self.left)
            .union(self.right)
            .union(() if self.conditioning is None else self.conditioning)
            .difference(positions)
        )
        if missing:
            raise ValueError(f"information-group ROIs are absent from the loaded XCP-D axis: {missing}")
        left = np.asarray([positions[name] for name in self.left], dtype=np.int64)
        right = np.asarray([positions[name] for name in self.right], dtype=np.int64)
        condition = (
            None
            if self.conditioning is None
            else np.asarray([positions[name] for name in self.conditioning], dtype=np.int64)
        )
        return _readonly(left), _readonly(right), None if condition is None else _readonly(condition)

    def payload(self) -> dict[str, object]:
        return {
            "conditioning": None if self.conditioning is None else list(self.conditioning),
            "left": list(self.left),
            "right": list(self.right),
        }


def load_information_groups(path: str | Path) -> InformationGroups:
    """Load the strict standalone ROI-group JSON schema."""
    source = Path(path)
    raw = _strict_json(source, label="information-group JSON")
    if not isinstance(raw, dict) or set(raw) != set(GROUP_FIELDS):
        raise ValueError(
            "information-group JSON must contain exactly conditioning, left, and right"
        )
    return InformationGroups(
        left=_roi_group(raw["left"], label="left"),  # type: ignore[arg-type]
        right=_roi_group(raw["right"], label="right"),  # type: ignore[arg-type]
        conditioning=_roi_group(raw["conditioning"], label="conditioning", optional=True),
    )


@dataclass(frozen=True)
class FrozenWindow:
    """One replayable fixed window in original-frame coordinates."""

    acquisition_id: str
    length: int
    draw: int
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "acquisition_id", _identifier(self.acquisition_id, label="acquisition_id"))
        object.__setattr__(self, "length", _integer(self.length, label="length", minimum=1))
        object.__setattr__(self, "draw", _integer(self.draw, label="draw"))
        object.__setattr__(self, "start_frame", _integer(self.start_frame, label="start_frame"))
        object.__setattr__(self, "end_frame", _integer(self.end_frame, label="end_frame"))
        if self.end_frame - self.start_frame + 1 != self.length:
            raise ValueError("frozen window end_frame - start_frame + 1 must equal length")


def _read_tsv(path: Path, *, fields: tuple[str, ...], label: str) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames != list(fields):
                raise ValueError(f"{label} header must be exactly {list(fields)}")
            rows = list(reader)
    except csv.Error as error:
        raise ValueError(f"cannot parse {label}: {error}") from error
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{label} contains a row with the wrong number of columns")
    return rows


def _parse_tsv_integer(value: str, *, label: str, minimum: int = 0) -> int:
    if not value or value.strip() != value:
        raise ValueError(f"{label} must be a canonical integer")
    try:
        output = int(value, 10)
    except ValueError as error:
        raise ValueError(f"{label} must be a canonical integer") from error
    if str(output) != value or output < minimum:
        raise ValueError(f"{label} must be a canonical integer at least {minimum}")
    return output


def load_fixed_window_schedule(path: str | Path) -> tuple[FrozenWindow, ...]:
    """Read a strict frozen-window TSV in original-frame coordinates."""
    source = Path(path)
    rows = _read_tsv(source, fields=SCHEDULE_FIELDS, label="fixed-window schedule")
    if not rows:
        raise ValueError("fixed-window schedule must contain at least one row")
    windows = tuple(
        FrozenWindow(
            acquisition_id=_identifier(row["acquisition_id"], label="acquisition_id"),
            length=_parse_tsv_integer(row["length"], label="length", minimum=1),
            draw=_parse_tsv_integer(row["draw"], label="draw"),
            start_frame=_parse_tsv_integer(row["start_frame"], label="start_frame"),
            end_frame=_parse_tsv_integer(row["end_frame"], label="end_frame"),
        )
        for row in rows
    )
    keys = [(window.acquisition_id, window.length, window.draw) for window in windows]
    if len(set(keys)) != len(keys):
        raise ValueError("fixed-window schedule contains duplicate acquisition/length/draw rows")
    return windows


@dataclass(frozen=True)
class InformationAcquisition:
    """Identity and retained-axis summary for one analyzed acquisition."""

    subject: str
    session: str | None
    acquisition_id: str
    n_retained_frames: int
    n_segments: int


@dataclass(frozen=True)
class InformationCell:
    """One acquisition-by-length unit with at least one eligible start."""

    acquisition_index: int
    length: int
    eligible_starts: int


@dataclass(frozen=True)
class FixedInformationArtifact:
    """Complete fixed-length information estimates and their replay metadata."""

    groups: InformationGroups
    roi_names: tuple[str, ...]
    left_indices: NDArray[np.int64]
    right_indices: NDArray[np.int64]
    conditioning_indices: NDArray[np.int64] | None
    acquisitions: tuple[InformationAcquisition, ...]
    cells: tuple[InformationCell, ...]
    lengths: tuple[int, ...]
    draws_per_length: int
    sample_seed: int
    schedule_mode: str
    schedule_source: str | None
    k: int
    metric: str
    jitter: float
    jitter_seed: int
    standardized: bool
    implementation: str
    acquisition_index: NDArray[np.int64]
    length: NDArray[np.int64]
    draw: NDArray[np.int64]
    segment_id: NDArray[np.int64]
    start_within_segment: NDArray[np.int64]
    start_frame: NDArray[np.int64]
    end_frame: NDArray[np.int64]
    eligible_starts: NDArray[np.int64]
    mutual_information: NDArray[np.float64]
    conditional_mutual_information: NDArray[np.float64] | None
    mean_mutual_information: NDArray[np.float64]
    mean_conditional_mutual_information: NDArray[np.float64] | None
    format_version: int = FORMAT_VERSION

    @property
    def n_draws(self) -> int:
        return len(self.draw)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def has_cmi(self) -> bool:
        return self.conditioning_indices is not None


def _validated_lengths(lengths: Sequence[int], *, k: int) -> tuple[int, ...]:
    output = tuple(_integer(length, label="length", minimum=1) for length in lengths)
    if not output or len(set(output)) != len(output):
        raise ValueError("lengths must contain unique values in the requested order")
    if any(length <= k + 2 for length in output):
        raise ValueError("every length must be greater than k + 2")
    return output


def _validated_dataset(dataset: TimeSeriesDataset) -> tuple[TimeSeriesRun, ...]:
    if not isinstance(dataset, TimeSeriesDataset):
        raise TypeError("dataset must be a TimeSeriesDataset")
    dataset.require_subject_ids("fixed-information analysis")
    runs = tuple(dataset.runs)
    if any(run.acquisition_id is None for run in runs):
        raise ValueError("fixed-information analysis requires acquisition IDs for every run")
    identifiers = tuple(str(run.acquisition_id) for run in runs)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("fixed-information acquisition IDs must be globally unique")
    return runs


def _frozen_samples(
    run: TimeSeriesRun,
    windows: Sequence[FrozenWindow],
    *,
    eligible_starts: int,
    sample_seed: int,
) -> FixedWindowSamples:
    values = []
    originals = []
    segment_ids = []
    starts_within = []
    segments = run.segments()
    for window in windows:
        start = int(np.searchsorted(run.original_indices, window.start_frame))
        stop = start + window.length
        original = run.original_indices[start:stop]
        if (
            len(original) != window.length
            or int(original[0]) != window.start_frame
            or int(original[-1]) != window.end_frame
            or np.any(np.diff(original) != 1)
        ):
            raise ValueError(
                "frozen window is not one contiguous retained interval: "
                f"{window.acquisition_id}, length={window.length}, draw={window.draw}"
            )
        matched = False
        for segment_id, positions in enumerate(segments):
            if start >= int(positions[0]) and stop <= int(positions[-1]) + 1:
                values.append(run.values[start:stop])
                originals.append(original)
                segment_ids.append(segment_id)
                starts_within.append(start - int(positions[0]))
                matched = True
                break
        if not matched:
            raise ValueError("frozen window does not lie within one retained segment")

    first = windows[0]
    return FixedWindowSamples(
        values=_readonly(np.stack(values)),
        original_indices=_readonly(np.stack(originals).astype(np.int64, copy=False)),
        draw_indices=_readonly(np.asarray([window.draw for window in windows], dtype=np.int64)),
        segment_ids=_readonly(np.asarray(segment_ids, dtype=np.int64)),
        starts_within_segment=_readonly(np.asarray(starts_within, dtype=np.int64)),
        start_frames=_readonly(
            np.asarray([window.start_frame for window in windows], dtype=np.int64)
        ),
        end_frames=_readonly(
            np.asarray([window.end_frame for window in windows], dtype=np.int64)
        ),
        eligible_starts=eligible_starts,
        length=first.length,
        requested_draws=len(windows),
        seed=sample_seed,
        subject=run.subject,
        session=run.session,
        acquisition_id=run.acquisition_id,
    )


def compute_fixed_information(
    dataset: TimeSeriesDataset,
    groups: InformationGroups,
    *,
    lengths: Sequence[int],
    draws: int,
    sample_seed: int,
    schedule: Sequence[FrozenWindow] | None = None,
    schedule_source: str | None = None,
    k: int = 3,
    jitter: float = 1e-10,
    jitter_seed: int = 20260811,
    standardize: bool = True,
    jobs: int = 1,
) -> FixedInformationArtifact:
    """Compute fixed-length block MI/CMI across a complete acquisition grid.

    ``jobs > 1`` shares one process pool across all acquisition/length/draw
    combinations. Windows are submitted in a bounded stream and results retain
    their serial order. Python scripts using parallel execution must call this
    function inside an ``if __name__ == "__main__":`` guard.
    """
    runs = _validated_dataset(dataset)
    if not isinstance(groups, InformationGroups):
        raise TypeError("groups must be an InformationGroups object")
    neighbors = _integer(k, label="k", minimum=1)
    selected_lengths = _validated_lengths(lengths, k=neighbors)
    n_draws = _integer(draws, label="draws", minimum=1)
    sampling_seed = _integer(sample_seed, label="sample_seed")
    tie_seed = _integer(jitter_seed, label="jitter_seed")
    tie_jitter = _finite(jitter, label="jitter", minimum=0.0)
    worker_count = _integer(jobs, label="jobs", minimum=1)
    left, right, condition = groups.indices(dataset.roi_names)
    schedule_mode = "sampled" if schedule is None else "frozen"
    if schedule is None and schedule_source is not None:
        raise ValueError("schedule_source requires a frozen schedule")
    if schedule_source is not None:
        schedule_source = _identifier(schedule_source, label="schedule_source")
    if schedule is not None and not schedule:
        raise ValueError("frozen schedule cannot be empty")
    if schedule is not None and any(
        not isinstance(window, FrozenWindow) for window in schedule
    ):
        raise TypeError("schedule must contain only FrozenWindow objects")
    scheduled = {
        (window.acquisition_id, window.length, window.draw): window
        for window in (() if schedule is None else schedule)
    }
    if schedule is not None and len(scheduled) != len(schedule):
        raise ValueError("frozen schedule contains duplicate acquisition/length/draw rows")

    run_ids = {str(run.acquisition_id) for run in runs}
    requested_cells = {
        (acquisition_id, length)
        for acquisition_id in run_ids
        for length in selected_lengths
    }
    schedule_cells = {
        (acquisition_id, length) for acquisition_id, length, _ in scheduled
    }
    unknown_cells = sorted(schedule_cells.difference(requested_cells))
    if unknown_cells:
        raise ValueError(
            f"frozen schedule contains unknown acquisition/length cells: {unknown_cells[:5]}"
        )
    for acquisition_id, length in schedule_cells:
        observed_draws = {
            draw
            for candidate_acquisition, candidate_length, draw in scheduled
            if candidate_acquisition == acquisition_id and candidate_length == length
        }
        if observed_draws != set(range(n_draws)):
            raise ValueError(
                f"frozen schedule cell {(acquisition_id, length)} must contain "
                f"draws 0..{n_draws - 1}"
            )

    if schedule is None:
        ineligible_acquisitions = [
            str(run.acquisition_id)
            for run in runs
            if not any(_eligible_fixed_window_count(run, length) for length in selected_lengths)
        ]
        if ineligible_acquisitions:
            raise ValueError(
                "every acquisition must contain at least one analyzable length cell; "
                f"ineligible acquisitions: {ineligible_acquisitions}"
            )

    acquisitions = tuple(
        InformationAcquisition(
            subject=str(run.subject),
            session=None if run.session is None else str(run.session),
            acquisition_id=str(run.acquisition_id),
            n_retained_frames=run.n_frames,
            n_segments=len(run.segments()),
        )
        for run in runs
    )
    scalar: dict[str, list[int]] = {
        "acquisition_index": [],
        "length": [],
        "draw": [],
        "segment_id": [],
        "start_within_segment": [],
        "start_frame": [],
        "end_frame": [],
        "eligible_starts": [],
    }
    mutual_information: list[NDArray[np.float64]] = []
    conditional_information: list[NDArray[np.float64]] = []
    mean_mutual_information: list[float] = []
    mean_conditional_information: list[float] = []
    cells: list[InformationCell] = []
    def windows() -> Iterator[NDArray[np.float64]]:
        for acquisition_index, run in enumerate(runs):
            acquisition_id = str(run.acquisition_id)
            for length in selected_lengths:
                eligible = _eligible_fixed_window_count(run, length)
                schedule_cell = (acquisition_id, length)
                if eligible == 0:
                    if schedule_cell in schedule_cells:
                        raise ValueError(
                            f"{run.acquisition_id}: frozen schedule contains an ineligible "
                            f"length {length}"
                        )
                    continue
                if schedule is not None and schedule_cell not in schedule_cells:
                    continue

                cells.append(InformationCell(acquisition_index, length, eligible))
                if schedule is None:
                    samples = sample_fixed_windows(
                        run,
                        length,
                        n_draws,
                        seed=sampling_seed,
                    )
                else:
                    samples = _frozen_samples(
                        run,
                        tuple(
                            scheduled[(acquisition_id, length, draw)]
                            for draw in range(n_draws)
                        ),
                        eligible_starts=eligible,
                        sample_seed=sampling_seed,
                    )
                for draw_index, window in enumerate(samples.values):
                    scalar["acquisition_index"].append(acquisition_index)
                    scalar["length"].append(length)
                    scalar["draw"].append(int(samples.draw_indices[draw_index]))
                    scalar["segment_id"].append(int(samples.segment_ids[draw_index]))
                    scalar["start_within_segment"].append(
                        int(samples.starts_within_segment[draw_index])
                    )
                    scalar["start_frame"].append(int(samples.start_frames[draw_index]))
                    scalar["end_frame"].append(int(samples.end_frames[draw_index]))
                    scalar["eligible_starts"].append(eligible)
                    yield window

    for result in _estimate_information_blocks(
        windows(),
        left,
        right,
        conditioning=condition,
        standardize=standardize,
        k=neighbors,
        jitter=tie_jitter,
        jitter_seed=tie_seed,
        jobs=worker_count,
    ):
        mutual_information.append(result.mutual_information)
        mean_mutual_information.append(result.mean_mutual_information)
        if result.conditional_mutual_information is not None:
            conditional_information.append(result.conditional_mutual_information)
            assert result.mean_conditional_mutual_information is not None
            mean_conditional_information.append(result.mean_conditional_mutual_information)

    if not cells:
        raise ValueError("fixed-information analysis produced no analyzable cells")

    scipy, _, _ = _information_dependencies()
    integers = {
        name: _readonly(np.asarray(values, dtype=np.int64))
        for name, values in scalar.items()
    }
    return FixedInformationArtifact(
        groups=groups,
        roi_names=dataset.roi_names,
        left_indices=left,
        right_indices=right,
        conditioning_indices=condition,
        acquisitions=acquisitions,
        cells=tuple(cells),
        lengths=selected_lengths,
        draws_per_length=n_draws,
        sample_seed=sampling_seed,
        schedule_mode=schedule_mode,
        schedule_source=schedule_source,
        k=neighbors,
        metric="chebyshev",
        jitter=tie_jitter,
        jitter_seed=tie_seed,
        standardized=bool(standardize),
        implementation=(
            f"scipy {scipy.__version__} cKDTree; Kraskov MI and Frenzel-Pompe CMI"
        ),
        acquisition_index=integers["acquisition_index"],
        length=integers["length"],
        draw=integers["draw"],
        segment_id=integers["segment_id"],
        start_within_segment=integers["start_within_segment"],
        start_frame=integers["start_frame"],
        end_frame=integers["end_frame"],
        eligible_starts=integers["eligible_starts"],
        mutual_information=_readonly(
            np.stack(mutual_information).astype(np.float64, copy=False)
        ),
        conditional_mutual_information=(
            None
            if condition is None
            else _readonly(
                np.stack(conditional_information).astype(np.float64, copy=False)
            )
        ),
        mean_mutual_information=_readonly(
            np.asarray(mean_mutual_information, dtype=np.float64)
        ),
        mean_conditional_mutual_information=(
            None
            if condition is None
            else _readonly(
                np.asarray(mean_conditional_information, dtype=np.float64)
            )
        ),
    )


def save_fixed_information(
    artifact: FixedInformationArtifact,
    path: str | Path,
) -> Path:
    """Atomically write a new portable fixed-information artifact directory."""
    from ._artifact import save_fixed_information as implementation

    return implementation(artifact, path)


def load_fixed_information(path: str | Path) -> FixedInformationArtifact:
    """Load and cross-validate a fixed-information artifact."""
    from ._artifact import load_fixed_information as implementation

    return implementation(path)


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "GROUP_FIELDS",
    "SCHEDULE_FIELDS",
    "FixedInformationArtifact",
    "FrozenWindow",
    "InformationAcquisition",
    "InformationCell",
    "InformationGroups",
    "compute_fixed_information",
    "load_fixed_information",
    "load_fixed_window_schedule",
    "load_information_groups",
    "save_fixed_information",
]
