"""Portable fixed-length MI/CMI analysis artifacts."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..connectivity.information import FixedLengthInformation, block_information
from ..data import TimeSeriesDataset, TimeSeriesRun

FORMAT_NAME = "dfckit-fixed-information"
FORMAT_VERSION = 1
GROUP_FIELDS = ("conditioning", "left", "right")
SCHEDULE_FIELDS = ("acquisition_id", "length", "draw", "start_frame", "end_frame")
DRAW_FIELDS = (
    "array_row",
    "subject",
    "session",
    "acquisition_id",
    "length",
    "draw",
    "segment_id",
    "start_within_segment",
    "start_frame",
    "end_frame",
    "eligible_starts",
    "mean_mi",
    "mean_cmi",
)
SESSION_FIELDS = (
    "subject",
    "session",
    "acquisition_id",
    "length",
    "draws",
    "eligible_starts",
    "mean_mi",
    "sd_mi",
    "mean_cmi",
    "sd_cmi",
)
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


def _readonly(values: NDArray) -> NDArray:
    output = np.asarray(values).copy()
    output.setflags(write=False)
    return output


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


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return int(value)


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


def _scheduled_window(
    run: TimeSeriesRun,
    window: FrozenWindow,
) -> tuple[NDArray[np.float64], int, int]:
    start_position = int(np.searchsorted(run.original_indices, window.start_frame))
    stop_position = start_position + window.length
    original = run.original_indices[start_position:stop_position]
    if (
        len(original) != window.length
        or int(original[0]) != window.start_frame
        or int(original[-1]) != window.end_frame
        or np.any(np.diff(original) != 1)
    ):
        raise ValueError(
            f"frozen window is not one contiguous retained interval: "
            f"{window.acquisition_id}, length={window.length}, draw={window.draw}"
        )
    for segment_id, positions in enumerate(run.segments()):
        if start_position >= int(positions[0]) and stop_position <= int(positions[-1]) + 1:
            start_within = start_position - int(positions[0])
            return run.values[start_position:stop_position], segment_id, start_within
    raise ValueError("frozen window does not lie within one retained segment")


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
    """Compute fixed-length block MI/CMI across a complete acquisition grid."""
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
    if schedule is not None and any(not isinstance(window, FrozenWindow) for window in schedule):
        raise TypeError("schedule must contain only FrozenWindow objects")
    scheduled = {
        (window.acquisition_id, window.length, window.draw): window
        for window in (() if schedule is None else schedule)
    }
    if schedule is not None and len(scheduled) != len(schedule):
        raise ValueError("frozen schedule contains duplicate acquisition/length/draw rows")
    run_ids = {str(run.acquisition_id) for run in runs}
    requested_keys = {(acquisition_id, length) for acquisition_id in run_ids for length in selected_lengths}
    schedule_cells: set[tuple[str, int]] = set()
    if schedule is not None:
        schedule_cells = {(acquisition_id, length) for acquisition_id, length, _ in scheduled}
        unknown_cells = sorted(schedule_cells.difference(requested_keys))
        if unknown_cells:
            raise ValueError(f"frozen schedule contains unknown acquisition/length cells: {unknown_cells[:5]}")
        for acquisition_id, length in schedule_cells:
            draws_in_cell = {
                draw for candidate_acquisition, candidate_length, draw in scheduled
                if candidate_acquisition == acquisition_id and candidate_length == length
            }
            expected_draws = set(range(n_draws))
            if draws_in_cell != expected_draws:
                raise ValueError(
                    f"frozen schedule cell {(acquisition_id, length)} must contain draws 0..{n_draws - 1}"
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
    mi_values: list[NDArray[np.float64]] = []
    cmi_values: list[NDArray[np.float64]] = []
    mi_means: list[float] = []
    cmi_means: list[float] = []
    implementation: str | None = None
    cells: list[InformationCell] = []

    for acquisition_index, run in enumerate(runs):
        for length in selected_lengths:
            eligible = int(
                sum(max(0, len(segment) - length + 1) for segment in run.segments())
            )
            schedule_cell = (str(run.acquisition_id), length)
            if eligible == 0:
                if schedule is not None and schedule_cell in schedule_cells:
                    raise ValueError(
                        f"{run.acquisition_id}: frozen schedule contains an ineligible length {length}"
                    )
                continue
            if schedule is not None and schedule_cell not in schedule_cells:
                continue
            cells.append(InformationCell(acquisition_index, length, eligible))
            if schedule is None:
                result = FixedLengthInformation(
                    length=length,
                    draws=n_draws,
                    sample_seed=sampling_seed,
                    k=neighbors,
                    jitter=tie_jitter,
                    jitter_seed=tie_seed,
                    standardize=standardize,
                    jobs=worker_count,
                ).transform(run, left, right, conditioning=condition)
                implementation = result.implementation
                windows = tuple(result.samples.values)
                segment_ids = result.samples.segment_ids
                starts_within = result.samples.starts_within_segment
                start_frames = result.samples.start_frames
                end_frames = result.samples.end_frames
                estimates = None
            else:
                scheduled_windows = tuple(
                    _scheduled_window(
                        run,
                        scheduled[(str(run.acquisition_id), length, draw)],
                    )
                    for draw in range(n_draws)
                )
                windows = tuple(item[0] for item in scheduled_windows)
                segment_ids = np.asarray([item[1] for item in scheduled_windows], dtype=np.int64)
                starts_within = np.asarray([item[2] for item in scheduled_windows], dtype=np.int64)
                start_frames = np.asarray(
                    [scheduled[(str(run.acquisition_id), length, draw)].start_frame for draw in range(n_draws)],
                    dtype=np.int64,
                )
                end_frames = np.asarray(
                    [scheduled[(str(run.acquisition_id), length, draw)].end_frame for draw in range(n_draws)],
                    dtype=np.int64,
                )

                def estimate(values: NDArray[np.float64]):
                    return block_information(
                        values,
                        left,
                        right,
                        conditioning=condition,
                        standardize=standardize,
                        k=neighbors,
                        jitter=tie_jitter,
                        seed=tie_seed,
                    )

                if worker_count == 1:
                    estimates = tuple(estimate(values) for values in windows)
                else:
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        estimates = tuple(executor.map(estimate, windows))
            for draw in range(n_draws):
                if estimates is None:
                    mi = result.mutual_information[draw]
                    cmi = (
                        None
                        if result.conditional_mutual_information is None
                        else result.conditional_mutual_information[draw]
                    )
                    mean_mi = float(result.mean_mutual_information[draw])
                    mean_cmi = (
                        None
                        if result.mean_conditional_mutual_information is None
                        else float(result.mean_conditional_mutual_information[draw])
                    )
                else:
                    estimate = estimates[draw]
                    mi = estimate.mutual_information
                    cmi = estimate.conditional_mutual_information
                    mean_mi = estimate.mean_mutual_information
                    mean_cmi = estimate.mean_conditional_mutual_information
                scalar["acquisition_index"].append(acquisition_index)
                scalar["length"].append(length)
                scalar["draw"].append(draw)
                scalar["segment_id"].append(int(segment_ids[draw]))
                scalar["start_within_segment"].append(int(starts_within[draw]))
                scalar["start_frame"].append(int(start_frames[draw]))
                scalar["end_frame"].append(int(end_frames[draw]))
                scalar["eligible_starts"].append(eligible)
                mi_values.append(np.asarray(mi, dtype=float))
                mi_means.append(float(mean_mi))
                if cmi is not None and mean_cmi is not None:
                    cmi_values.append(np.asarray(cmi, dtype=float))
                    cmi_means.append(float(mean_cmi))

    if schedule is None and any(
        not any(cell.acquisition_index == acquisition_index for cell in cells)
        for acquisition_index in range(len(runs))
    ):
        raise ValueError("every acquisition must contain at least one analyzable length cell")

    if implementation is None:
        import scipy

        implementation = f"scipy {scipy.__version__} cKDTree; Kraskov MI and Frenzel-Pompe CMI"
    integers = {
        name: _readonly(np.asarray(values, dtype=np.int64)) for name, values in scalar.items()
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
        implementation=implementation,
        acquisition_index=integers["acquisition_index"],
        length=integers["length"],
        draw=integers["draw"],
        segment_id=integers["segment_id"],
        start_within_segment=integers["start_within_segment"],
        start_frame=integers["start_frame"],
        end_frame=integers["end_frame"],
        eligible_starts=integers["eligible_starts"],
        mutual_information=_readonly(np.stack(mi_values).astype(np.float64, copy=False)),
        conditional_mutual_information=(
            None
            if condition is None
            else _readonly(np.stack(cmi_values).astype(np.float64, copy=False))
        ),
        mean_mutual_information=_readonly(np.asarray(mi_means, dtype=np.float64)),
        mean_conditional_mutual_information=(
            None if condition is None else _readonly(np.asarray(cmi_means, dtype=np.float64))
        ),
    )


def _arrays(artifact: FixedInformationArtifact) -> dict[str, NDArray]:
    output = {
        "acquisition_index": artifact.acquisition_index,
        "draw": artifact.draw,
        "eligible_starts": artifact.eligible_starts,
        "end_frame": artifact.end_frame,
        "length": artifact.length,
        "mean_mutual_information": artifact.mean_mutual_information,
        "mutual_information": artifact.mutual_information,
        "segment_id": artifact.segment_id,
        "start_frame": artifact.start_frame,
        "start_within_segment": artifact.start_within_segment,
    }
    if artifact.conditional_mutual_information is not None:
        output["conditional_mutual_information"] = artifact.conditional_mutual_information
        assert artifact.mean_conditional_mutual_information is not None
        output["mean_conditional_mutual_information"] = artifact.mean_conditional_mutual_information
    return output


def _manifest(artifact: FixedInformationArtifact, arrays: Mapping[str, NDArray]) -> dict[str, object]:
    return {
        "acquisitions": [
            {
                "acquisition_id": item.acquisition_id,
                "index": index,
                "n_retained_frames": item.n_retained_frames,
                "n_segments": item.n_segments,
                "session": item.session,
                "subject": item.subject,
            }
            for index, item in enumerate(artifact.acquisitions)
        ],
        "array_names": sorted(arrays),
        "cells": [
            {
                "acquisition_index": cell.acquisition_index,
                "eligible_starts": cell.eligible_starts,
                "length": cell.length,
            }
            for cell in artifact.cells
        ],
        "draws_per_length": artifact.draws_per_length,
        "estimator": {
            "implementation": artifact.implementation,
            "jitter": artifact.jitter,
            "jitter_seed": artifact.jitter_seed,
            "k": artifact.k,
            "metric": artifact.metric,
            "standardized": artifact.standardized,
        },
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "groups": artifact.groups.payload(),
        "group_indices": {
            "conditioning": (
                None
                if artifact.conditioning_indices is None
                else artifact.conditioning_indices.tolist()
            ),
            "left": artifact.left_indices.tolist(),
            "right": artifact.right_indices.tolist(),
        },
        "lengths": list(artifact.lengths),
        "n_cells": artifact.n_cells,
        "n_draws": artifact.n_draws,
        "roi_names": list(artifact.roi_names),
        "sampling": {
            "sample_seed": artifact.sample_seed,
            "schedule_mode": artifact.schedule_mode,
            "schedule_source": artifact.schedule_source,
        },
    }


def _float_text(value: float | None) -> str:
    return "" if value is None else format(float(value), ".17g")


def _draw_rows(artifact: FixedInformationArtifact) -> list[dict[str, object]]:
    output = []
    for row in range(artifact.n_draws):
        acquisition = artifact.acquisitions[int(artifact.acquisition_index[row])]
        output.append(
            {
                "array_row": row,
                "subject": acquisition.subject,
                "session": "" if acquisition.session is None else acquisition.session,
                "acquisition_id": acquisition.acquisition_id,
                "length": int(artifact.length[row]),
                "draw": int(artifact.draw[row]),
                "segment_id": int(artifact.segment_id[row]),
                "start_within_segment": int(artifact.start_within_segment[row]),
                "start_frame": int(artifact.start_frame[row]),
                "end_frame": int(artifact.end_frame[row]),
                "eligible_starts": int(artifact.eligible_starts[row]),
                "mean_mi": _float_text(float(artifact.mean_mutual_information[row])),
                "mean_cmi": _float_text(
                    None
                    if artifact.mean_conditional_mutual_information is None
                    else float(artifact.mean_conditional_mutual_information[row])
                ),
            }
        )
    return output


def _session_rows(artifact: FixedInformationArtifact) -> list[dict[str, object]]:
    output = []
    for cell in artifact.cells:
        acquisition = artifact.acquisitions[cell.acquisition_index]
        selected = np.flatnonzero(
            (artifact.acquisition_index == cell.acquisition_index)
            & (artifact.length == cell.length)
        )
        mi = artifact.mean_mutual_information[selected]
        cmi = (
            None
            if artifact.mean_conditional_mutual_information is None
            else artifact.mean_conditional_mutual_information[selected]
        )
        eligible = np.unique(artifact.eligible_starts[selected])
        if (
            len(selected) != artifact.draws_per_length
            or len(eligible) != 1
            or int(eligible[0]) != cell.eligible_starts
        ):
            raise ValueError("artifact does not contain one complete declared information cell")
        output.append(
            {
                "subject": acquisition.subject,
                "session": "" if acquisition.session is None else acquisition.session,
                "acquisition_id": acquisition.acquisition_id,
                "length": cell.length,
                "draws": artifact.draws_per_length,
                "eligible_starts": cell.eligible_starts,
                "mean_mi": _float_text(float(mi.mean())),
                "sd_mi": _float_text(float(mi.std(ddof=0))),
                "mean_cmi": _float_text(None if cmi is None else float(cmi.mean())),
                "sd_cmi": _float_text(None if cmi is None else float(cmi.std(ddof=0))),
            }
        )
    return output


def _write_tsv(path: Path, fields: tuple[str, ...], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_fixed_information(
    artifact: FixedInformationArtifact,
    path: str | Path,
) -> Path:
    """Atomically write a new portable fixed-information artifact directory."""
    if not isinstance(artifact, FixedInformationArtifact):
        raise TypeError("artifact must be a FixedInformationArtifact")
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"fixed-information artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = _arrays(artifact)
    manifest = _manifest(artifact, arrays)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        np.savez(temporary / "arrays.npz", **arrays)
        _write_tsv(temporary / "draw_metrics.tsv", DRAW_FIELDS, _draw_rows(artifact))
        _write_tsv(temporary / "session_metrics.tsv", SESSION_FIELDS, _session_rows(artifact))
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_fixed_information(temporary)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"fixed-information artifact already exists: {target}")
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _manifest_fields() -> set[str]:
    return {
        "acquisitions",
        "array_names",
        "cells",
        "draws_per_length",
        "estimator",
        "format",
        "format_version",
        "group_indices",
        "groups",
        "lengths",
        "n_cells",
        "n_draws",
        "roi_names",
        "sampling",
    }


def _validate_indices(
    values: object,
    *,
    label: str,
    roi_names: tuple[str, ...],
    expected_names: tuple[str, ...] | None,
) -> NDArray[np.int64] | None:
    if expected_names is None:
        if values is not None:
            raise ValueError(f"{label} indices must be null")
        return None
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ValueError(f"{label} indices must be an integer array")
    indices = np.asarray(values, dtype=np.int64)
    if len(indices) != len(expected_names) or np.any(indices < 0) or np.any(indices >= len(roi_names)):
        raise ValueError(f"{label} indices have an invalid shape or range")
    if tuple(roi_names[index] for index in indices) != expected_names:
        raise ValueError(f"{label} indices do not resolve to the declared ROI names")
    return _readonly(indices)


def load_fixed_information(path: str | Path) -> FixedInformationArtifact:
    """Load and cross-validate all arrays, manifest fields, and TSV summaries."""
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"fixed-information artifact directory does not exist: {root}")
    expected_files = {"arrays.npz", "draw_metrics.tsv", "manifest.json", "session_metrics.tsv"}
    if {entry.name for entry in root.iterdir()} != expected_files:
        raise ValueError("fixed-information artifact must contain exactly its four specified files")
    for name in expected_files:
        if not (root / name).is_file() or (root / name).is_symlink():
            raise ValueError(f"fixed-information artifact member is not a regular file: {name}")

    manifest = _strict_json(root / "manifest.json", label="fixed-information manifest")
    if not isinstance(manifest, dict) or set(manifest) != _manifest_fields():
        raise ValueError("fixed-information manifest fields do not match the schema")
    if manifest["format"] != FORMAT_NAME or _integer(
        manifest["format_version"], label="format_version", minimum=1
    ) != FORMAT_VERSION:
        raise ValueError("unsupported fixed-information artifact format or version")
    groups_raw = manifest["groups"]
    if not isinstance(groups_raw, dict) or set(groups_raw) != set(GROUP_FIELDS):
        raise ValueError("fixed-information groups are invalid")
    groups = InformationGroups(
        left=_roi_group(groups_raw["left"], label="left"),  # type: ignore[arg-type]
        right=_roi_group(groups_raw["right"], label="right"),  # type: ignore[arg-type]
        conditioning=_roi_group(
            groups_raw["conditioning"], label="conditioning", optional=True
        ),
    )
    roi_raw = manifest["roi_names"]
    if not isinstance(roi_raw, list):
        raise TypeError("roi_names must be an array")
    roi_names = tuple(_identifier(name, label="ROI name") for name in roi_raw)
    if not roi_names or len(set(roi_names)) != len(roi_names):
        raise ValueError("roi_names must be non-empty and unique")
    group_indices = manifest["group_indices"]
    if not isinstance(group_indices, dict) or set(group_indices) != set(GROUP_FIELDS):
        raise ValueError("group_indices fields are invalid")
    left = _validate_indices(
        group_indices["left"], label="left", roi_names=roi_names, expected_names=groups.left
    )
    right = _validate_indices(
        group_indices["right"], label="right", roi_names=roi_names, expected_names=groups.right
    )
    condition = _validate_indices(
        group_indices["conditioning"],
        label="conditioning",
        roi_names=roi_names,
        expected_names=groups.conditioning,
    )
    assert left is not None and right is not None

    estimator = manifest["estimator"]
    if not isinstance(estimator, dict) or set(estimator) != {
        "implementation",
        "jitter",
        "jitter_seed",
        "k",
        "metric",
        "standardized",
    }:
        raise ValueError("fixed-information estimator metadata is invalid")
    implementation = _identifier(estimator["implementation"], label="implementation")
    k = _integer(estimator["k"], label="k", minimum=1)
    if estimator["metric"] != "chebyshev":
        raise ValueError("fixed-information metric must be chebyshev")
    if not isinstance(estimator["standardized"], bool):
        raise TypeError("standardized must be boolean")
    jitter = _finite(estimator["jitter"], label="jitter", minimum=0.0)
    jitter_seed = _integer(estimator["jitter_seed"], label="jitter_seed")
    lengths_raw = manifest["lengths"]
    if not isinstance(lengths_raw, list):
        raise TypeError("lengths must be an array")
    lengths = _validated_lengths(lengths_raw, k=k)
    draws = _integer(manifest["draws_per_length"], label="draws_per_length", minimum=1)
    sampling = manifest["sampling"]
    if not isinstance(sampling, dict) or set(sampling) != {
        "sample_seed",
        "schedule_mode",
        "schedule_source",
    }:
        raise ValueError("fixed-information sampling metadata is invalid")
    sample_seed = _integer(sampling["sample_seed"], label="sample_seed")
    if sampling["schedule_mode"] not in {"sampled", "frozen"}:
        raise ValueError("schedule_mode must be sampled or frozen")
    schedule_source = sampling["schedule_source"]
    if schedule_source is not None:
        schedule_source = _identifier(schedule_source, label="schedule_source")
    if sampling["schedule_mode"] == "sampled" and schedule_source is not None:
        raise ValueError("sampled artifacts cannot have a schedule_source")

    acquisition_raw = manifest["acquisitions"]
    if not isinstance(acquisition_raw, list) or not acquisition_raw:
        raise ValueError("acquisitions must be a non-empty array")
    acquisitions = []
    for index, item in enumerate(acquisition_raw):
        if not isinstance(item, dict) or set(item) != {
            "acquisition_id",
            "index",
            "n_retained_frames",
            "n_segments",
            "session",
            "subject",
        }:
            raise ValueError("fixed-information acquisition record is invalid")
        if _integer(item["index"], label="acquisition index") != index:
            raise ValueError("acquisition indices must be consecutive and ordered")
        session = item["session"]
        if session is not None and (not isinstance(session, str) or not session):
            raise ValueError("acquisition session must be a non-empty string or null")
        acquisitions.append(
            InformationAcquisition(
                subject=_identifier(item["subject"], label="subject"),
                session=session,
                acquisition_id=_identifier(item["acquisition_id"], label="acquisition_id"),
                n_retained_frames=_integer(
                    item["n_retained_frames"], label="n_retained_frames", minimum=1
                ),
                n_segments=_integer(item["n_segments"], label="n_segments", minimum=1),
            )
        )
    if len({item.acquisition_id for item in acquisitions}) != len(acquisitions):
        raise ValueError("acquisition IDs must be unique")

    cell_raw = manifest["cells"]
    if not isinstance(cell_raw, list) or not cell_raw:
        raise ValueError("cells must be a non-empty array")
    cells = []
    for item in cell_raw:
        if not isinstance(item, dict) or set(item) != {
            "acquisition_index",
            "eligible_starts",
            "length",
        }:
            raise ValueError("fixed-information cell record is invalid")
        acquisition_index = _integer(item["acquisition_index"], label="cell acquisition_index")
        if acquisition_index >= len(acquisitions):
            raise ValueError("cell acquisition_index is outside the acquisition array")
        length = _integer(item["length"], label="cell length", minimum=1)
        if length not in lengths:
            raise ValueError("cell length is absent from the requested length set")
        cells.append(
            InformationCell(
                acquisition_index=acquisition_index,
                length=length,
                eligible_starts=_integer(
                    item["eligible_starts"], label="cell eligible_starts", minimum=1
                ),
            )
        )
    cell_keys = [(cell.acquisition_index, cell.length) for cell in cells]
    if len(set(cell_keys)) != len(cell_keys):
        raise ValueError("fixed-information cells contain duplicate acquisition/length units")
    length_order = {length: index for index, length in enumerate(lengths)}
    canonical_keys = sorted(cell_keys, key=lambda key: (key[0], length_order[key[1]]))
    if cell_keys != canonical_keys:
        raise ValueError("fixed-information cells are not in canonical acquisition/length order")
    if sampling["schedule_mode"] == "sampled" and {
        cell.acquisition_index for cell in cells
    } != set(range(len(acquisitions))):
        raise ValueError("sampled artifacts require at least one cell for every acquisition")

    expected_total = len(cells) * draws
    if _integer(manifest["n_cells"], label="n_cells") != len(cells) or _integer(
        manifest["n_draws"], label="n_draws"
    ) != expected_total:
        raise ValueError("fixed-information manifest counts are inconsistent")

    expected_arrays = BASE_ARRAY_NAMES | (CONDITIONAL_ARRAY_NAMES if condition is not None else set())
    if manifest["array_names"] != sorted(expected_arrays):
        raise ValueError("fixed-information array_names do not match the schema")
    try:
        with np.load(root / "arrays.npz", allow_pickle=False) as archive:
            if set(archive.files) != expected_arrays:
                raise ValueError("fixed-information arrays do not match the manifest")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read fixed-information arrays: {error}") from error
    integer_names = BASE_ARRAY_NAMES - {"mutual_information", "mean_mutual_information"}
    for name in integer_names:
        if arrays[name].dtype != np.dtype("int64") or arrays[name].shape != (expected_total,):
            raise ValueError(f"fixed-information integer array {name} has an invalid dtype or shape")
    mi_shape = (expected_total, len(left), len(right))
    if arrays["mutual_information"].dtype != np.dtype("float64") or arrays["mutual_information"].shape != mi_shape:
        raise ValueError("mutual_information has an invalid dtype or shape")
    if arrays["mean_mutual_information"].dtype != np.dtype("float64") or arrays["mean_mutual_information"].shape != (expected_total,):
        raise ValueError("mean_mutual_information has an invalid dtype or shape")
    if condition is not None:
        if arrays["conditional_mutual_information"].dtype != np.dtype("float64") or arrays["conditional_mutual_information"].shape != mi_shape:
            raise ValueError("conditional_mutual_information has an invalid dtype or shape")
        if arrays["mean_conditional_mutual_information"].dtype != np.dtype("float64") or arrays["mean_conditional_mutual_information"].shape != (expected_total,):
            raise ValueError("mean_conditional_mutual_information has an invalid dtype or shape")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("fixed-information arrays contain non-finite values")

    expected_acquisition = np.concatenate(
        [np.repeat(cell.acquisition_index, draws) for cell in cells]
    ).astype(np.int64, copy=False)
    expected_length = np.concatenate([np.repeat(cell.length, draws) for cell in cells]).astype(
        np.int64, copy=False
    )
    expected_eligible = np.concatenate(
        [np.repeat(cell.eligible_starts, draws) for cell in cells]
    ).astype(np.int64, copy=False)
    expected_draw = np.tile(np.arange(draws, dtype=np.int64), len(cells))
    if not np.array_equal(arrays["acquisition_index"], expected_acquisition):
        raise ValueError("acquisition_index does not follow the canonical grid order")
    if not np.array_equal(arrays["length"], expected_length) or not np.array_equal(arrays["draw"], expected_draw):
        raise ValueError("length/draw arrays do not follow the canonical grid order")
    if not np.array_equal(arrays["eligible_starts"], expected_eligible):
        raise ValueError("eligible_starts does not match the declared cells")
    if np.any(arrays["segment_id"] < 0) or np.any(arrays["start_within_segment"] < 0):
        raise ValueError("segment IDs and within-segment starts cannot be negative")
    for row, acquisition_index in enumerate(arrays["acquisition_index"]):
        acquisition = acquisitions[int(acquisition_index)]
        if arrays["segment_id"][row] >= acquisition.n_segments:
            raise ValueError("stored segment_id exceeds the acquisition segment count")
        if arrays["start_within_segment"][row] + arrays["length"][row] > acquisition.n_retained_frames:
            raise ValueError("stored within-segment interval exceeds retained acquisition frames")
    if np.any(arrays["start_frame"] < 0):
        raise ValueError("stored start_frame cannot be negative")
    if np.any(arrays["eligible_starts"] < 1):
        raise ValueError("eligible_starts must be positive")
    if not np.array_equal(arrays["end_frame"] - arrays["start_frame"] + 1, arrays["length"]):
        raise ValueError("stored frame bounds do not match stored lengths")
    if not np.array_equal(
        arrays["mean_mutual_information"], arrays["mutual_information"].mean(axis=(1, 2))
    ):
        raise ValueError("stored mean MI does not match pairwise MI")
    if condition is not None and not np.array_equal(
        arrays["mean_conditional_mutual_information"],
        arrays["conditional_mutual_information"].mean(axis=(1, 2)),
    ):
        raise ValueError("stored mean CMI does not match pairwise CMI")

    artifact = FixedInformationArtifact(
        groups=groups,
        roi_names=roi_names,
        left_indices=left,
        right_indices=right,
        conditioning_indices=condition,
        acquisitions=tuple(acquisitions),
        cells=tuple(cells),
        lengths=lengths,
        draws_per_length=draws,
        sample_seed=sample_seed,
        schedule_mode=str(sampling["schedule_mode"]),
        schedule_source=schedule_source,
        k=k,
        metric="chebyshev",
        jitter=jitter,
        jitter_seed=jitter_seed,
        standardized=estimator["standardized"],
        implementation=implementation,
        acquisition_index=_readonly(arrays["acquisition_index"]),
        length=_readonly(arrays["length"]),
        draw=_readonly(arrays["draw"]),
        segment_id=_readonly(arrays["segment_id"]),
        start_within_segment=_readonly(arrays["start_within_segment"]),
        start_frame=_readonly(arrays["start_frame"]),
        end_frame=_readonly(arrays["end_frame"]),
        eligible_starts=_readonly(arrays["eligible_starts"]),
        mutual_information=_readonly(arrays["mutual_information"]),
        conditional_mutual_information=(
            None if condition is None else _readonly(arrays["conditional_mutual_information"])
        ),
        mean_mutual_information=_readonly(arrays["mean_mutual_information"]),
        mean_conditional_mutual_information=(
            None
            if condition is None
            else _readonly(arrays["mean_conditional_mutual_information"])
        ),
    )
    draw_rows = _read_tsv(root / "draw_metrics.tsv", fields=DRAW_FIELDS, label="draw metrics")
    if len(draw_rows) != expected_total:
        raise ValueError("draw_metrics.tsv row count does not match the arrays")
    expected_draw_rows = _draw_rows(artifact)
    for index, (stored, expected) in enumerate(zip(draw_rows, expected_draw_rows, strict=True)):
        normalized = {field: str(expected[field]) for field in DRAW_FIELDS}
        if stored != normalized:
            raise ValueError(f"draw_metrics.tsv row {index + 2} does not match the arrays")
    session_rows = _read_tsv(
        root / "session_metrics.tsv", fields=SESSION_FIELDS, label="session metrics"
    )
    expected_session_rows = _session_rows(artifact)
    if len(session_rows) != len(expected_session_rows):
        raise ValueError("session_metrics.tsv row count does not match the arrays")
    for index, (stored, expected) in enumerate(
        zip(session_rows, expected_session_rows, strict=True)
    ):
        normalized = {field: str(expected[field]) for field in SESSION_FIELDS}
        if stored != normalized:
            raise ValueError(f"session_metrics.tsv row {index + 2} does not match the arrays")
    return artifact


__all__ = [
    "DRAW_FIELDS",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "GROUP_FIELDS",
    "SCHEDULE_FIELDS",
    "SESSION_FIELDS",
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
