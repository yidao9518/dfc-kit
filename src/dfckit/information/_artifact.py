"""Persistence engine for fixed-length information artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .._arrays import readonly_copy as _readonly
from ..artifacts._numpy import write_numpy_artifact
from .fixed import (
    BASE_ARRAY_NAMES,
    CONDITIONAL_ARRAY_NAMES,
    FORMAT_NAME,
    FORMAT_VERSION,
    GROUP_FIELDS,
    FixedInformationArtifact,
    InformationAcquisition,
    InformationCell,
    InformationGroups,
    _finite,
    _identifier,
    _integer,
    _roi_group,
    _strict_json,
    _validated_lengths,
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


def save_fixed_information(
    artifact: FixedInformationArtifact,
    path: str | Path,
) -> Path:
    """Atomically write a new portable fixed-information artifact directory."""
    if not isinstance(artifact, FixedInformationArtifact):
        raise TypeError("artifact must be a FixedInformationArtifact")
    arrays = _arrays(artifact)
    manifest = _manifest(artifact, arrays)
    return write_numpy_artifact(
        path,
        manifest,
        arrays,
        label="fixed-information",
        validate=load_fixed_information,
    )


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
    """Load and cross-validate all arrays and manifest fields."""
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"fixed-information artifact directory does not exist: {root}")
    expected_files = {"arrays.npz", "manifest.json"}
    if {entry.name for entry in root.iterdir()} != expected_files:
        raise ValueError("fixed-information artifact must contain manifest.json and arrays.npz")
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
    return artifact
