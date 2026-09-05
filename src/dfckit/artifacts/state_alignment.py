"""Portable persistence for state-number alignment mappings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..states.alignment import StateAlignment
from ._numpy import load_numpy_artifact, write_numpy_artifact

FORMAT_NAME = "dfckit-state-alignment"
FORMAT_VERSION = 2


def save_state_alignment(alignment: StateAlignment, path: str | Path) -> Path:
    """Write one alignment as strict JSON metadata and numeric arrays."""
    if not isinstance(alignment, StateAlignment):
        raise TypeError("alignment must be a StateAlignment object")
    arrays = {
        "candidate_to_reference": np.asarray(alignment.candidate_to_reference),
        "matched_costs": np.asarray(alignment.matched_costs),
        "cost_matrix": np.asarray(alignment.cost_matrix),
    }
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "array_names": sorted(arrays),
        "reference_seed": alignment.reference_seed,
        "candidate_seed": alignment.candidate_seed,
        "feature_keys": [list(key) for key in alignment.feature_keys],
        "source_contract": alignment.source_contract,
        "sample_interval_seconds": alignment.sample_interval_seconds,
        "metric": alignment.metric,
    }
    return write_numpy_artifact(path, manifest, arrays, label="state-alignment")


def load_state_alignment(path: str | Path) -> StateAlignment:
    """Load and validate an alignment written by :func:`save_state_alignment`."""
    expected_fields = {
        "array_names",
        "candidate_seed",
        "feature_keys",
        "format",
        "format_version",
        "metric",
        "reference_seed",
        "sample_interval_seconds",
        "source_contract",
    }
    manifest, arrays = load_numpy_artifact(
        path,
        label="state-alignment",
        manifest_fields=expected_fields,
    )
    if manifest["format"] != FORMAT_NAME or manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported state-alignment artifact format or version")
    expected_arrays = {
        "candidate_to_reference",
        "matched_costs",
        "cost_matrix",
    }
    names = manifest["array_names"]
    if not isinstance(names, list) or set(names) != expected_arrays or len(names) != 3:
        raise ValueError("state-alignment array_names is invalid")
    keys = manifest["feature_keys"]
    if not isinstance(keys, list):
        raise TypeError("state-alignment feature_keys must be a list")
    if any(
        not isinstance(key, list)
        or not key
        or any(not isinstance(part, str) or not part for part in key)
        for key in keys
    ):
        raise ValueError("state-alignment feature_keys are invalid")
    feature_keys = tuple(tuple(key) for key in keys)
    return StateAlignment(
        candidate_to_reference=arrays["candidate_to_reference"],
        matched_costs=arrays["matched_costs"],
        cost_matrix=arrays["cost_matrix"],
        reference_seed=manifest["reference_seed"],
        candidate_seed=manifest["candidate_seed"],
        feature_keys=feature_keys,
        source_contract=manifest["source_contract"],
        sample_interval_seconds=manifest["sample_interval_seconds"],
        metric=manifest["metric"],
    )


__all__ = ["load_state_alignment", "save_state_alignment"]
