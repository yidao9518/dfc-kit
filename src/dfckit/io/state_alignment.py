"""Portable persistence for state-number alignment mappings."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from ..states.alignment import StateAlignment

FORMAT_NAME = "dfckit-state-alignment"
FORMAT_VERSION = 1


def save_state_alignment(alignment: StateAlignment, path: str | Path) -> Path:
    """Write one alignment as strict JSON metadata and numeric arrays."""
    if not isinstance(alignment, StateAlignment):
        raise TypeError("alignment must be a StateAlignment object")
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"state-alignment artifact path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "candidate_to_reference": np.asarray(alignment.candidate_to_reference),
        "matched_correlations": np.asarray(alignment.matched_correlations),
        "correlation_matrix": np.asarray(alignment.correlation_matrix),
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
        "method": alignment.method,
    }
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        np.savez(temporary / "arrays.npz", **arrays)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"state-alignment artifact path already exists: {target}")
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def load_state_alignment(path: str | Path) -> StateAlignment:
    """Load and validate an alignment written by :func:`save_state_alignment`."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"state-alignment artifact directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    arrays_path = root / "arrays.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError("state-alignment artifact requires manifest.json and arrays.npz")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"cannot read state-alignment manifest: {error}") from error
    expected_fields = {
        "array_names",
        "candidate_seed",
        "feature_keys",
        "format",
        "format_version",
        "method",
        "reference_seed",
        "sample_interval_seconds",
        "source_contract",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise ValueError("state-alignment manifest fields do not match the schema")
    if manifest["format"] != FORMAT_NAME or manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported state-alignment artifact format or version")
    expected_arrays = {
        "candidate_to_reference",
        "matched_correlations",
        "correlation_matrix",
    }
    names = manifest["array_names"]
    if not isinstance(names, list) or set(names) != expected_arrays or len(names) != 3:
        raise ValueError("state-alignment array_names is invalid")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_arrays:
                raise ValueError("state-alignment arrays do not match the manifest")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read state-alignment arrays: {error}") from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("state-alignment arrays cannot contain objects")
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
        matched_correlations=arrays["matched_correlations"],
        correlation_matrix=arrays["correlation_matrix"],
        reference_seed=manifest["reference_seed"],
        candidate_seed=manifest["candidate_seed"],
        feature_keys=feature_keys,
        source_contract=manifest["source_contract"],
        sample_interval_seconds=manifest["sample_interval_seconds"],
        method=manifest["method"],
    )


__all__ = ["load_state_alignment", "save_state_alignment"]
