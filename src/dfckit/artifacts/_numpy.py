"""Shared writer for directory artifacts containing JSON and NumPy arrays."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._json import load_json_object, write_json_atomic


def load_numpy_artifact(
    path: str | Path,
    *,
    label: str,
    manifest_fields: Collection[str],
) -> tuple[dict[str, Any], dict[str, NDArray]]:
    """Read the common ``manifest.json`` plus ``arrays.npz`` artifact shell."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"{label} artifact directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    arrays_path = root / "arrays.npz"
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"{label} artifact requires manifest.json and arrays.npz")
    manifest = load_json_object(manifest_path, context=f"{label} artifact manifest")
    if set(manifest) != set(manifest_fields):
        raise ValueError(f"{label} artifact manifest fields do not match the schema")
    names = manifest.get("array_names")
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"{label} artifact array_names is invalid")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            if set(archive.files) != set(names):
                raise ValueError(f"{label} artifact arrays do not match the manifest")
            arrays = {name: np.array(archive[name], copy=True) for name in names}
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {label} artifact arrays: {error}") from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError(f"{label} artifact arrays cannot contain objects")
    return manifest, arrays


def write_numpy_artifact(
    target: str | Path,
    manifest: Mapping[str, object],
    arrays: Mapping[str, NDArray],
    *,
    label: str,
    validate: Callable[[Path], object] | None = None,
) -> Path:
    """Atomically create a new ``manifest.json`` plus ``arrays.npz`` directory.

    Format-specific loaders remain responsible for schema validation. ``validate``
    is an optional round-trip check run against the temporary directory before it
    becomes visible at ``target``.
    """
    destination = Path(target)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{label} artifact path already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    if not normalized or any(not isinstance(name, str) or not name for name in normalized):
        raise ValueError(f"{label} artifact arrays must have non-empty names")
    if any(array.dtype.hasobject or not np.isfinite(array).all() for array in normalized.values()):
        raise ValueError(f"{label} artifact arrays must be finite numeric arrays")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        np.savez(temporary / "arrays.npz", **normalized)
        write_json_atomic(temporary / "manifest.json", dict(manifest))
        if validate is not None:
            validate(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"{label} artifact path already exists: {destination}")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = ["load_numpy_artifact", "write_numpy_artifact"]
