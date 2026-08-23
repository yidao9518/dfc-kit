"""Shared writer for directory artifacts containing JSON and NumPy arrays."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._json import write_json_atomic


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


__all__ = ["write_numpy_artifact"]
