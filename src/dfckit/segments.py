"""Censor-aware segmentation and window indexing."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validated_indices(original_indices: ArrayLike) -> NDArray[np.int64]:
    indices = np.asarray(original_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("original_indices must be one-dimensional")
    if len(indices) and np.any(np.diff(indices) <= 0):
        raise ValueError("original_indices must be strictly increasing")
    return indices


def contiguous_segments(
    original_indices: ArrayLike, min_length: int = 1
) -> tuple[NDArray[np.int64], ...]:
    """Return retained row positions grouped by consecutive original frames."""
    if min_length < 1:
        raise ValueError("min_length must be at least one")
    indices = _validated_indices(original_indices)
    if not len(indices):
        return ()
    split_at = np.flatnonzero(np.diff(indices) != 1) + 1
    positions = np.arange(len(indices), dtype=np.int64)
    return tuple(part for part in np.split(positions, split_at) if len(part) >= min_length)


def window_positions(
    original_indices: ArrayLike, length: int, step: int
) -> tuple[tuple[int, NDArray[np.int64]], ...]:
    """Return segment IDs and retained row positions for valid windows."""
    if length < 2:
        raise ValueError("length must be at least two")
    if step < 1:
        raise ValueError("step must be at least one")
    output = []
    for segment_id, segment in enumerate(contiguous_segments(original_indices)):
        for start in range(0, len(segment) - length + 1, step):
            output.append((segment_id, segment[start : start + length]))
    return tuple(output)
