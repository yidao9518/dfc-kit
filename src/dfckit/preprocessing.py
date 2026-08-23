"""Shared preprocessing for censor-bounded time series."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._arrays import readonly_copy as _readonly
from .data import TimeSeriesRun


def segment_standardized_samples(
    run: TimeSeriesRun,
    *,
    method_name: str = "ETS",
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Z-score each ROI within each retained contiguous segment.

    Segments shorter than two frames are omitted. Constant ROI columns become
    zero after centering, and returned rows retain their original frame and
    segment identities.
    """
    samples: list[NDArray[np.float64]] = []
    original_indices: list[NDArray[np.int64]] = []
    segment_ids: list[NDArray[np.int64]] = []

    for segment_id, positions in enumerate(run.segments()):
        if len(positions) < 2:
            continue
        values = run.values[positions]
        scale = values.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        samples.append((values - values.mean(axis=0)) / scale)
        original_indices.append(run.original_indices[positions])
        segment_ids.append(np.full(len(positions), segment_id, dtype=np.int64))

    if not samples:
        raise ValueError(f"{method_name} requires at least one retained segment with two frames")
    standardized = np.concatenate(samples, axis=0)
    if not np.isfinite(standardized).all():
        raise ValueError(f"standardized {method_name} samples contain non-finite values")
    return (
        _readonly(standardized),
        _readonly(np.concatenate(original_indices)),
        _readonly(np.concatenate(segment_ids)),
    )
