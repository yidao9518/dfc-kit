"""Censor-aware multiplication of temporal derivatives (MTD)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..data import TimeSeriesRun
from .correlation import edge_index


@dataclass(frozen=True)
class MTDResult:
    """Instantaneous edge products of globally standardized valid derivatives."""

    features: NDArray[np.float64]
    standardized_derivatives: NDArray[np.float64]
    start_frames: NDArray[np.int64]
    end_frames: NDArray[np.int64]
    segment_ids: NDArray[np.int64]
    edge_i: NDArray[np.int64]
    edge_j: NDArray[np.int64]
    roi_names: tuple[str, ...]
    subject: str | None
    session: str | None
    acquisition_id: str | None = None


def _readonly(values: NDArray) -> NDArray:
    output = np.asarray(values).copy()
    output.setflags(write=False)
    return output


def _validated_nodes(
    nodes: Iterable[int],
    *,
    n_rois: int,
    label: str,
    minimum: int,
) -> NDArray[np.int64]:
    raw = tuple(nodes)
    if len(raw) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} ROI indices")
    if any(
        isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer))
        for node in raw
    ):
        raise TypeError(f"{label} must contain integer ROI indices")
    output = np.asarray(raw, dtype=np.int64)
    if len(set(output.tolist())) != len(output):
        raise ValueError(f"{label} contains duplicate ROI indices")
    if np.any(output < 0) or np.any(output >= n_rois):
        raise ValueError(f"{label} contains an ROI index outside [0, {n_rois})")
    return output


def globally_standardized_derivatives(
    run: TimeSeriesRun,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Return standardized valid derivatives and their original-frame metadata.

    First differences are formed only within contiguous retained-frame
    segments. All valid derivative rows are then centered and divided by the
    population standard deviation once over the complete run.
    """
    derivative_parts: list[NDArray[np.float64]] = []
    start_parts: list[NDArray[np.int64]] = []
    end_parts: list[NDArray[np.int64]] = []
    segment_parts: list[NDArray[np.int64]] = []

    for segment_id, positions in enumerate(run.segments()):
        if len(positions) < 2:
            continue
        derivative_parts.append(np.diff(run.values[positions], axis=0))
        original = run.original_indices[positions]
        start_parts.append(original[:-1])
        end_parts.append(original[1:])
        segment_parts.append(np.full(len(positions) - 1, segment_id, dtype=np.int64))

    if not derivative_parts:
        raise ValueError("MTD requires at least one valid within-segment derivative")

    derivatives = np.concatenate(derivative_parts, axis=0)
    scale = derivatives.std(axis=0, ddof=0)
    invalid = np.flatnonzero((~np.isfinite(scale)) | (scale < 1e-8))
    if len(invalid):
        raise ValueError(f"MTD derivative scale is undefined for ROI indices {invalid.tolist()}")
    standardized = (derivatives - derivatives.mean(axis=0)) / scale
    if not np.isfinite(standardized).all():
        raise ValueError("standardized MTD derivatives contain non-finite values")

    return (
        _readonly(standardized),
        _readonly(np.concatenate(start_parts)),
        _readonly(np.concatenate(end_parts)),
        _readonly(np.concatenate(segment_parts)),
    )


def cross_block_mtd(
    standardized_derivatives: ArrayLike,
    left: Iterable[int],
    right: Iterable[int],
) -> NDArray[np.float64]:
    """Return the mean cross-block MTD value at every valid derivative frame."""
    derivatives = np.asarray(standardized_derivatives, dtype=float)
    if derivatives.ndim != 2 or derivatives.shape[1] < 2:
        raise ValueError("standardized_derivatives must be a frames-by-ROI array")
    if not np.isfinite(derivatives).all():
        raise ValueError("standardized_derivatives contain non-finite values")
    left_nodes = _validated_nodes(left, n_rois=derivatives.shape[1], label="left block", minimum=1)
    right_nodes = _validated_nodes(
        right, n_rois=derivatives.shape[1], label="right block", minimum=1
    )
    overlap = sorted(set(left_nodes.tolist()).intersection(right_nodes.tolist()))
    if overlap:
        raise ValueError(f"cross-block MTD requires disjoint blocks; overlap={overlap}")
    values = (
        derivatives[:, left_nodes, None] * derivatives[:, None, right_nodes]
    ).mean(axis=(1, 2))
    return _readonly(values)


def within_block_mtd(
    standardized_derivatives: ArrayLike,
    nodes: Iterable[int],
) -> NDArray[np.float64]:
    """Return the off-diagonal mean within-block MTD at every valid frame."""
    derivatives = np.asarray(standardized_derivatives, dtype=float)
    if derivatives.ndim != 2 or derivatives.shape[1] < 2:
        raise ValueError("standardized_derivatives must be a frames-by-ROI array")
    if not np.isfinite(derivatives).all():
        raise ValueError("standardized_derivatives contain non-finite values")
    selected = _validated_nodes(
        nodes, n_rois=derivatives.shape[1], label="within block", minimum=2
    )
    left, right = np.triu_indices(len(selected), k=1)
    values = (derivatives[:, selected[left]] * derivatives[:, selected[right]]).mean(axis=1)
    return _readonly(values)


class MTD:
    """Calculate instantaneous MTD edge products without crossing censor gaps."""

    def transform(self, run: TimeSeriesRun) -> MTDResult:
        derivatives, start_frames, end_frames, segment_ids = globally_standardized_derivatives(
            run
        )
        left, right = edge_index(run.n_rois)
        features = derivatives[:, left] * derivatives[:, right]
        if not np.isfinite(features).all():
            raise ValueError("MTD edge products contain non-finite values")
        return MTDResult(
            features=_readonly(features),
            standardized_derivatives=derivatives,
            start_frames=start_frames,
            end_frames=end_frames,
            segment_ids=segment_ids,
            edge_i=_readonly(left),
            edge_j=_readonly(right),
            roi_names=run.roi_names,
            subject=run.subject,
            session=run.session,
            acquisition_id=run.acquisition_id,
        )
