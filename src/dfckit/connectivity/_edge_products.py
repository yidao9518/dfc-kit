"""Shared instantaneous edge-product kernels.

ETS and MTD differ in how they construct their ROI samples. Once those
samples have been constructed, both methods use the same upper-triangular
outer-product operation.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._arrays import readonly_copy as _readonly
from .._validation import validated_roi_indices
from .correlation import edge_index


def _validated_samples(samples: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("samples must be a frames-by-ROI array")
    if not np.isfinite(values).all():
        raise ValueError("samples contain non-finite values")
    return values


def edge_products(samples: ArrayLike) -> NDArray[np.float64]:
    """Return all upper-triangular instantaneous ROI edge products.

    The rows may be segment-standardized signals (ETS) or standardized
    temporal derivatives (MTD); the kernel intentionally has no method-specific
    temporal semantics.
    """
    values = _validated_samples(samples)
    left, right = edge_index(values.shape[1])
    products = values[:, left] * values[:, right]
    if not np.isfinite(products).all():
        raise ValueError("edge products contain non-finite values")
    return _readonly(products)


def edge_rss(samples: ArrayLike) -> NDArray[np.float64]:
    """Compute all-edge root-sum-square amplitude without materializing edges."""
    values = _validated_samples(samples)
    squared = values * values
    squared_rss = ((squared.sum(axis=1) ** 2 - (squared * squared).sum(axis=1)) / 2).clip(min=0.0)
    return _readonly(np.sqrt(squared_rss))


def cross_block_products(
    samples: ArrayLike,
    left: Iterable[int],
    right: Iterable[int],
) -> NDArray[np.float64]:
    """Return the mean product across two disjoint ROI blocks per row."""
    values = _validated_samples(samples)
    left_nodes = validated_roi_indices(left, n_rois=values.shape[1], label="left block", minimum=1)
    right_nodes = validated_roi_indices(
        right, n_rois=values.shape[1], label="right block", minimum=1
    )
    overlap = sorted(set(left_nodes.tolist()).intersection(right_nodes.tolist()))
    if overlap:
        raise ValueError(f"cross-block products require disjoint blocks; overlap={overlap}")
    products = (values[:, left_nodes, None] * values[:, None, right_nodes]).mean(axis=(1, 2))
    return _readonly(products)


def within_block_products(
    samples: ArrayLike,
    nodes: Iterable[int],
) -> NDArray[np.float64]:
    """Return the off-diagonal mean product within one ROI block per row."""
    values = _validated_samples(samples)
    selected = validated_roi_indices(nodes, n_rois=values.shape[1], label="within block", minimum=2)
    left, right = np.triu_indices(len(selected), k=1)
    products = (values[:, selected[left]] * values[:, selected[right]]).mean(axis=1)
    return _readonly(products)


def _edge_block_mask(
    result: object,
    left_nodes: NDArray[np.int64],
    right_nodes: NDArray[np.int64],
) -> NDArray[np.bool_]:
    edge_i = np.asarray(result.edge_i, dtype=np.int64)
    edge_j = np.asarray(result.edge_j, dtype=np.int64)
    return np.isin(edge_i, left_nodes) & np.isin(edge_j, right_nodes) | (
        np.isin(edge_i, right_nodes) & np.isin(edge_j, left_nodes)
    )


def cross_block_edge_summary(
    result: object,
    left: Iterable[int],
    right: Iterable[int],
) -> NDArray[np.float64]:
    """Summarize cross-block products from a common edge result."""
    n_rois = len(result.roi_names)
    left_nodes = validated_roi_indices(left, n_rois=n_rois, label="left block", minimum=1)
    right_nodes = validated_roi_indices(right, n_rois=n_rois, label="right block", minimum=1)
    overlap = sorted(set(left_nodes.tolist()).intersection(right_nodes.tolist()))
    if overlap:
        raise ValueError(f"cross-block products require disjoint blocks; overlap={overlap}")
    if result.features is not None:
        selected = _edge_block_mask(result, left_nodes, right_nodes)
        features = np.asarray(result.features, dtype=float)
        if not np.any(selected):
            raise ValueError("cross-block product summary selected no edges")
        return _readonly(features[:, selected].mean(axis=1))
    raise ValueError("cross-block summary requires materialized edge features")


def within_block_edge_summary(
    result: object,
    nodes: Iterable[int],
) -> NDArray[np.float64]:
    """Summarize within-block products from a common edge result."""
    n_rois = len(result.roi_names)
    selected_nodes = validated_roi_indices(nodes, n_rois=n_rois, label="within block", minimum=2)
    if result.features is not None:
        edge_i = np.asarray(result.edge_i, dtype=np.int64)
        edge_j = np.asarray(result.edge_j, dtype=np.int64)
        selected = np.isin(edge_i, selected_nodes) & np.isin(edge_j, selected_nodes)
        features = np.asarray(result.features, dtype=float)
        if not np.any(selected):
            raise ValueError("within-block product summary selected no edges")
        return _readonly(features[:, selected].mean(axis=1))
    raise ValueError("within-block summary requires materialized edge features")
