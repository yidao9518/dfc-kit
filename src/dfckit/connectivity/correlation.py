"""Correlation matrices and edge-vector utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def edge_index(n_rois: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if n_rois < 2:
        raise ValueError("at least two ROIs are required")
    left, right = np.triu_indices(n_rois, k=1)
    return left.astype(np.int64), right.astype(np.int64)


def weighted_correlation(
    values: ArrayLike,
    weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute a finite weighted Pearson correlation matrix."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("values must have at least two frames and two ROIs")
    if not np.isfinite(data).all():
        raise ValueError("values contain non-finite samples")

    if weights is None:
        weight = np.ones(data.shape[0], dtype=float)
    else:
        weight = np.asarray(weights, dtype=float)
    if weight.ndim != 1 or len(weight) != len(data):
        raise ValueError("weights must be one-dimensional and match the frame count")
    if not np.isfinite(weight).all() or np.any(weight < 0) or weight.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and have positive sum")

    mean = np.average(data, axis=0, weights=weight)
    centered = data - mean
    covariance = (centered * weight[:, None]).T @ centered / weight.sum()
    variance = np.diag(covariance)
    if np.any(~np.isfinite(variance)) or np.any(variance <= 1e-12):
        raise ValueError("correlation is undefined for a constant or invalid ROI")
    scale = np.sqrt(variance)
    correlation = covariance / np.outer(scale, scale)
    np.fill_diagonal(correlation, 1.0)
    if not np.isfinite(correlation).all():
        raise ValueError("correlation matrix contains non-finite values")
    return np.clip(correlation, -1.0, 1.0)


def fisher_z_edges(
    correlation: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Return finite upper-triangular Fisher-z edges and their node indices."""
    matrix = np.asarray(correlation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("correlation must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("correlation contains non-finite values")
    left, right = edge_index(matrix.shape[0])
    edges = np.arctanh(np.clip(matrix[left, right], -0.999999, 0.999999))
    return edges, left, right
