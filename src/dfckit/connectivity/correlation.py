"""Correlation matrices and edge-vector utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._arrays import readonly_copy as _readonly
from .._validation import validated_integer


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


def edge_vector_to_symmetric_matrix(
    values: ArrayLike,
    edge_i: ArrayLike,
    edge_j: ArrayLike,
    *,
    n_nodes: int | None = None,
    diagonal: float = 0.0,
) -> NDArray[np.float64]:
    """Restore an undirected symmetric matrix from indexed edge values.

    ``edge_i`` and ``edge_j`` must identify each undirected edge once using
    upper-triangular orientation (``edge_i < edge_j``). The function accepts a
    complete upper triangle or a subset of edges; unspecified off-diagonal
    entries remain zero.
    """
    edges = np.asarray(values, dtype=float)
    raw_left = np.asarray(edge_i)
    raw_right = np.asarray(edge_j)
    if edges.ndim != 1 or raw_left.ndim != 1 or raw_right.ndim != 1:
        raise ValueError("values, edge_i, and edge_j must be one-dimensional")
    if edges.shape != raw_left.shape or edges.shape != raw_right.shape:
        raise ValueError("values, edge_i, and edge_j must have equal length")
    if not len(edges):
        raise ValueError("at least one edge is required")
    if not np.isfinite(edges).all():
        raise ValueError("values contain non-finite edges")
    if not np.issubdtype(raw_left.dtype, np.integer) or not np.issubdtype(
        raw_right.dtype, np.integer
    ):
        raise TypeError("edge_i and edge_j must contain integer node indices")
    left = raw_left.astype(np.int64, copy=False)
    right = raw_right.astype(np.int64, copy=False)
    if np.any(left < 0) or np.any(right < 0) or np.any(left >= right):
        raise ValueError("edge indices must satisfy 0 <= edge_i < edge_j")

    inferred_nodes = int(max(left.max(), right.max())) + 1
    nodes = (
        inferred_nodes
        if n_nodes is None
        else validated_integer(n_nodes, label="n_nodes", minimum=2)
    )
    if inferred_nodes > nodes:
        raise ValueError("edge index lies outside the requested matrix")
    keys = left * nodes + right
    if len(np.unique(keys)) != len(keys):
        raise ValueError("edge indices contain a duplicate undirected edge")
    diagonal_value = float(diagonal)
    if not np.isfinite(diagonal_value):
        raise ValueError("diagonal must be finite")

    matrix = np.zeros((nodes, nodes), dtype=float)
    matrix[left, right] = edges
    matrix[right, left] = edges
    np.fill_diagonal(matrix, diagonal_value)
    return _readonly(matrix)
