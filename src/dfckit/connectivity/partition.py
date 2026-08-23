"""Metrics for predefined network partitions on positive weighted graphs."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._arrays import readonly_copy as _readonly


def _validated_matrix(
    values: ArrayLike,
    *,
    label: str,
    nonnegative: bool = False,
    zero_diagonal: bool = False,
) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(f"{label} must be a square matrix with at least two nodes")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{label} must be symmetric")
    if nonnegative and np.any(matrix < -1e-12):
        raise ValueError(f"{label} must contain non-negative weights")
    if zero_diagonal and not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label} diagonal must be zero")
    return matrix


def _partition_codes(
    partition: Iterable[Hashable],
    *,
    n_nodes: int,
) -> tuple[NDArray[np.int64], tuple[Hashable, ...]]:
    labels = tuple(partition)
    if len(labels) != n_nodes:
        raise ValueError("partition must contain exactly one label per node")
    mapping: dict[Hashable, int] = {}
    codes = np.empty(n_nodes, dtype=np.int64)
    for index, label in enumerate(labels):
        if label is None:
            raise ValueError("partition labels cannot be None")
        if isinstance(label, (float, np.floating)) and not np.isfinite(label):
            raise ValueError("partition labels cannot be non-finite")
        try:
            if label not in mapping:
                mapping[label] = len(mapping)
        except TypeError as error:
            raise TypeError("partition labels must be hashable") from error
        codes[index] = mapping[label]
    return _readonly(codes), tuple(mapping)


def _normalized_trapezoid(
    values: ArrayLike,
    coordinates: ArrayLike,
    *,
    axis: int = 0,
) -> NDArray[np.float64]:
    samples = np.asarray(values, dtype=float)
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 1 or len(points) < 2 or np.any(np.diff(points) <= 0):
        raise ValueError("coordinates must contain at least two strictly increasing values")
    if samples.shape[axis] != len(points):
        raise ValueError("values along axis must match coordinates")
    moved = np.moveaxis(samples, axis, 0)
    widths = np.diff(points)
    width_shape = (-1,) + (1,) * (moved.ndim - 1)
    trapezoids = (moved[:-1] + moved[1:]) * widths.reshape(width_shape) / 2.0
    integrated = np.sum(trapezoids, axis=0)
    return np.asarray(integrated / (points[-1] - points[0]), dtype=float)


@dataclass(frozen=True)
class PartitionEdgeSummary:
    """Edge-weight summaries within and between predefined communities."""

    within_mean: float
    between_mean: float
    within_minus_between: float
    segregation: float
    within_by_community: NDArray[np.float64]
    between_by_community: NDArray[np.float64]
    partition_codes: NDArray[np.int64]
    community_labels: tuple[Hashable, ...]


@dataclass(frozen=True)
class FixedPartitionGraphResult:
    """Fixed-partition graph metrics evaluated over proportional densities."""

    densities: NDArray[np.float64]
    achieved_densities: NDArray[np.float64]
    modularity: NDArray[np.float64]
    participation: NDArray[np.float64]
    node_strength: NDArray[np.float64]
    modularity_auc: float
    participation_auc: NDArray[np.float64]
    node_strength_auc: NDArray[np.float64]
    achieved_density_auc: float
    partition_codes: NDArray[np.int64]
    community_labels: tuple[Hashable, ...]


def positive_proportional_adjacency(
    connectivity: ArrayLike,
    density: float,
) -> NDArray[np.float64]:
    """Keep the strongest positive undirected edges at a nominal density.

    The target edge count is rounded from all possible undirected edges. If
    fewer positive edges exist, every positive edge is retained and the
    achieved density is lower than the nominal density.
    """
    matrix = _validated_matrix(connectivity, label="connectivity")
    if not np.isfinite(density) or not 0.0 < density <= 1.0:
        raise ValueError("density must be finite and in (0, 1]")
    left, right = np.triu_indices(len(matrix), k=1)
    weights = matrix[left, right]
    positive = np.flatnonzero(weights > 0.0)
    target = max(1, round(float(density) * len(weights)))
    count = min(target, len(positive))
    adjacency = np.zeros_like(matrix, dtype=float)
    if count:
        order = np.argsort(weights[positive], kind="stable")
        selected = positive[order[-count:]]
        adjacency[left[selected], right[selected]] = weights[selected]
    adjacency += adjacency.T
    return _readonly(adjacency)


def achieved_density(adjacency: ArrayLike) -> float:
    """Return the fraction of possible undirected edges with positive weight."""
    matrix = _validated_matrix(
        adjacency,
        label="adjacency",
        nonnegative=True,
        zero_diagonal=True,
    )
    possible = len(matrix) * (len(matrix) - 1) // 2
    return float(np.count_nonzero(matrix[np.triu_indices(len(matrix), k=1)] > 0) / possible)


def fixed_partition_modularity(
    adjacency: ArrayLike,
    partition: Iterable[Hashable],
) -> float:
    """Return positive weighted Newman-Girvan modularity for a fixed partition."""
    matrix = _validated_matrix(
        adjacency,
        label="adjacency",
        nonnegative=True,
        zero_diagonal=True,
    )
    codes, _ = _partition_codes(partition, n_nodes=len(matrix))
    total = float(matrix.sum())
    if total <= 1e-12:
        return 0.0
    strength = matrix.sum(axis=1)
    expected = np.outer(strength, strength) / total
    same_community = codes[:, None] == codes[None, :]
    return float((matrix - expected)[same_community].sum() / total)


def participation_coefficient(
    adjacency: ArrayLike,
    partition: Iterable[Hashable],
) -> NDArray[np.float64]:
    """Return each node's positive weighted participation coefficient."""
    matrix = _validated_matrix(
        adjacency,
        label="adjacency",
        nonnegative=True,
        zero_diagonal=True,
    )
    codes, community_labels = _partition_codes(partition, n_nodes=len(matrix))
    strength = matrix.sum(axis=1)
    output = np.zeros(len(matrix), dtype=float)
    connected = strength > 1e-12
    for community in range(len(community_labels)):
        community_strength = matrix[:, codes == community].sum(axis=1)
        share = np.zeros(len(matrix), dtype=float)
        np.divide(community_strength, strength, out=share, where=connected)
        output -= share**2
    output[connected] += 1.0
    return _readonly(np.clip(output, 0.0, 1.0))


def partition_edge_summary(
    edge_weights: ArrayLike,
    partition: Iterable[Hashable],
) -> PartitionEdgeSummary:
    """Summarize each undirected edge once under a predefined partition."""
    matrix = _validated_matrix(edge_weights, label="edge_weights")
    codes, community_labels = _partition_codes(partition, n_nodes=len(matrix))
    left, right = np.triu_indices(len(matrix), k=1)
    values = matrix[left, right]
    within_mask = codes[left] == codes[right]
    between_mask = ~within_mask
    within = float(values[within_mask].mean()) if np.any(within_mask) else np.nan
    between = float(values[between_mask].mean()) if np.any(between_mask) else np.nan
    difference = float(within - between) if np.isfinite(within) and np.isfinite(between) else np.nan
    segregation = (
        float(difference / within)
        if np.isfinite(difference) and abs(within) > 1e-12
        else np.nan
    )

    n_communities = len(community_labels)
    within_by = np.full(n_communities, np.nan, dtype=float)
    between_by = np.full((n_communities, n_communities), np.nan, dtype=float)
    for community in range(n_communities):
        selected = within_mask & (codes[left] == community)
        if np.any(selected):
            within_by[community] = values[selected].mean()
        for other in range(community + 1, n_communities):
            selected = (
                ((codes[left] == community) & (codes[right] == other))
                | ((codes[left] == other) & (codes[right] == community))
            )
            if np.any(selected):
                between_by[community, other] = between_by[other, community] = values[
                    selected
                ].mean()

    return PartitionEdgeSummary(
        within_mean=within,
        between_mean=between,
        within_minus_between=difference,
        segregation=segregation,
        within_by_community=_readonly(within_by),
        between_by_community=_readonly(between_by),
        partition_codes=codes,
        community_labels=community_labels,
    )


class FixedPartitionGraph:
    """Evaluate fixed-partition graph organization across positive densities."""

    def __init__(self, densities: Iterable[float] = (0.10, 0.15, 0.20, 0.25, 0.30)) -> None:
        values = np.asarray(tuple(densities), dtype=float)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("densities must contain at least two values")
        if not np.isfinite(values).all() or np.any((values <= 0.0) | (values > 1.0)):
            raise ValueError("densities must be finite and in (0, 1]")
        if np.any(np.diff(values) <= 0):
            raise ValueError("densities must be strictly increasing")
        self.densities = _readonly(values)

    def transform(
        self,
        connectivity: ArrayLike,
        partition: Iterable[Hashable],
    ) -> FixedPartitionGraphResult:
        matrix = _validated_matrix(connectivity, label="connectivity")
        codes, labels = _partition_codes(partition, n_nodes=len(matrix))
        modularity = np.empty(len(self.densities), dtype=float)
        participation = np.empty((len(self.densities), len(matrix)), dtype=float)
        strength = np.empty_like(participation)
        actual_density = np.empty(len(self.densities), dtype=float)

        for index, density in enumerate(self.densities):
            adjacency = positive_proportional_adjacency(matrix, float(density))
            actual_density[index] = achieved_density(adjacency)
            modularity[index] = fixed_partition_modularity(adjacency, codes)
            participation[index] = participation_coefficient(adjacency, codes)
            strength[index] = adjacency.sum(axis=1)

        modularity_auc = float(_normalized_trapezoid(modularity, self.densities))
        participation_auc = _normalized_trapezoid(participation, self.densities, axis=0)
        strength_auc = _normalized_trapezoid(strength, self.densities, axis=0)
        density_auc = float(_normalized_trapezoid(actual_density, self.densities))
        return FixedPartitionGraphResult(
            densities=self.densities,
            achieved_densities=_readonly(actual_density),
            modularity=_readonly(modularity),
            participation=_readonly(participation),
            node_strength=_readonly(strength),
            modularity_auc=modularity_auc,
            participation_auc=_readonly(participation_auc),
            node_strength_auc=_readonly(strength_auc),
            achieved_density_auc=density_auc,
            partition_codes=codes,
            community_labels=labels,
        )
