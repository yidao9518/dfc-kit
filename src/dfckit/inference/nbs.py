"""Paired network-based statistic (NBS)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._arrays import readonly_copy as _readonly
from .._validation import validated_integer, validated_subject_ids

DIRECTIONS = ("positive", "negative")
ALTERNATIVES = ("two-sided", "greater", "less")
COMPONENT_STATISTICS = ("edge_extent", "sum_abs_statistic")
COMPONENT_SIGN_MODES = ("separate", "pooled")


def _validated_edge_data(
    edge_i: ArrayLike,
    edge_j: ArrayLike,
    *,
    n_nodes: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    left = np.asarray(edge_i)
    right = np.asarray(edge_j)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or not len(left):
        raise ValueError("edge_i and edge_j must be aligned non-empty vectors")
    if not np.issubdtype(left.dtype, np.integer) or not np.issubdtype(
        right.dtype, np.integer
    ):
        raise TypeError("edge endpoints must be integers")
    left = left.astype(np.int64, copy=False)
    right = right.astype(np.int64, copy=False)
    if np.any(left < 0) or np.any(right >= n_nodes) or np.any(left >= right):
        raise ValueError(
            "every undirected edge must satisfy 0 <= edge_i < edge_j < n_nodes"
        )
    keys = left * n_nodes + right
    if len(np.unique(keys)) != len(keys):
        raise ValueError("edge endpoint arrays contain duplicate undirected edges")
    return left, right


def _validated_thresholds(thresholds: Iterable[float]) -> tuple[float, ...]:
    output = tuple(float(threshold) for threshold in thresholds)
    if not output:
        raise ValueError("thresholds must contain at least one value")
    if any(not np.isfinite(threshold) or threshold <= 0.0 for threshold in output):
        raise ValueError("NBS thresholds must be finite and positive")
    if len(set(output)) != len(output):
        raise ValueError("NBS thresholds must be unique")
    return output


def _validated_differences(
    differences: ArrayLike,
    *,
    n_edges: int,
) -> NDArray[np.float64]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != n_edges:
        raise ValueError(
            "differences must be a participants-by-edges array with at least two participants"
        )
    if not np.isfinite(values).all():
        raise ValueError("differences contain non-finite values")
    return values


def _prepare_design(
    n_observations: int,
    confounds: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64], int]:
    if confounds is None:
        centered = None
        design = np.ones((n_observations, 1), dtype=float)
    else:
        nuisance = np.asarray(confounds, dtype=float)
        if nuisance.ndim == 1:
            nuisance = nuisance[:, None]
        if nuisance.ndim != 2 or len(nuisance) != n_observations or not nuisance.shape[1]:
            raise ValueError(
                "confounds must align with participants and contain at least one column"
            )
        if not np.isfinite(nuisance).all():
            raise ValueError("confounds contain non-finite values")
        centered = nuisance - nuisance.mean(axis=0, keepdims=True)
        design = np.column_stack((np.ones(n_observations), centered))
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError("NBS design matrix is rank deficient")
    degrees_of_freedom = n_observations - rank
    if degrees_of_freedom <= 0:
        raise ValueError("NBS design has no residual degrees of freedom")
    inverse = np.linalg.inv(design.T @ design)
    return design, centered, inverse, degrees_of_freedom


def _intercept_t_with_design(
    values: NDArray[np.float64],
    design: NDArray[np.float64],
    inverse: NDArray[np.float64],
    degrees_of_freedom: int,
) -> NDArray[np.float64]:
    coefficients = inverse @ design.T @ values
    residual = values - design @ coefficients
    variance = np.square(residual).sum(axis=0) / degrees_of_freedom
    standard_error = np.sqrt(variance * inverse[0, 0])
    invalid = (~np.isfinite(standard_error)) | (standard_error <= 0.0)
    if np.any(invalid):
        indices = np.flatnonzero(invalid)[:10].tolist()
        raise ValueError(
            "edge t statistic has zero or non-finite standard error at edge indices "
            f"{indices}"
        )
    statistic = coefficients[0] / standard_error
    if not np.isfinite(statistic).all():
        raise ValueError("computed edge t statistics are non-finite")
    return statistic


def _intercept_t_statistic(
    differences: ArrayLike,
    confounds: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return the classical t statistic for each paired-difference intercept.

    Confounds are centered internally, so the intercept estimates the paired
    difference at the observed mean of every nuisance variable.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not values.shape[1]:
        raise ValueError("differences must be a non-empty participants-by-edges array")
    if not np.isfinite(values).all():
        raise ValueError("differences contain non-finite values")
    design, _, inverse, degrees = _prepare_design(len(values), confounds)
    return _readonly(_intercept_t_with_design(values, design, inverse, degrees))


@dataclass(frozen=True)
class NBSComponent:
    """One connected set of supra-threshold edges."""

    direction: str
    node_indices: tuple[int, ...]
    edge_indices: tuple[int, ...]
    component_statistic: str
    statistic_value: float
    fwe_pvalue: float | None = None

    @property
    def n_nodes(self) -> int:
        return len(self.node_indices)

    @property
    def edge_extent(self) -> int:
        return len(self.edge_indices)


def _component_value(
    edge_indices: Sequence[int],
    statistics: NDArray[np.float64],
    component_statistic: str,
) -> float:
    if component_statistic == "edge_extent":
        return float(len(edge_indices))
    if component_statistic == "sum_abs_statistic":
        return float(np.abs(statistics[np.asarray(edge_indices, dtype=np.int64)]).sum())
    raise ValueError(f"unknown component_statistic: {component_statistic}")


def _tail_components(
    statistics: NDArray[np.float64],
    edge_i: NDArray[np.int64],
    edge_j: NDArray[np.int64],
    *,
    n_nodes: int,
    selected: NDArray[np.bool_],
    direction: str,
    component_statistic: str,
) -> tuple[NBSComponent, ...]:
    chosen = np.flatnonzero(selected)
    if not len(chosen):
        return ()
    parent = np.arange(n_nodes, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for edge in chosen:
        union(int(edge_i[edge]), int(edge_j[edge]))
    grouped: dict[int, list[int]] = {}
    for edge in chosen:
        grouped.setdefault(find(int(edge_i[edge])), []).append(int(edge))

    output = []
    for indices in grouped.values():
        sorted_edges = tuple(sorted(indices))
        nodes = tuple(
            sorted(
                {int(edge_i[index]) for index in sorted_edges}
                | {int(edge_j[index]) for index in sorted_edges}
            )
        )
        output.append(
            NBSComponent(
                direction=direction,
                node_indices=nodes,
                edge_indices=sorted_edges,
                component_statistic=component_statistic,
                statistic_value=_component_value(
                    sorted_edges, statistics, component_statistic
                ),
            )
        )
    output.sort(
        key=lambda component: (
            -component.statistic_value,
            -component.edge_extent,
            -component.n_nodes,
            component.node_indices,
            component.edge_indices,
        )
    )
    return tuple(output)


def _threshold_components(
    statistics: ArrayLike,
    edge_i: ArrayLike,
    edge_j: ArrayLike,
    n_nodes: int,
    threshold: float,
    *,
    component_statistic: str = "edge_extent",
    component_sign_mode: str = "separate",
) -> dict[str, tuple[NBSComponent, ...]]:
    """Return supra-threshold components under the selected sign definition."""
    nodes = validated_integer(n_nodes, label="n_nodes", minimum=2)
    left, right = _validated_edge_data(edge_i, edge_j, n_nodes=nodes)
    values = np.asarray(statistics, dtype=float)
    if values.ndim != 1 or len(values) != len(left):
        raise ValueError("statistics must contain one value per edge")
    if not np.isfinite(values).all():
        raise ValueError("statistics contain non-finite values")
    cutoff = _validated_thresholds((threshold,))[0]
    if component_statistic not in COMPONENT_STATISTICS:
        raise ValueError(f"component_statistic must be one of {COMPONENT_STATISTICS}")
    if component_sign_mode not in COMPONENT_SIGN_MODES:
        raise ValueError(f"component_sign_mode must be one of {COMPONENT_SIGN_MODES}")
    if component_sign_mode == "pooled":
        return {
            "pooled": _tail_components(
                values,
                left,
                right,
                n_nodes=nodes,
                selected=np.abs(values) > cutoff,
                direction="pooled",
                component_statistic=component_statistic,
            )
        }
    return {
        "positive": _tail_components(
            values,
            left,
            right,
            n_nodes=nodes,
            selected=values > cutoff,
            direction="positive",
            component_statistic=component_statistic,
        ),
        "negative": _tail_components(
            values,
            left,
            right,
            n_nodes=nodes,
            selected=values < -cutoff,
            direction="negative",
            component_statistic=component_statistic,
        ),
    }


def _maximum_component(components: Sequence[NBSComponent]) -> float:
    return max((component.statistic_value for component in components), default=0.0)


@dataclass(frozen=True)
class NBSThresholdResult:
    """Observed components and maximum-component null for one threshold."""

    threshold: float
    observed_t: NDArray[np.float64]
    observed_mean_difference: NDArray[np.float64]
    positive_components: tuple[NBSComponent, ...]
    negative_components: tuple[NBSComponent, ...]
    pooled_components: tuple[NBSComponent, ...]
    null_positive: NDArray[np.float64] | None
    null_negative: NDArray[np.float64] | None
    null_pooled: NDArray[np.float64] | None
    null_maximum: NDArray[np.float64]
    alternative: str
    component_statistic: str
    component_sign_mode: str
    fwe_scope: str

    @property
    def components(self) -> tuple[NBSComponent, ...]:
        if self.component_sign_mode == "pooled":
            return self.pooled_components
        return self.positive_components + self.negative_components


@dataclass(frozen=True)
class PairedNBSResult:
    """Participant-sign-flip NBS over fixed thresholds."""

    threshold_results: tuple[NBSThresholdResult, ...]
    subject_ids: tuple[str, ...]
    edge_i: NDArray[np.int64]
    edge_j: NDArray[np.int64]
    n_nodes: int
    alternative: str
    component_statistic: str
    component_sign_mode: str
    difference_direction: str
    confound_names: tuple[str, ...]
    n_permutations: int
    seed: int
    permutation_unit: str
    permutation_method: str
    threshold_correction: str

    def at_threshold(self, threshold: float) -> NBSThresholdResult:
        for result in self.threshold_results:
            if result.threshold == float(threshold):
                return result
        available = tuple(result.threshold for result in self.threshold_results)
        raise KeyError(f"threshold {threshold} was not fitted; available={available}")


def _component_pvalue(component: NBSComponent, null: NDArray[np.float64]) -> float:
    return float((1.0 + np.count_nonzero(null >= component.statistic_value)) / (len(null) + 1.0))


def paired_nbs(
    differences: ArrayLike,
    subject_ids: Sequence[str],
    edge_i: ArrayLike,
    edge_j: ArrayLike,
    n_nodes: int,
    *,
    thresholds: Iterable[float],
    n_permutations: int,
    seed: int,
    difference_direction: str,
    alternative: str = "two-sided",
    component_statistic: str = "edge_extent",
    component_sign_mode: str = "separate",
    confounds: ArrayLike | None = None,
    confound_names: Sequence[str] = (),
) -> PairedNBSResult:
    """Run NBS on one edge vector of paired differences per participant.

    With confounds, residuals from a centered nuisance-only reduced model are
    sign-flipped and added back to its fitted values. Without confounds, each
    participant's complete edge vector is sign-flipped directly.

    ``component_sign_mode="separate"`` constructs positive and negative
    components independently. ``"pooled"`` thresholds the absolute t
    statistic and permits opposite-sign edges to join the same component.
    """
    nodes = validated_integer(n_nodes, label="n_nodes", minimum=2)
    left, right = _validated_edge_data(edge_i, edge_j, n_nodes=nodes)
    values = _validated_differences(differences, n_edges=len(left))
    identifiers = validated_subject_ids(subject_ids, len(values))
    cutoffs = _validated_thresholds(thresholds)
    permutations = validated_integer(
        n_permutations, label="n_permutations", minimum=1
    )
    permutation_seed = validated_integer(seed, label="seed", minimum=0)
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}")
    if component_statistic not in COMPONENT_STATISTICS:
        raise ValueError(f"component_statistic must be one of {COMPONENT_STATISTICS}")
    if component_sign_mode not in COMPONENT_SIGN_MODES:
        raise ValueError(f"component_sign_mode must be one of {COMPONENT_SIGN_MODES}")
    if component_sign_mode == "pooled" and alternative != "two-sided":
        raise ValueError(
            'component_sign_mode="pooled" requires alternative="two-sided"'
        )
    if not str(difference_direction).strip():
        raise ValueError("difference_direction must describe the paired subtraction")

    names = tuple(str(name) for name in confound_names)
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("confound_names must be unique and non-empty")
    if confounds is None and names:
        raise ValueError("confound_names were supplied without confounds")
    if confounds is not None:
        raw_confounds = np.asarray(confounds)
        if raw_confounds.ndim not in {1, 2}:
            raise ValueError("confounds must be one- or two-dimensional")
        n_confounds = 1 if raw_confounds.ndim == 1 else raw_confounds.shape[1]
        if len(names) != n_confounds:
            raise ValueError("confound_names must identify every confound column")

    design, centered, inverse, degrees = _prepare_design(len(values), confounds)
    observed_t = _intercept_t_with_design(values, design, inverse, degrees)
    observed_mean = values.mean(axis=0)
    observed = {
        cutoff: _threshold_components(
            observed_t,
            left,
            right,
            nodes,
            cutoff,
            component_statistic=component_statistic,
            component_sign_mode=component_sign_mode,
        )
        for cutoff in cutoffs
    }

    if centered is None:
        fitted = np.zeros_like(values)
        residual = values.copy()
    else:
        nuisance_coefficients = np.linalg.lstsq(centered, values, rcond=None)[0]
        fitted = centered @ nuisance_coefficients
        residual = values - fitted

    null_positive = (
        {cutoff: np.zeros(permutations, dtype=float) for cutoff in cutoffs}
        if component_sign_mode == "separate"
        else None
    )
    null_negative = (
        {cutoff: np.zeros(permutations, dtype=float) for cutoff in cutoffs}
        if component_sign_mode == "separate"
        else None
    )
    null_pooled = (
        {cutoff: np.zeros(permutations, dtype=float) for cutoff in cutoffs}
        if component_sign_mode == "pooled"
        else None
    )
    rng = np.random.default_rng(permutation_seed)
    for permutation in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=(len(values), 1))
        permuted = fitted + residual * signs
        permuted_t = _intercept_t_with_design(permuted, design, inverse, degrees)
        for cutoff in cutoffs:
            components = _threshold_components(
                permuted_t,
                left,
                right,
                nodes,
                cutoff,
                component_statistic=component_statistic,
                component_sign_mode=component_sign_mode,
            )
            if component_sign_mode == "pooled":
                assert null_pooled is not None
                null_pooled[cutoff][permutation] = _maximum_component(
                    components["pooled"]
                )
            else:
                assert null_positive is not None and null_negative is not None
                null_positive[cutoff][permutation] = _maximum_component(
                    components["positive"]
                )
                null_negative[cutoff][permutation] = _maximum_component(
                    components["negative"]
                )

    threshold_results = []
    for cutoff in cutoffs:
        if component_sign_mode == "pooled":
            assert null_pooled is not None
            positive_null = None
            negative_null = None
            pooled_null = null_pooled[cutoff]
            family_null = pooled_null
            positive = ()
            negative = ()
            pooled = observed[cutoff]["pooled"]
        else:
            assert null_positive is not None and null_negative is not None
            positive_null = null_positive[cutoff]
            negative_null = null_negative[cutoff]
            pooled_null = None
            pooled = ()
        if component_sign_mode == "separate" and alternative == "two-sided":
            positive = observed[cutoff]["positive"]
            negative = observed[cutoff]["negative"]
            family_null = np.maximum(positive_null, negative_null)
        elif component_sign_mode == "separate" and alternative == "greater":
            family_null = positive_null
            positive = observed[cutoff]["positive"]
            negative = ()
        elif component_sign_mode == "separate":
            family_null = negative_null
            positive = ()
            negative = observed[cutoff]["negative"]
        positive = tuple(
            replace(component, fwe_pvalue=_component_pvalue(component, family_null))
            for component in positive
        )
        negative = tuple(
            replace(component, fwe_pvalue=_component_pvalue(component, family_null))
            for component in negative
        )
        pooled = tuple(
            replace(component, fwe_pvalue=_component_pvalue(component, family_null))
            for component in pooled
        )
        threshold_results.append(
            NBSThresholdResult(
                threshold=cutoff,
                observed_t=_readonly(observed_t),
                observed_mean_difference=_readonly(observed_mean),
                positive_components=positive,
                negative_components=negative,
                pooled_components=pooled,
                null_positive=(
                    None if positive_null is None else _readonly(positive_null)
                ),
                null_negative=(
                    None if negative_null is None else _readonly(negative_null)
                ),
                null_pooled=None if pooled_null is None else _readonly(pooled_null),
                null_maximum=_readonly(family_null),
                alternative=alternative,
                component_statistic=component_statistic,
                component_sign_mode=component_sign_mode,
                fwe_scope=(
                    "maximum absolute-thresholded pooled component within this threshold"
                    if component_sign_mode == "pooled"
                    else "maximum component across tested tails within this threshold"
                ),
            )
        )

    correction_scope = (
        "component-level FWE for the pooled absolute-threshold support "
        "within each fixed threshold"
        if component_sign_mode == "pooled"
        else "component-level FWE across tested sign-separated tails "
        "within each fixed threshold"
    )
    return PairedNBSResult(
        threshold_results=tuple(threshold_results),
        subject_ids=identifiers,
        edge_i=_readonly(left),
        edge_j=_readonly(right),
        n_nodes=nodes,
        alternative=alternative,
        component_statistic=component_statistic,
        component_sign_mode=component_sign_mode,
        difference_direction=str(difference_direction),
        confound_names=names,
        n_permutations=permutations,
        seed=permutation_seed,
        permutation_unit="participant complete edge vector",
        permutation_method=(
            "direct participant sign flip"
            if centered is None
            else "reduced-model residual sign flip with centered confounds"
        ),
        threshold_correction=(
            f"{correction_scope}; no correction or independent-replication claim "
            "across thresholds"
        ),
    )
