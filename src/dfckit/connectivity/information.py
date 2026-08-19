"""Fixed-length, censor-bounded mutual-information estimators."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..data import TimeSeriesRun

DEFAULT_JITTER_SEED = 20260811


def _readonly(values: NDArray) -> NDArray:
    output = np.asarray(values).copy()
    output.setflags(write=False)
    return output


def _validated_integer(value: int, *, label: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return int(value)


def _validated_seed(value: int, *, label: str) -> int:
    return _validated_integer(value, label=label, minimum=0)


def _validated_metric(metric: str) -> str:
    if metric != "chebyshev":
        raise ValueError("metric must be 'chebyshev' for the Kraskov/Frenzel-Pompe estimator")
    return metric


def _validated_jitter(jitter: float) -> float:
    value = float(jitter)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("jitter must be finite and non-negative")
    return value


def _validated_scalar(values: ArrayLike, *, label: str) -> NDArray[np.float64]:
    output = np.asarray(values, dtype=float)
    if output.ndim != 1 or not len(output):
        raise ValueError(f"{label} must be a non-empty one-dimensional series")
    if not np.isfinite(output).all():
        raise ValueError(f"{label} contains non-finite samples")
    if output.std(ddof=0) <= 1e-12:
        raise ValueError(f"{label} is constant, so information estimation is undefined")
    return output


def _information_dependencies():
    try:
        import scipy
        from scipy.spatial import cKDTree
        from scipy.special import digamma
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MI/CMI estimation requires the 'information' extra: "
            "pip install 'dfc-kit[information]'"
        ) from error
    return scipy, cKDTree, digamma


def _jitter_columns(
    values: NDArray[np.float64],
    *,
    magnitude: float,
    seed: int,
) -> NDArray[np.float64]:
    if magnitude == 0.0:
        return values.copy()
    scale = np.maximum(values.std(axis=0, ddof=0, keepdims=True), 1.0)
    rng = np.random.default_rng(seed)
    return values + rng.normal(scale=magnitude * scale, size=values.shape)


def zscore_columns(values: ArrayLike) -> NDArray[np.float64]:
    """Center and population-SD scale every column independently."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1:
        raise ValueError("values must contain at least two frames and one column")
    if not np.isfinite(data).all():
        raise ValueError("values contain non-finite samples")
    scale = data.std(axis=0, ddof=0)
    invalid = np.flatnonzero(scale <= 1e-12)
    if len(invalid):
        raise ValueError(
            f"information estimation is undefined for constant column indices {invalid.tolist()}"
        )
    return _readonly((data - data.mean(axis=0)) / scale)


def knn_mi(
    x: ArrayLike,
    y: ArrayLike,
    *,
    k: int = 3,
    metric: str = "chebyshev",
    jitter: float = 1e-10,
    seed: int = DEFAULT_JITTER_SEED,
) -> float:
    """Estimate symmetric scalar mutual information with the Kraskov kNN estimator.

    The maximum/Chebyshev norm is part of this estimator's joint-versus-marginal
    neighbourhood-counting definition. Small seeded jitter resolves exact ties.
    Estimates are not clipped at zero because finite-sample kNN estimates may be
    slightly negative.
    """
    neighbors = _validated_integer(k, label="k", minimum=1)
    _validated_metric(metric)
    noise = _validated_jitter(jitter)
    jitter_seed = _validated_seed(seed, label="seed")
    left = _validated_scalar(x, label="x")
    right = _validated_scalar(y, label="y")
    if len(left) != len(right):
        raise ValueError("x and y must contain the same number of samples")
    if len(left) <= neighbors + 2:
        raise ValueError("MI requires more than k + 2 samples")

    _, tree_type, digamma = _information_dependencies()
    joint_values = _jitter_columns(
        np.column_stack((left, right)), magnitude=noise, seed=jitter_seed
    )
    joint = tree_type(joint_values)
    distance, _ = joint.query(joint_values, k=neighbors + 1, p=np.inf)
    radius = np.nextafter(distance[:, -1], 0.0)
    left_counts = (
        tree_type(joint_values[:, [0]]).query_ball_point(
            joint_values[:, [0]], radius, p=np.inf, return_length=True
        )
        - 1
    )
    right_counts = (
        tree_type(joint_values[:, [1]]).query_ball_point(
            joint_values[:, [1]], radius, p=np.inf, return_length=True
        )
        - 1
    )
    estimate = digamma(neighbors) + digamma(len(joint_values)) - np.mean(
        digamma(left_counts + 1) + digamma(right_counts + 1)
    )
    if not np.isfinite(estimate):
        raise ValueError("MI estimate is non-finite")
    return float(estimate)


def knn_cmi(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    k: int = 3,
    metric: str = "chebyshev",
    jitter: float = 1e-10,
    seed: int = DEFAULT_JITTER_SEED,
) -> float:
    """Estimate scalar ``I(X; Y | Z)`` with the Frenzel-Pompe kNN estimator."""
    neighbors = _validated_integer(k, label="k", minimum=1)
    _validated_metric(metric)
    noise = _validated_jitter(jitter)
    jitter_seed = _validated_seed(seed, label="seed")
    left = _validated_scalar(x, label="x")
    right = _validated_scalar(y, label="y")
    condition = _validated_scalar(z, label="z")
    if len({len(left), len(right), len(condition)}) != 1:
        raise ValueError("x, y, and z must contain the same number of samples")
    if len(left) <= neighbors + 2:
        raise ValueError("CMI requires more than k + 2 samples")

    _, tree_type, digamma = _information_dependencies()
    joint_values = _jitter_columns(
        np.column_stack((left, right, condition)), magnitude=noise, seed=jitter_seed
    )
    joint = tree_type(joint_values)
    distance, _ = joint.query(joint_values, k=neighbors + 1, p=np.inf)
    radius = np.nextafter(distance[:, -1], 0.0)
    condition_counts = (
        tree_type(joint_values[:, [2]]).query_ball_point(
            joint_values[:, [2]], radius, p=np.inf, return_length=True
        )
        - 1
    )
    left_condition_counts = (
        tree_type(joint_values[:, [0, 2]]).query_ball_point(
            joint_values[:, [0, 2]], radius, p=np.inf, return_length=True
        )
        - 1
    )
    right_condition_counts = (
        tree_type(joint_values[:, [1, 2]]).query_ball_point(
            joint_values[:, [1, 2]], radius, p=np.inf, return_length=True
        )
        - 1
    )
    estimate = digamma(neighbors) + np.mean(
        digamma(condition_counts + 1)
        - digamma(left_condition_counts + 1)
        - digamma(right_condition_counts + 1)
    )
    if not np.isfinite(estimate):
        raise ValueError("CMI estimate is non-finite")
    return float(estimate)


def _validated_nodes(
    nodes: Iterable[int],
    *,
    n_rois: int,
    label: str,
) -> NDArray[np.int64]:
    raw = tuple(nodes)
    if not raw:
        raise ValueError(f"{label} must contain at least one ROI index")
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


@dataclass(frozen=True)
class BlockInformationResult:
    """Pairwise information estimates and their block means for one window."""

    mutual_information: NDArray[np.float64]
    conditional_mutual_information: NDArray[np.float64] | None
    mean_mutual_information: float
    mean_conditional_mutual_information: float | None
    left_indices: NDArray[np.int64]
    right_indices: NDArray[np.int64]
    conditioning_indices: NDArray[np.int64] | None
    standardized: bool
    k: int
    metric: str
    jitter: float
    jitter_seed: int


def block_information(
    values: ArrayLike,
    left: Iterable[int],
    right: Iterable[int],
    *,
    conditioning: Iterable[int] | None = None,
    standardize: bool = True,
    k: int = 3,
    metric: str = "chebyshev",
    jitter: float = 1e-10,
    seed: int = DEFAULT_JITTER_SEED,
) -> BlockInformationResult:
    """Estimate every left-by-right MI and optional CMI, then average the block.

    When multiple conditioning ROIs are supplied, their independently standardized
    series are averaged into one scalar condition. This preserves the scalar
    Frenzel-Pompe estimator and makes the aggregation rule explicit.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("values must have at least two frames and two ROIs")
    if not np.isfinite(data).all():
        raise ValueError("values contain non-finite samples")
    left_nodes = _validated_nodes(left, n_rois=data.shape[1], label="left block")
    right_nodes = _validated_nodes(right, n_rois=data.shape[1], label="right block")
    overlap = sorted(set(left_nodes.tolist()).intersection(right_nodes.tolist()))
    if overlap:
        raise ValueError(f"left and right blocks must be disjoint; overlap={overlap}")

    condition_nodes = None
    if conditioning is not None:
        condition_nodes = _validated_nodes(
            conditioning, n_rois=data.shape[1], label="conditioning block"
        )
        blocked = set(left_nodes.tolist()).union(right_nodes.tolist())
        condition_overlap = sorted(blocked.intersection(condition_nodes.tolist()))
        if condition_overlap:
            raise ValueError(
                "conditioning block must be disjoint from left and right blocks; "
                f"overlap={condition_overlap}"
            )

    transformed = zscore_columns(data) if standardize else data
    condition_signal = (
        None if condition_nodes is None else transformed[:, condition_nodes].mean(axis=1)
    )
    mi = np.empty((len(left_nodes), len(right_nodes)), dtype=float)
    cmi = None if condition_signal is None else np.empty_like(mi)
    for left_index, left_node in enumerate(left_nodes):
        for right_index, right_node in enumerate(right_nodes):
            mi[left_index, right_index] = knn_mi(
                transformed[:, left_node],
                transformed[:, right_node],
                k=k,
                metric=metric,
                jitter=jitter,
                seed=seed,
            )
            if cmi is not None and condition_signal is not None:
                cmi[left_index, right_index] = knn_cmi(
                    transformed[:, left_node],
                    transformed[:, right_node],
                    condition_signal,
                    k=k,
                    metric=metric,
                    jitter=jitter,
                    seed=seed,
                )

    return BlockInformationResult(
        mutual_information=_readonly(mi),
        conditional_mutual_information=None if cmi is None else _readonly(cmi),
        mean_mutual_information=float(mi.mean()),
        mean_conditional_mutual_information=None if cmi is None else float(cmi.mean()),
        left_indices=_readonly(left_nodes),
        right_indices=_readonly(right_nodes),
        conditioning_indices=None if condition_nodes is None else _readonly(condition_nodes),
        standardized=bool(standardize),
        k=_validated_integer(k, label="k", minimum=1),
        metric=_validated_metric(metric),
        jitter=_validated_jitter(jitter),
        jitter_seed=_validated_seed(seed, label="seed"),
    )


@dataclass(frozen=True)
class FixedWindowSamples:
    """Equal-length windows sampled uniformly from all censor-safe starts."""

    values: NDArray[np.float64]
    original_indices: NDArray[np.int64]
    draw_indices: NDArray[np.int64]
    segment_ids: NDArray[np.int64]
    starts_within_segment: NDArray[np.int64]
    start_frames: NDArray[np.int64]
    end_frames: NDArray[np.int64]
    eligible_starts: int
    length: int
    requested_draws: int
    seed: int
    subject: str | None
    session: str | None
    acquisition_id: str | None = None


def eligible_fixed_window_count(run: TimeSeriesRun, length: int) -> int:
    """Count every fixed-length start wholly contained in a retained segment."""
    if not isinstance(run, TimeSeriesRun):
        raise TypeError("run must be a TimeSeriesRun")
    window_length = _validated_integer(length, label="length", minimum=1)
    return int(sum(max(0, len(segment) - window_length + 1) for segment in run.segments()))


def sample_fixed_windows(
    run: TimeSeriesRun,
    length: int,
    draws: int,
    *,
    seed: int,
) -> FixedWindowSamples:
    """Sample with replacement uniformly over all censor-safe window starts."""
    if not isinstance(run, TimeSeriesRun):
        raise TypeError("run must be a TimeSeriesRun")
    window_length = _validated_integer(length, label="length", minimum=1)
    n_draws = _validated_integer(draws, label="draws", minimum=1)
    sampling_seed = _validated_seed(seed, label="seed")
    segments = run.segments()
    counts = np.asarray(
        [max(0, len(segment) - window_length + 1) for segment in segments],
        dtype=np.int64,
    )
    total = int(counts.sum())
    if total == 0:
        raise ValueError("no contiguous retained segment is long enough for the requested length")

    cumulative = np.cumsum(counts)
    flat_starts = np.random.default_rng(sampling_seed).integers(0, total, size=n_draws)
    values = np.empty((n_draws, window_length, run.n_rois), dtype=float)
    original = np.empty((n_draws, window_length), dtype=np.int64)
    segment_ids = np.empty(n_draws, dtype=np.int64)
    starts = np.empty(n_draws, dtype=np.int64)
    for draw, flat_start in enumerate(flat_starts):
        segment_id = int(np.searchsorted(cumulative, flat_start, side="right"))
        previous = int(cumulative[segment_id - 1]) if segment_id else 0
        start = int(flat_start - previous)
        positions = segments[segment_id][start : start + window_length]
        if len(positions) != window_length:  # pragma: no cover - guards indexing invariant
            raise RuntimeError("internal fixed-window indexing error")
        values[draw] = run.values[positions]
        original[draw] = run.original_indices[positions]
        segment_ids[draw] = segment_id
        starts[draw] = start

    return FixedWindowSamples(
        values=_readonly(values),
        original_indices=_readonly(original),
        draw_indices=_readonly(np.arange(n_draws, dtype=np.int64)),
        segment_ids=_readonly(segment_ids),
        starts_within_segment=_readonly(starts),
        start_frames=_readonly(original[:, 0]),
        end_frames=_readonly(original[:, -1]),
        eligible_starts=total,
        length=window_length,
        requested_draws=n_draws,
        seed=sampling_seed,
        subject=run.subject,
        session=run.session,
        acquisition_id=run.acquisition_id,
    )


@dataclass(frozen=True)
class FixedLengthInformationResult:
    """Window-level block MI/CMI with complete sampling and estimator metadata."""

    samples: FixedWindowSamples
    mutual_information: NDArray[np.float64]
    conditional_mutual_information: NDArray[np.float64] | None
    mean_mutual_information: NDArray[np.float64]
    mean_conditional_mutual_information: NDArray[np.float64] | None
    left_indices: NDArray[np.int64]
    right_indices: NDArray[np.int64]
    conditioning_indices: NDArray[np.int64] | None
    standardized: bool
    k: int
    metric: str
    jitter: float
    jitter_seed: int
    implementation: str


class FixedLengthInformation:
    """Estimate block MI/CMI on equally long, censor-safe Monte Carlo windows."""

    def __init__(
        self,
        length: int,
        draws: int,
        *,
        sample_seed: int,
        k: int = 3,
        metric: str = "chebyshev",
        jitter: float = 1e-10,
        jitter_seed: int = DEFAULT_JITTER_SEED,
        standardize: bool = True,
        jobs: int = 1,
    ) -> None:
        self.length = _validated_integer(length, label="length", minimum=1)
        self.draws = _validated_integer(draws, label="draws", minimum=1)
        self.sample_seed = _validated_seed(sample_seed, label="sample_seed")
        self.k = _validated_integer(k, label="k", minimum=1)
        if self.length <= self.k + 2:
            raise ValueError("length must be greater than k + 2")
        self.metric = _validated_metric(metric)
        self.jitter = _validated_jitter(jitter)
        self.jitter_seed = _validated_seed(jitter_seed, label="jitter_seed")
        self.standardize = bool(standardize)
        self.jobs = _validated_integer(jobs, label="jobs", minimum=1)

    def transform(
        self,
        run: TimeSeriesRun,
        left: Iterable[int],
        right: Iterable[int],
        *,
        conditioning: Iterable[int] | None = None,
    ) -> FixedLengthInformationResult:
        left_nodes = tuple(left)
        right_nodes = tuple(right)
        condition_nodes = None if conditioning is None else tuple(conditioning)
        samples = sample_fixed_windows(
            run,
            self.length,
            self.draws,
            seed=self.sample_seed,
        )

        def estimate(window: NDArray[np.float64]) -> BlockInformationResult:
            return block_information(
                window,
                left_nodes,
                right_nodes,
                conditioning=condition_nodes,
                standardize=self.standardize,
                k=self.k,
                metric=self.metric,
                jitter=self.jitter,
                seed=self.jitter_seed,
            )

        if self.jobs == 1:
            estimates = [estimate(window) for window in samples.values]
        else:
            with ThreadPoolExecutor(max_workers=self.jobs) as executor:
                estimates = list(executor.map(estimate, samples.values))
        first = estimates[0]
        mi = np.stack([estimate.mutual_information for estimate in estimates])
        cmi = (
            None
            if first.conditional_mutual_information is None
            else np.stack(
                [
                    estimate.conditional_mutual_information
                    for estimate in estimates
                    if estimate.conditional_mutual_information is not None
                ]
            )
        )
        mean_cmi = (
            None
            if first.mean_conditional_mutual_information is None
            else np.asarray(
                [estimate.mean_conditional_mutual_information for estimate in estimates],
                dtype=float,
            )
        )
        scipy, _, _ = _information_dependencies()
        return FixedLengthInformationResult(
            samples=samples,
            mutual_information=_readonly(mi),
            conditional_mutual_information=None if cmi is None else _readonly(cmi),
            mean_mutual_information=_readonly(
                np.asarray([estimate.mean_mutual_information for estimate in estimates])
            ),
            mean_conditional_mutual_information=(
                None if mean_cmi is None else _readonly(mean_cmi)
            ),
            left_indices=first.left_indices,
            right_indices=first.right_indices,
            conditioning_indices=first.conditioning_indices,
            standardized=self.standardize,
            k=self.k,
            metric=self.metric,
            jitter=self.jitter,
            jitter_seed=self.jitter_seed,
            implementation=(
                f"scipy {scipy.__version__} cKDTree; "
                "Kraskov MI and Frenzel-Pompe CMI"
            ),
        )
