"""Method-specific writers for :class:`dfckit.storage.FeatureStore`."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..connectivity._edge_products import edge_products
from ..connectivity.correlation import edge_index, weighted_correlation
from ..connectivity.instantaneous import _InstantaneousEstimator, _InstantaneousRows
from ..connectivity.leida import LEiDA
from ..connectivity.windows import SlidingWindowFC
from ..data import TimeSeriesDataset, TimeSeriesRun
from ..preprocessing import segment_standardized_samples
from .store import FeatureStore, _validated_chunk_size

FeatureKey = tuple[str, ...]


def _edge_feature_keys(roi_names: Sequence[str]) -> tuple[FeatureKey, ...]:
    left, right = edge_index(len(roi_names))
    return tuple((roi_names[int(i)], roi_names[int(j)]) for i, j in zip(left, right, strict=True))


def _roi_feature_keys(roi_names: Sequence[str]) -> tuple[FeatureKey, ...]:
    return tuple((str(name),) for name in roi_names)


def _window_fc_contract(estimator: SlidingWindowFC) -> str:
    return f"window-fc:length={estimator.length};step={estimator.step};taper={estimator.taper}"


def append_window_fc(
    store: FeatureStore,
    run: TimeSeriesRun,
    estimator: SlidingWindowFC,
    *,
    chunk_size: int = 128,
) -> None:
    """Compute and append sliding-window FC without materializing the full result."""
    if run.subject is None:
        raise ValueError("stored window FC requires a subject identifier")
    size = _validated_chunk_size(chunk_size)
    keys = _edge_feature_keys(run.roi_names)
    interval = None if run.tr is None else estimator.step * run.tr
    store.require_contract(
        feature_keys=keys,
        source_contract=_window_fc_contract(estimator),
        sample_interval_seconds=interval,
    )
    edge_i, edge_j = edge_index(run.n_rois)
    weights = estimator._weights()
    appended = 0
    for segment_id, positions in enumerate(run.segments()):
        n_windows = max(0, (len(positions) - estimator.length) // estimator.step + 1)
        if not n_windows:
            continue

        def parts(
            segment_positions: NDArray[np.int64] = positions,
            window_count: int = n_windows,
        ) -> Iterator[tuple[NDArray, NDArray, NDArray]]:
            for first in range(0, window_count, size):
                last = min(first + size, window_count)
                features = []
                starts = np.empty(last - first, dtype=np.int64)
                ends = np.empty(last - first, dtype=np.int64)
                for output_index, window_index in enumerate(range(first, last)):
                    start = window_index * estimator.step
                    selected = segment_positions[start : start + estimator.length]
                    correlation = weighted_correlation(run.values[selected], weights)
                    features.append(
                        np.arctanh(np.clip(correlation[edge_i, edge_j], -0.999999, 0.999999))
                    )
                    original = run.original_indices[selected]
                    starts[output_index] = original[0]
                    ends[output_index] = original[-1]
                yield np.stack(features), starts, ends

        store.append_sequence_parts(
            parts(),
            subject=run.subject,
            session=run.session,
            segment_id=segment_id,
            acquisition_id=run.acquisition_id,
        )
        appended += n_windows
    if not appended:
        raise ValueError("no contiguous segment is long enough for the requested window")


def write_window_fc_store(
    root: str | Path,
    runs: Sequence[TimeSeriesRun],
    estimator: SlidingWindowFC,
    *,
    chunk_size: int = 128,
    dtype: str | np.dtype = "float64",
) -> FeatureStore:
    """Create a store and stream sliding-window FC from compatible runs into it."""
    dataset = TimeSeriesDataset(runs)
    dataset.require_subject_ids("stored window FC")
    store = FeatureStore.create(
        root,
        feature_keys=_edge_feature_keys(dataset.roi_names),
        source_contract=_window_fc_contract(estimator),
        sample_interval_seconds=None if dataset.tr is None else estimator.step * dataset.tr,
        dtype=dtype,
    )
    for run in dataset.runs:
        append_window_fc(store, run, estimator, chunk_size=chunk_size)
    return store


def _append_instantaneous_edge_samples(
    store: FeatureStore,
    source: _InstantaneousRows,
    *,
    chunk_size: int,
) -> None:
    """Stream any standardized sample matrix through the common edge kernel."""
    if source.subject is None:
        raise ValueError("stored instantaneous-edge samples require a subject identifier")
    size = _validated_chunk_size(chunk_size)
    store.require_contract(
        feature_keys=_edge_feature_keys(source.roi_names),
        source_contract=source.contract,
        sample_interval_seconds=source.tr,
    )
    for segment_id in dict.fromkeys(source.segments.tolist()):
        positions = np.flatnonzero(source.segments == segment_id)

        def parts(
            segment_positions: NDArray[np.int64] = positions,
        ) -> Iterator[tuple[NDArray, NDArray, NDArray]]:
            for first in range(0, len(segment_positions), size):
                selected = segment_positions[first : first + size]
                yield (
                    edge_products(source.values[selected]),
                    source.starts[selected],
                    source.ends[selected],
                )

        store.append_sequence_parts(
            parts(),
            subject=source.subject,
            session=source.session,
            segment_id=int(segment_id),
            acquisition_id=source.acquisition_id,
        )


def append_instantaneous_edges(
    store: FeatureStore,
    run: TimeSeriesRun,
    estimator: _InstantaneousEstimator,
    *,
    chunk_size: int = 128,
) -> None:
    """Generate and append ETS or MTD edge samples through one writer."""
    if not isinstance(estimator, _InstantaneousEstimator):
        raise TypeError("estimator must be ETS or MTD")
    _append_instantaneous_edge_samples(
        store,
        estimator._rows(run),
        chunk_size=chunk_size,
    )


def write_instantaneous_edge_store(
    root: str | Path,
    runs: Sequence[TimeSeriesRun],
    estimator: _InstantaneousEstimator,
    *,
    chunk_size: int = 128,
    dtype: str | np.dtype = "float64",
) -> FeatureStore:
    """Create a store and stream ETS or MTD through one writer."""
    if not isinstance(estimator, _InstantaneousEstimator):
        raise TypeError("estimator must be ETS or MTD")
    dataset = TimeSeriesDataset(runs)
    dataset.require_subject_ids("stored instantaneous-edge analysis")
    first = estimator._rows(dataset.runs[0])
    store = FeatureStore.create(
        root,
        feature_keys=_edge_feature_keys(first.roi_names),
        source_contract=first.contract,
        sample_interval_seconds=first.tr,
        dtype=dtype,
    )
    _append_instantaneous_edge_samples(store, first, chunk_size=chunk_size)
    for run in dataset.runs[1:]:
        append_instantaneous_edges(store, run, estimator, chunk_size=chunk_size)
    return store


def append_cap(
    store: FeatureStore,
    run: TimeSeriesRun,
    *,
    chunk_size: int = 128,
) -> None:
    """Append segment-standardized instantaneous ROI patterns in bounded chunks."""
    if run.subject is None:
        raise ValueError("stored CAP requires a subject identifier")
    size = _validated_chunk_size(chunk_size)
    keys = _roi_feature_keys(run.roi_names)
    contract = "cap:within-segment-roi-zscore-ddof0"
    store.require_contract(
        feature_keys=keys,
        source_contract=contract,
        sample_interval_seconds=run.tr,
    )
    standardized, original, segment_ids = segment_standardized_samples(run, method_name="CAP")
    for segment_id in dict.fromkeys(segment_ids.tolist()):
        positions = np.flatnonzero(segment_ids == segment_id)

        def parts(
            segment_positions: NDArray[np.int64] = positions,
        ) -> Iterator[tuple[NDArray, NDArray, NDArray]]:
            for first in range(0, len(segment_positions), size):
                selected = segment_positions[first : first + size]
                indices = original[selected]
                yield standardized[selected], indices, indices

        store.append_sequence_parts(
            parts(),
            subject=run.subject,
            session=run.session,
            segment_id=int(segment_id),
            acquisition_id=run.acquisition_id,
        )


def write_cap_store(
    root: str | Path,
    runs: Sequence[TimeSeriesRun],
    *,
    chunk_size: int = 128,
    dtype: str | np.dtype = "float64",
) -> FeatureStore:
    """Create a store of segment-standardized instantaneous ROI patterns."""
    dataset = TimeSeriesDataset(runs)
    dataset.require_subject_ids("stored CAP")
    store = FeatureStore.create(
        root,
        feature_keys=_roi_feature_keys(dataset.roi_names),
        source_contract="cap:within-segment-roi-zscore-ddof0",
        sample_interval_seconds=dataset.tr,
        dtype=dtype,
    )
    for run in dataset.runs:
        append_cap(store, run, chunk_size=chunk_size)
    return store


def _leida_contract(estimator: LEiDA) -> str:
    return (
        "leida:hilbert=within-segment;"
        f"minimum-segment-length={estimator.minimum_segment_length};"
        "orientation=positive-vector-sum"
    )


def append_leida(
    store: FeatureStore,
    run: TimeSeriesRun,
    estimator: LEiDA,
    *,
    chunk_size: int = 128,
) -> None:
    """Append censor-bounded LEiDA leading eigenvectors in row chunks."""
    if run.subject is None:
        raise ValueError("stored LEiDA requires a subject identifier")
    size = _validated_chunk_size(chunk_size)
    keys = _roi_feature_keys(run.roi_names)
    contract = _leida_contract(estimator)
    store.require_contract(
        feature_keys=keys,
        source_contract=contract,
        sample_interval_seconds=run.tr,
    )
    result = estimator.transform(run)
    for segment_id in dict.fromkeys(result.segment_ids.tolist()):
        positions = np.flatnonzero(result.segment_ids == segment_id)

        def parts(
            segment_positions: NDArray[np.int64] = positions,
        ) -> Iterator[tuple[NDArray, NDArray, NDArray]]:
            for first in range(0, len(segment_positions), size):
                selected = segment_positions[first : first + size]
                indices = result.original_indices[selected]
                yield result.leading_vectors[selected], indices, indices

        store.append_sequence_parts(
            parts(),
            subject=run.subject,
            session=run.session,
            segment_id=int(segment_id),
            acquisition_id=run.acquisition_id,
        )


def write_leida_store(
    root: str | Path,
    runs: Sequence[TimeSeriesRun],
    estimator: LEiDA | None = None,
    *,
    chunk_size: int = 128,
    dtype: str | np.dtype = "float64",
) -> FeatureStore:
    """Create a store of censor-bounded LEiDA leading eigenvectors."""
    dataset = TimeSeriesDataset(runs)
    dataset.require_subject_ids("stored LEiDA")
    fitted_estimator = LEiDA() if estimator is None else estimator
    if not isinstance(fitted_estimator, LEiDA):
        raise TypeError("estimator must be a LEiDA instance")
    store = FeatureStore.create(
        root,
        feature_keys=_roi_feature_keys(dataset.roi_names),
        source_contract=_leida_contract(fitted_estimator),
        sample_interval_seconds=dataset.tr,
        dtype=dtype,
    )
    for run in dataset.runs:
        append_leida(store, run, fitted_estimator, chunk_size=chunk_size)
    return store
