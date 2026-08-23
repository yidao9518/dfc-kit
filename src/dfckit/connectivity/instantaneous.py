"""Instantaneous edge products from ETS and MTD samples.

ETS and MTD differ only in how they construct their time-aligned ROI rows.
After that step both methods produce the same edge result, summaries, state
sequences, and chunked stores.  The row representation is deliberately
private: callers normally need the edge result, not another intermediate
object to carry through the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._arrays import readonly_copy as _readonly
from ..data import TimeSeriesRun
from ..preprocessing import segment_standardized_samples


@dataclass(frozen=True)
class _InstantaneousRows:
    values: NDArray[np.float64]
    starts: NDArray[np.int64]
    ends: NDArray[np.int64]
    segments: NDArray[np.int64]
    roi_names: tuple[str, ...]
    subject: str | None
    session: str | None
    sample_kind: str
    contract: str
    tr: float | None
    acquisition_id: str | None


class _InstantaneousEstimator:
    """Small shared implementation for the two public estimators."""

    def _rows(self, run: TimeSeriesRun) -> _InstantaneousRows:
        raise NotImplementedError

    def transform(self, run: TimeSeriesRun) -> InstantaneousEdgeResult:
        return _result_from_rows(self._rows(run), materialize_edges=True)

    def rss(self, run: TimeSeriesRun) -> InstantaneousEdgeResult:
        return _result_from_rows(self._rows(run), materialize_edges=False)


class ETS(_InstantaneousEstimator):
    """Instantaneous edge products from segment-standardized BOLD values."""

    def _rows(self, run: TimeSeriesRun) -> _InstantaneousRows:
        values, indices, segments = segment_standardized_samples(run, method_name="ETS")
        return _InstantaneousRows(
            values=values,
            starts=indices,
            ends=indices,
            segments=segments,
            roi_names=run.roi_names,
            subject=run.subject,
            session=run.session,
            sample_kind="frame",
            contract="ets:normalization=segment",
            tr=run.tr,
            acquisition_id=run.acquisition_id,
        )


def _standardized_derivatives(
    run: TimeSeriesRun,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return run-standardized derivatives without crossing censor gaps."""
    values: list[NDArray[np.float64]] = []
    starts: list[NDArray[np.int64]] = []
    ends: list[NDArray[np.int64]] = []
    segments: list[NDArray[np.int64]] = []
    for segment_id, positions in enumerate(run.segments()):
        if len(positions) < 2:
            continue
        values.append(np.diff(run.values[positions], axis=0))
        original = run.original_indices[positions]
        starts.append(original[:-1])
        ends.append(original[1:])
        segments.append(np.full(len(positions) - 1, segment_id, dtype=np.int64))
    if not values:
        raise ValueError("MTD requires at least one valid within-segment derivative")
    derivatives = np.concatenate(values, axis=0)
    scale = derivatives.std(axis=0, ddof=0)
    invalid = np.flatnonzero((~np.isfinite(scale)) | (scale < 1e-8))
    if len(invalid):
        raise ValueError(f"MTD derivative scale is undefined for ROI indices {invalid.tolist()}")
    standardized = (derivatives - derivatives.mean(axis=0)) / scale
    if not np.isfinite(standardized).all():
        raise ValueError("standardized MTD derivatives contain non-finite values")
    return standardized, np.concatenate(starts), np.concatenate(ends), np.concatenate(segments)


class MTD(_InstantaneousEstimator):
    """Instantaneous edge products from standardized within-segment derivatives."""

    def _rows(self, run: TimeSeriesRun) -> _InstantaneousRows:
        values, starts, ends, segments = _standardized_derivatives(run)
        return _InstantaneousRows(
            values=values,
            starts=starts,
            ends=ends,
            segments=segments,
            roi_names=run.roi_names,
            subject=run.subject,
            session=run.session,
            sample_kind="interval",
            contract="mtd:difference=within-segment;normalization=run",
            tr=run.tr,
            acquisition_id=run.acquisition_id,
        )


@dataclass(frozen=True)
class InstantaneousEdgeResult:
    """One frame- or interval-aligned matrix of instantaneous edge products."""

    features: NDArray[np.float64] | None
    rss: NDArray[np.float64]
    sample_start_frames: NDArray[np.int64]
    sample_end_frames: NDArray[np.int64]
    segment_ids: NDArray[np.int64]
    edge_i: NDArray[np.int64]
    edge_j: NDArray[np.int64]
    roi_names: tuple[str, ...]
    subject: str | None
    session: str | None
    sample_kind: str
    source_contract: str
    tr: float | None = None
    acquisition_id: str | None = None

    def __post_init__(self) -> None:
        from .correlation import edge_index

        features = None if self.features is None else np.asarray(self.features, dtype=float)
        rss = np.asarray(self.rss, dtype=float)
        starts = np.asarray(self.sample_start_frames, dtype=np.int64)
        ends = np.asarray(self.sample_end_frames, dtype=np.int64)
        segments = np.asarray(self.segment_ids, dtype=np.int64)
        left = np.asarray(self.edge_i, dtype=np.int64)
        right = np.asarray(self.edge_j, dtype=np.int64)
        names = tuple(str(name) for name in self.roi_names)
        if len(names) < 2 or len(set(names)) != len(names) or any(not name.strip() for name in names):
            raise ValueError("roi_names must contain at least two unique non-empty names")
        if rss.ndim != 1 or not len(rss) or not np.isfinite(rss).all():
            raise ValueError("rss must be a non-empty finite one-dimensional array")
        if any(array.ndim != 1 or len(array) != len(rss) for array in (starts, ends, segments)):
            raise ValueError("sample metadata must align with rss")
        if np.any(ends < starts) or np.any(segments < 0):
            raise ValueError("sample intervals and segment IDs must be non-negative")
        if len(starts) > 1 and np.any(np.diff(starts) <= 0):
            raise ValueError("sample start frames must be strictly increasing")
        expected_left, expected_right = edge_index(len(names))
        if not np.array_equal(left, expected_left) or not np.array_equal(right, expected_right):
            raise ValueError("edge endpoints must be the unique upper-triangular ROI order")
        if features is not None:
            if features.ndim != 2 or features.shape != (len(rss), len(left)):
                raise ValueError("features must align with samples and edge endpoints")
            if not np.isfinite(features).all():
                raise ValueError("edge features contain non-finite samples")
        if self.sample_kind not in {"frame", "interval"}:
            raise ValueError("sample_kind must be 'frame' or 'interval'")
        if self.sample_kind == "frame" and np.any(starts != ends):
            raise ValueError("frame samples must have equal start and end frames")
        if not str(self.source_contract).strip():
            raise ValueError("source_contract must be non-empty")
        if self.subject is not None and not str(self.subject).strip():
            raise ValueError("subject cannot be empty")
        if self.session is not None and not str(self.session).strip():
            raise ValueError("session cannot be empty")
        if self.tr is not None and (not np.isfinite(self.tr) or self.tr <= 0):
            raise ValueError("tr must be finite and positive")
        if self.acquisition_id is not None and not str(self.acquisition_id).strip():
            raise ValueError("acquisition_id cannot be empty")
        object.__setattr__(self, "features", None if features is None else _readonly(features))
        object.__setattr__(self, "rss", _readonly(rss))
        object.__setattr__(self, "sample_start_frames", _readonly(starts))
        object.__setattr__(self, "sample_end_frames", _readonly(ends))
        object.__setattr__(self, "segment_ids", _readonly(segments))
        object.__setattr__(self, "edge_i", _readonly(left))
        object.__setattr__(self, "edge_j", _readonly(right))
        object.__setattr__(self, "roi_names", names)
        object.__setattr__(self, "subject", None if self.subject is None else str(self.subject))
        object.__setattr__(self, "session", None if self.session is None else str(self.session))
        object.__setattr__(self, "source_contract", str(self.source_contract))
        object.__setattr__(self, "acquisition_id", None if self.acquisition_id is None else str(self.acquisition_id))

    @property
    def n_samples(self) -> int:
        return len(self.rss)

    @property
    def n_edges(self) -> int:
        return len(self.edge_i)

    @property
    def n_rois(self) -> int:
        return len(self.roi_names)

    @property
    def feature_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (self.roi_names[int(left)], self.roi_names[int(right)])
            for left, right in zip(self.edge_i, self.edge_j, strict=True)
        )

    def require_features(self) -> NDArray[np.float64]:
        if self.features is None:
            raise ValueError("this result contains RSS only and no materialized edge features")
        return self.features

    def cross_block(self, left: Sequence[int], right: Sequence[int]) -> NDArray[np.float64]:
        from ._edge_products import cross_block_edge_summary

        return cross_block_edge_summary(self, left, right)

    def within_block(self, nodes: Sequence[int]) -> NDArray[np.float64]:
        from ._edge_products import within_block_edge_summary

        return within_block_edge_summary(self, nodes)


def _result_from_rows(rows: _InstantaneousRows, *, materialize_edges: bool) -> InstantaneousEdgeResult:
    from ._edge_products import edge_products, edge_rss
    from .correlation import edge_index

    left, right = edge_index(len(rows.roi_names))
    return InstantaneousEdgeResult(
        features=edge_products(rows.values) if materialize_edges else None,
        rss=edge_rss(rows.values),
        sample_start_frames=rows.starts,
        sample_end_frames=rows.ends,
        segment_ids=rows.segments,
        edge_i=left,
        edge_j=right,
        roi_names=rows.roi_names,
        subject=rows.subject,
        session=rows.session,
        sample_kind=rows.sample_kind,
        source_contract=rows.contract,
        tr=rows.tr,
        acquisition_id=rows.acquisition_id,
    )


__all__ = ["ETS", "MTD", "InstantaneousEdgeResult"]
