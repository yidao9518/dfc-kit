"""Segment-normalized edge time series and co-fluctuation events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..data import TimeSeriesDataset, TimeSeriesRun
from .correlation import edge_index


def _readonly(values: NDArray) -> NDArray:
    output = np.asarray(values).copy()
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class ETSRSSResult:
    """Edge co-fluctuation RSS without a materialized frame-by-edge matrix."""

    rss: NDArray[np.float64]
    original_indices: NDArray[np.int64]
    segment_ids: NDArray[np.int64]
    roi_names: tuple[str, ...]
    subject: str | None
    session: str | None
    normalization: str = "segment"
    acquisition_id: str | None = None


@dataclass(frozen=True)
class ETSResult:
    """Materialized edge time series plus their root-sum-square amplitude."""

    features: NDArray[np.float64]
    rss: NDArray[np.float64]
    original_indices: NDArray[np.int64]
    segment_ids: NDArray[np.int64]
    edge_i: NDArray[np.int64]
    edge_j: NDArray[np.int64]
    roi_names: tuple[str, ...]
    subject: str | None
    session: str | None
    normalization: str = "segment"
    acquisition_id: str | None = None


@dataclass(frozen=True)
class ETSEventThreshold:
    """A pooled RSS percentile fitted on explicitly identified participants."""

    value: float
    percentile: float
    fit_subjects: tuple[str, ...]
    roi_names: tuple[str, ...]
    normalization: str = "segment"

    def event_mask(self, result: ETSRSSResult | ETSResult) -> NDArray[np.bool_]:
        if result.roi_names != self.roi_names:
            raise ValueError("ETS event threshold and result use different ROI identities or order")
        if result.normalization != self.normalization:
            raise ValueError("ETS event threshold and result use different normalization")
        return _readonly(np.asarray(result.rss >= self.value, dtype=bool))


def segment_standardized_samples(
    run: TimeSeriesRun,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
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
        raise ValueError("ETS requires at least one retained segment with two frames")
    standardized = np.concatenate(samples, axis=0)
    if not np.isfinite(standardized).all():
        raise ValueError("standardized ETS samples contain non-finite values")
    return (
        _readonly(standardized),
        _readonly(np.concatenate(original_indices)),
        _readonly(np.concatenate(segment_ids)),
    )


def edge_rss(standardized_samples: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute all-edge RSS without constructing the edge time series."""
    values = np.asarray(standardized_samples, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("standardized_samples must be a frames-by-ROI array")
    if not np.isfinite(values).all():
        raise ValueError("standardized_samples contain non-finite values")
    squared = values * values
    squared_rss = ((squared.sum(axis=1) ** 2 - (squared * squared).sum(axis=1)) / 2).clip(
        min=0.0
    )
    return _readonly(np.sqrt(squared_rss))


class ETS:
    """Calculate segment-normalized instantaneous edge co-fluctuations."""

    def rss(self, run: TimeSeriesRun) -> ETSRSSResult:
        """Return only co-fluctuation amplitude using the memory-efficient identity."""
        standardized, original_indices, segment_ids = segment_standardized_samples(run)
        return ETSRSSResult(
            rss=edge_rss(standardized),
            original_indices=original_indices,
            segment_ids=segment_ids,
            roi_names=run.roi_names,
            subject=run.subject,
            session=run.session,
            acquisition_id=run.acquisition_id,
        )

    def transform(self, run: TimeSeriesRun) -> ETSResult:
        """Return the complete frame-by-edge time series and RSS amplitude."""
        standardized, original_indices, segment_ids = segment_standardized_samples(run)
        left, right = edge_index(run.n_rois)
        features = standardized[:, left] * standardized[:, right]
        if not np.isfinite(features).all():
            raise ValueError("edge time series contain non-finite values")
        return ETSResult(
            features=_readonly(features),
            rss=edge_rss(standardized),
            original_indices=original_indices,
            segment_ids=segment_ids,
            edge_i=_readonly(left),
            edge_j=_readonly(right),
            roi_names=run.roi_names,
            subject=run.subject,
            session=run.session,
            acquisition_id=run.acquisition_id,
        )


def fit_ets_event_threshold(
    dataset: TimeSeriesDataset,
    percentile: float = 95.0,
) -> ETSEventThreshold:
    """Fit a pooled RSS percentile and record the participant IDs used."""
    if not np.isfinite(percentile) or not 0.0 < percentile < 100.0:
        raise ValueError("percentile must be finite and strictly between 0 and 100")
    fit_subjects = dataset.require_subject_ids("ETS event-threshold fitting")
    estimator = ETS()
    pooled = np.concatenate([estimator.rss(run).rss for run in dataset.runs])
    value = float(np.percentile(pooled, percentile))
    return ETSEventThreshold(
        value=value,
        percentile=float(percentile),
        fit_subjects=fit_subjects,
        roi_names=dataset.roi_names,
    )
