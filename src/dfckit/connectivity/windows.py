"""Censor-bounded sliding-window functional connectivity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._arrays import readonly_copy as _readonly
from ..data import TimeSeriesDataset, TimeSeriesRun
from .correlation import edge_index, weighted_correlation


@dataclass(frozen=True)
class WindowFCResult:
    features: NDArray[np.float64]
    start_frames: NDArray[np.int64]
    end_frames: NDArray[np.int64]
    segment_ids: NDArray[np.int64]
    edge_i: NDArray[np.int64]
    edge_j: NDArray[np.int64]
    roi_names: tuple[str, ...]
    subject: str | None
    session: str | None
    tr: float | None
    length: int
    step: int
    taper: str
    acquisition_id: str | None = None


@dataclass(frozen=True)
class AdjacentWindowSimilarityResult:
    """Spatial correlations for consecutive windows within retained segments."""

    similarities: NDArray[np.float64]
    left_windows: NDArray[np.int64]
    right_windows: NDArray[np.int64]


def periodic_hamming(length: int) -> NDArray[np.float64]:
    positions = np.arange(length, dtype=float)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * positions / length)


class SlidingWindowFC:
    """Weighted Pearson FC calculated inside censor-bounded windows."""

    def __init__(self, length: int, step: int, taper: str = "hamming") -> None:
        if length < 2:
            raise ValueError("length must be at least two")
        if step < 1:
            raise ValueError("step must be at least one")
        if taper not in {"hamming", "uniform"}:
            raise ValueError("taper must be 'hamming' or 'uniform'")
        self.length = int(length)
        self.step = int(step)
        self.taper = taper

    def _weights(self) -> NDArray[np.float64]:
        if self.taper == "hamming":
            return periodic_hamming(self.length)
        return np.ones(self.length, dtype=float)

    def transform(self, run: TimeSeriesRun) -> WindowFCResult:
        windows = run.windows(self.length, self.step)
        if not windows:
            raise ValueError("no contiguous segment is long enough for the requested window")
        weights = self._weights()
        left, right = edge_index(run.n_rois)
        features = []
        for window in windows:
            correlation = weighted_correlation(window.values, weights)
            features.append(
                np.arctanh(np.clip(correlation[left, right], -0.999999, 0.999999))
            )
        return WindowFCResult(
            features=np.stack(features),
            start_frames=np.asarray([window.start_frame for window in windows], dtype=np.int64),
            end_frames=np.asarray([window.end_frame for window in windows], dtype=np.int64),
            segment_ids=np.asarray([window.segment_id for window in windows], dtype=np.int64),
            edge_i=left,
            edge_j=right,
            roi_names=run.roi_names,
            subject=run.subject,
            session=run.session,
            acquisition_id=run.acquisition_id,
            tr=run.tr,
            length=self.length,
            step=self.step,
            taper=self.taper,
        )


def adjacent_window_pattern_similarity(
    result: WindowFCResult,
) -> AdjacentWindowSimilarityResult:
    """Correlate consecutive FC edge patterns without crossing censor gaps."""
    if not isinstance(result, WindowFCResult):
        raise TypeError("result must be a WindowFCResult")
    features = np.asarray(result.features, dtype=float)
    segments = np.asarray(result.segment_ids)
    if features.ndim != 2 or not len(features) or not features.shape[1]:
        raise ValueError("result features must be a non-empty windows-by-edges array")
    if not np.isfinite(features).all():
        raise ValueError("result features contain non-finite values")
    if segments.ndim != 1 or len(segments) != len(features):
        raise ValueError("result segment_ids must identify every window")

    left_windows = np.flatnonzero(segments[1:] == segments[:-1]).astype(np.int64)
    right_windows = left_windows + 1
    if not len(left_windows):
        return AdjacentWindowSimilarityResult(
            similarities=_readonly(np.empty(0, dtype=float)),
            left_windows=_readonly(left_windows),
            right_windows=_readonly(right_windows),
        )

    left = features[left_windows]
    right = features[right_windows]
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 1e-12):
        raise ValueError(
            "adjacent-window Pearson similarity is undefined for a constant pattern"
        )
    similarities = np.sum(left * right, axis=1) / denominator
    return AdjacentWindowSimilarityResult(
        similarities=_readonly(similarities),
        left_windows=_readonly(left_windows),
        right_windows=_readonly(right_windows),
    )


def all_pair_window_pattern_similarity(result: WindowFCResult) -> float:
    """Average every within-segment FC-pattern pair with ``m - 1`` weights.

    Each retained segment first contributes the mean Pearson similarity across
    all of its unordered window pairs. Segment means are then weighted by the
    number of adjacent pairs, ``m - 1``, so the result is directly comparable
    with the mean returned by :func:`adjacent_window_pattern_similarity`.
    Window order is not used.
    """
    if not isinstance(result, WindowFCResult):
        raise TypeError("result must be a WindowFCResult")
    features = np.asarray(result.features, dtype=float)
    segments = np.asarray(result.segment_ids)
    if features.ndim != 2 or not len(features) or not features.shape[1]:
        raise ValueError("result features must be a non-empty windows-by-edges array")
    if not np.isfinite(features).all():
        raise ValueError("result features contain non-finite values")
    if segments.ndim != 1 or len(segments) != len(features):
        raise ValueError("result segment_ids must identify every window")

    weighted_total = 0.0
    total_weight = 0
    for segment in np.unique(segments):
        patterns = features[segments == segment]
        n_windows = len(patterns)
        if n_windows < 2:
            continue
        centered = patterns - patterns.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
            raise ValueError(
                "all-pair window Pearson similarity is undefined for a constant pattern"
            )
        normalized = centered / norms[:, None]
        upper = np.triu_indices(n_windows, k=1)
        segment_mean = float(np.mean((normalized @ normalized.T)[upper]))
        weight = n_windows - 1
        weighted_total += weight * segment_mean
        total_weight += weight
    if total_weight == 0:
        raise ValueError("at least one retained segment must contain two windows")
    return float(weighted_total / total_weight)


def window_pattern_adjacency_excess(result: WindowFCResult) -> float:
    """Return adjacent-pattern similarity beyond the all-pair expectation."""
    adjacent = adjacent_window_pattern_similarity(result).similarities
    if not len(adjacent):
        raise ValueError("at least one retained segment must contain two windows")
    return float(np.mean(adjacent) - all_pair_window_pattern_similarity(result))


def summarize_window_pattern_dataset(
    dataset: TimeSeriesDataset,
    estimator: SlidingWindowFC,
) -> dict[str, Any]:
    """Return acquisition-level whole-edge pattern-similarity endpoints."""
    if not isinstance(dataset, TimeSeriesDataset):
        raise TypeError("dataset must be a TimeSeriesDataset")
    if not isinstance(estimator, SlidingWindowFC):
        raise TypeError("estimator must be a SlidingWindowFC")
    rows: list[dict[str, Any]] = []
    for run in dataset.runs:
        result = estimator.transform(run)
        adjacent = adjacent_window_pattern_similarity(result).similarities
        has_pairs = bool(len(adjacent))
        all_pair = (
            all_pair_window_pattern_similarity(result) if has_pairs else None
        )
        adjacent_mean = float(np.mean(adjacent)) if has_pairs else None
        values = {
            "all_pair_similarity": all_pair,
            "adjacent_similarity": adjacent_mean,
            "adjacency_excess": (
                None
                if all_pair is None or adjacent_mean is None
                else adjacent_mean - all_pair
            ),
        }
        base = {
            "subject": run.subject,
            "session": run.session,
            "acquisition_id": run.acquisition_id,
            "length": estimator.length,
            "step": estimator.step,
            "taper": estimator.taper,
            "n_windows": len(result.start_frames),
            "n_adjacent_pairs": len(adjacent),
            "statistic": "mean",
        }
        for measure, value in values.items():
            rows.append(
                {
                    **base,
                    "endpoint": f"{measure}.mean",
                    "feature": ["whole_edge_pattern", measure],
                    "measure": measure,
                    "value": value,
                }
            )
    return {
        "format": "dfc-kit-window-pattern-endpoints",
        "format_version": 1,
        "source_contract": (
            f"window-pattern:length={estimator.length};step={estimator.step};"
            f"taper={estimator.taper};Fisher-z-edges;"
            "within-retained-segments"
        ),
        "summary": "within-acquisition whole-edge pattern similarity",
        "length": estimator.length,
        "step": estimator.step,
        "taper": estimator.taper,
        "roi_names": list(dataset.roi_names),
        "n_acquisitions": dataset.n_runs,
        "rows": rows,
    }
