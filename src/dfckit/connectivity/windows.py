"""Censor-bounded sliding-window functional connectivity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..data import TimeSeriesRun
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
