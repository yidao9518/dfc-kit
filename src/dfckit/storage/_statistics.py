"""Bounded-memory moments for stored feature rows."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .store import FeatureStore


@dataclass
class StreamingFeatureMoments:
    """Accumulate per-feature population moments across row chunks."""

    count: int
    mean: NDArray[np.float64]
    m2: NDArray[np.float64]
    minimum: NDArray[np.float64]
    maximum: NDArray[np.float64]

    @classmethod
    def empty(cls, n_features: int) -> StreamingFeatureMoments:
        return cls(
            count=0,
            mean=np.zeros(n_features, dtype=np.float64),
            m2=np.zeros(n_features, dtype=np.float64),
            minimum=np.full(n_features, np.inf, dtype=np.float64),
            maximum=np.full(n_features, -np.inf, dtype=np.float64),
        )

    def update(self, rows: NDArray[np.floating]) -> None:
        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[1:] != self.mean.shape:
            raise ValueError("feature rows must be a rows-by-features array")
        if len(values) == 0:
            return
        chunk_count = len(values)
        chunk_mean = values.mean(axis=0)
        centered = values - chunk_mean
        chunk_m2 = np.einsum("ij,ij->j", centered, centered)
        combined = self.count + chunk_count
        difference = chunk_mean - self.mean
        self.m2 += (
            chunk_m2
            + np.square(difference) * (self.count * chunk_count / combined)
        )
        self.mean += difference * (chunk_count / combined)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        self.count = combined

    @property
    def variance(self) -> NDArray[np.float64]:
        if self.count == 0:
            raise ValueError("feature moments contain no samples")
        return np.maximum(self.m2 / self.count, 0.0)

    @property
    def standard_deviation(self) -> NDArray[np.float64]:
        return np.sqrt(self.variance)


def store_feature_moments(
    store: FeatureStore,
    *,
    subjects: Iterable[str] | None = None,
) -> StreamingFeatureMoments:
    """Summarize selected store rows without materializing them together."""
    moments = StreamingFeatureMoments.empty(store.n_features)
    for chunk in store.iter_chunks(subjects=subjects, mmap=True):
        moments.update(chunk.values)
    if moments.count == 0:
        raise ValueError("no selected samples are available in the FeatureStore")
    return moments


__all__ = ["StreamingFeatureMoments", "store_feature_moments"]
