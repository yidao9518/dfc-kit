"""Shared validation for selecting named feature columns."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FeatureKey = tuple[str, ...]


def resolve_feature_selection(
    available_keys: Sequence[FeatureKey],
    *,
    feature_keys: Iterable[Sequence[str]] | None,
    feature_mask: ArrayLike | None,
) -> tuple[NDArray[np.int64], tuple[FeatureKey, ...]]:
    """Resolve one exact-key or Boolean-mask selection against known features."""
    if (feature_keys is None) == (feature_mask is None):
        raise ValueError("provide exactly one of feature_keys or feature_mask")

    available = tuple(tuple(str(part) for part in key) for key in available_keys)
    if feature_keys is not None:
        requested = tuple(tuple(str(part) for part in key) for key in feature_keys)
        if not requested or any(not key for key in requested):
            raise ValueError("feature_keys must select at least one non-empty key")
        if len(set(requested)) != len(requested):
            raise ValueError("feature_keys must be unique")
        positions = {key: index for index, key in enumerate(available)}
        missing = tuple(key for key in requested if key not in positions)
        if missing:
            raise KeyError(f"selected feature keys are absent: {missing}")
        indices = np.asarray([positions[key] for key in requested], dtype=np.int64)
        return indices, requested

    mask = np.asarray(feature_mask)
    if mask.dtype.kind != "b" or mask.shape != (len(available),):
        raise ValueError(f"feature_mask must be a Boolean vector with shape ({len(available)},)")
    indices = np.flatnonzero(mask).astype(np.int64, copy=False)
    if not len(indices):
        raise ValueError("feature_mask must select at least one feature")
    return indices, tuple(available[int(index)] for index in indices)
