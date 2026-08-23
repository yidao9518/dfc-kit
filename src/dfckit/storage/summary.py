"""Acquisition-level summaries of disk-backed dFC feature stores."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._statistics import StreamingFeatureMoments
from .store import FeatureStore

_STORE_STATISTICS = (
    "mean",
    "variance",
    "standard_deviation",
    "minimum",
    "maximum",
)


def _validated_statistics(statistics: Sequence[str]) -> tuple[str, ...]:
    if isinstance(statistics, str):
        raise TypeError("statistics must be a sequence of statistic names")
    selected = tuple(statistics)
    if not selected:
        raise ValueError("statistics must contain at least one value")
    if any(not isinstance(statistic, str) for statistic in selected):
        raise TypeError("statistics must contain strings")
    unsupported = sorted(set(selected) - set(_STORE_STATISTICS))
    if unsupported:
        raise ValueError(f"unsupported store statistics: {unsupported}")
    if len(set(selected)) != len(selected):
        raise ValueError("statistics must be unique")
    return selected


def _feature_type(store: FeatureStore) -> str:
    arities = {len(key) for key in store.feature_keys}
    if arities == {2}:
        return "edge"
    if arities == {1}:
        return "node"
    raise ValueError("mixed feature-key arities cannot be summarized as one endpoint family")


def summarize_store_statistics(
    store: FeatureStore,
    statistics: Sequence[str] = ("mean",),
) -> dict[str, Any]:
    """Summarize every named feature within each acquisition.

    Mean and second moments are combined across chunks with the parallel
    Welford update. Variance is the population variance over all retained
    feature samples in the acquisition (``ddof=0``).
    """
    selected = _validated_statistics(statistics)
    feature_type = _feature_type(store)
    moments_by_acquisition: dict[
        tuple[str, str | None, str | None], StreamingFeatureMoments
    ] = {}
    for chunk in store.iter_chunks(mmap=True):
        identity = (chunk.subject, chunk.session, chunk.acquisition_id)
        moments = moments_by_acquisition.setdefault(
            identity,
            StreamingFeatureMoments.empty(store.n_features),
        )
        moments.update(chunk.values)
    if not moments_by_acquisition:
        raise ValueError("cannot summarize an empty FeatureStore")
    rows = []
    for identity in sorted(
        moments_by_acquisition,
        key=lambda item: tuple("" if value is None else value for value in item),
    ):
        moments = moments_by_acquisition[identity]
        values_by_statistic = {
            "mean": moments.mean,
            "variance": moments.variance,
            "standard_deviation": moments.standard_deviation,
            "minimum": moments.minimum,
            "maximum": moments.maximum,
        }
        for index, feature in enumerate(store.feature_keys):
            for statistic in selected:
                rows.append(
                    {
                        "subject": identity[0],
                        "session": identity[1],
                        "acquisition_id": identity[2],
                        "endpoint": f"feature_{index}.{statistic}",
                        "feature": list(feature),
                        "statistic": statistic,
                        "value": float(values_by_statistic[statistic][index]),
                        "n_samples": moments.count,
                    }
                )
    return {
        "format": "dfc-kit-store-endpoints",
        "format_version": 2,
        "source_contract": store.source_contract,
        "summary": "within-acquisition feature statistics",
        "statistics": list(selected),
        "variance_definition": "population variance across retained samples (ddof=0)",
        "feature_type": feature_type,
        "n_features": store.n_features,
        "n_acquisitions": len(moments_by_acquisition),
        "rows": rows,
    }


def _summarize_store_file(
    path: str | Path,
    statistics: Sequence[str] = ("mean",),
) -> dict[str, Any]:
    return summarize_store_statistics(FeatureStore.open(path), statistics)


def _write_store_summary(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"store-summary output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "summarize_store_statistics",
]
