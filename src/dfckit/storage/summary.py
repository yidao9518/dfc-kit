"""Acquisition-level summaries of disk-backed dFC feature stores."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..connectivity.correlation import fisher_z_edges, weighted_correlation
from ..data import TimeSeriesDataset
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


def summarize_static_fc_dataset(dataset: TimeSeriesDataset) -> dict[str, Any]:
    """Return whole-acquisition Fisher-z FC endpoints for a dataset."""
    if not isinstance(dataset, TimeSeriesDataset):
        raise TypeError("dataset must be a TimeSeriesDataset")
    rows = []
    edge_names: tuple[tuple[str, str], ...] | None = None
    for run in dataset.runs:
        values, edge_i, edge_j = fisher_z_edges(weighted_correlation(run.values))
        current_names = tuple(
            (run.roi_names[int(left)], run.roi_names[int(right)])
            for left, right in zip(edge_i, edge_j, strict=True)
        )
        if edge_names is None:
            edge_names = current_names
        elif current_names != edge_names:
            raise RuntimeError("static FC edge order changed between acquisitions")
        rows.extend(
            {
                "subject": run.subject,
                "session": run.session,
                "acquisition_id": run.acquisition_id,
                "endpoint": f"feature_{index}.mean",
                "feature": list(feature),
                "statistic": "mean",
                "measure": "whole_acquisition_fisher_z_fc",
                "value": float(value),
                "n_samples": run.n_frames,
            }
            for index, (feature, value) in enumerate(
                zip(current_names, values, strict=True)
            )
        )
    assert edge_names is not None
    return {
        "format": "dfc-kit-store-endpoints",
        "format_version": 2,
        "source_contract": "static-fc:retained-xcpd-frames:fisher-z:v1",
        "summary": "whole-acquisition Fisher-z FC over retained XCP-D frames",
        "statistics": ["mean"],
        "feature_type": "edge",
        "n_features": len(edge_names),
        "n_acquisitions": dataset.n_runs,
        "rows": rows,
    }


def _summarize_store_file(
    path: str | Path,
    statistics: Sequence[str] = ("mean",),
) -> dict[str, Any]:
    return summarize_store_statistics(FeatureStore.open(path), statistics)


def _write_store_summary(payload: dict[str, Any], path: str | Path) -> Path:
    from ..artifacts._json import write_json_atomic

    return write_json_atomic(path, payload)


__all__ = [
    "summarize_static_fc_dataset",
    "summarize_store_statistics",
]
