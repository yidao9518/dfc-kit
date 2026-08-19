"""Deterministic descriptions of fitted state-model spatial patterns."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .io import fitted_model_fingerprint, load_fitted_model
from .states.kmeans import KMeansStateModel
from .storage import FeatureStore


def load_network_map(path: str | Path | None) -> dict[str, str] | None:
    """Load an optional exact ROI-to-network mapping."""
    if path is None:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("network map must be a non-empty JSON object")
    mapping: dict[str, str] = {}
    for roi, network in raw.items():
        if not isinstance(roi, str) or not roi.strip():
            raise ValueError("network map ROI names must be non-empty strings")
        if not isinstance(network, str) or not network.strip():
            raise ValueError("network map labels must be non-empty strings")
        mapping[roi] = network
    return mapping


def _store_statistics(
    store: FeatureStore,
    subjects: tuple[str, ...],
) -> tuple[int, np.ndarray, np.ndarray]:
    count = 0
    mean = np.zeros(store.n_features, dtype=np.float64)
    m2 = np.zeros(store.n_features, dtype=np.float64)
    for chunk in store.iter_chunks(subjects=subjects, mmap=True):
        values = np.asarray(chunk.values, dtype=np.float64)
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.einsum("ij,ij->j", centered, centered)
        combined = count + batch_count
        delta = batch_mean - mean
        m2 += batch_m2 + np.square(delta) * (count * batch_count / combined)
        mean += delta * (batch_count / combined)
        count = combined
    if count == 0:
        raise ValueError("no fitted-subject samples are available in the FeatureStore")
    scale = np.sqrt(m2 / count)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return count, mean, scale


def _feature_record(
    key: tuple[str, ...],
    *,
    center: float,
    baseline: float,
    deviation_z: float,
    network_map: Mapping[str, str] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "feature": list(key),
        "center": float(center),
        "pooled_mean": float(baseline),
        "deviation": float(center - baseline),
        "deviation_z": float(deviation_z),
    }
    if network_map is not None:
        record["networks"] = [network_map[item] for item in key]
    return record


def _block_summaries(
    keys: tuple[tuple[str, ...], ...],
    centers: np.ndarray,
    deviations: np.ndarray,
    network_map: Mapping[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[int]] = {}
    for index, key in enumerate(keys):
        networks = tuple(network_map[item] for item in key)
        block = tuple(sorted(networks)) if len(networks) == 2 else networks
        grouped.setdefault(block, []).append(index)
    output = []
    for block, indices in grouped.items():
        selected = np.asarray(indices, dtype=np.int64)
        output.append(
            {
                "networks": list(block),
                "n_features": len(indices),
                "mean_center": float(np.mean(centers[selected])),
                "mean_deviation_z": float(np.mean(deviations[selected])),
                "mean_absolute_deviation_z": float(np.mean(np.abs(deviations[selected]))),
            }
        )
    output.sort(key=lambda item: (-item["mean_absolute_deviation_z"], item["networks"]))
    return output


def describe_kmeans_states(
    store: FeatureStore,
    model: KMeansStateModel,
    *,
    top_features: int = 10,
    network_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe each centroid relative to its fitted cohort's pooled feature distribution."""
    if not isinstance(model, KMeansStateModel):
        raise TypeError("state description currently requires a KMeans model")
    if not isinstance(top_features, int) or isinstance(top_features, bool) or top_features < 1:
        raise ValueError("top_features must be a positive integer")
    if store.feature_keys != model.feature_keys:
        raise ValueError("model and FeatureStore feature identities or order differ")
    if store.source_contract != model.source_contract:
        raise ValueError("model and FeatureStore source contracts differ")
    missing_subjects = sorted(set(model.fit_subjects) - set(store.subjects))
    if missing_subjects:
        raise ValueError(f"fitted subjects are absent from the FeatureStore: {missing_subjects}")
    if network_map is not None:
        rois = {roi for key in model.feature_keys for roi in key}
        missing_rois = sorted(rois - set(network_map))
        if missing_rois:
            raise ValueError(f"network map is missing model ROIs: {missing_rois}")

    sample_count, pooled_mean, pooled_scale = _store_statistics(store, model.fit_subjects)
    feature_type = "edge" if all(len(key) == 2 for key in model.feature_keys) else "node"
    expected_arity = 2 if feature_type == "edge" else 1
    if not all(len(key) == expected_arity for key in model.feature_keys):
        raise ValueError("mixed feature-key arities cannot be interpreted as one state model")
    limit = min(top_features, len(model.feature_keys))
    states = []
    for state, center_values in enumerate(np.asarray(model.centers, dtype=float)):
        deviation_z = (center_values - pooled_mean) / pooled_scale
        positive = np.argsort(-deviation_z, kind="stable")[:limit]
        negative = np.argsort(deviation_z, kind="stable")[:limit]
        absolute = np.argsort(-np.abs(deviation_z), kind="stable")[:limit]

        def records(
            indices: np.ndarray,
            values: np.ndarray = center_values,
            standardized_deviation: np.ndarray = deviation_z,
        ) -> list[dict[str, Any]]:
            return [
                _feature_record(
                    model.feature_keys[int(index)],
                    center=float(values[index]),
                    baseline=float(pooled_mean[index]),
                    deviation_z=float(standardized_deviation[index]),
                    network_map=network_map,
                )
                for index in indices
            ]

        states.append(
            {
                "state": state,
                "top_positive_features": records(positive),
                "top_negative_features": records(negative),
                "top_distinguishing_features": records(absolute),
                "network_blocks": (
                    []
                    if network_map is None
                    else _block_summaries(
                        model.feature_keys,
                        center_values,
                        deviation_z,
                        network_map,
                    )
                ),
            }
        )
    return {
        "format": "dfc-kit-state-description",
        "format_version": 1,
        "model_kind": "kmeans-state",
        "model_fingerprint": fitted_model_fingerprint(model),
        "training_data_fingerprint": model.training_data_fingerprint,
        "source_contract": model.source_contract,
        "feature_type": feature_type,
        "n_states": model.n_states,
        "n_features": len(model.feature_keys),
        "fit_subjects": list(model.fit_subjects),
        "fit_sample_count": model.fit_sample_count,
        "reference_sample_count": sample_count,
        "reference": "pooled fitted-subject FeatureStore distribution",
        "ranking": "centroid deviation from pooled mean divided by pooled feature SD",
        "top_features": limit,
        "network_map_applied": network_map is not None,
        "states": states,
    }


def describe_kmeans_state_artifacts(
    store_path: str | Path,
    model_path: str | Path,
    *,
    top_features: int = 10,
    network_map_path: str | Path | None = None,
) -> dict[str, Any]:
    store = FeatureStore.open(store_path)
    model = load_fitted_model(model_path)
    if not isinstance(model, KMeansStateModel):
        raise TypeError("state description currently requires a KMeans model artifact")
    return describe_kmeans_states(
        store,
        model,
        top_features=top_features,
        network_map=load_network_map(network_map_path),
    )


def write_state_description(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"state-description output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "describe_kmeans_state_artifacts",
    "describe_kmeans_states",
    "load_network_map",
    "write_state_description",
]
