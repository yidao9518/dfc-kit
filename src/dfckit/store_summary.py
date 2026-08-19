"""Acquisition-level summaries of disk-backed dFC feature stores."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .storage import FeatureStore


def summarize_store_means(store: FeatureStore) -> dict[str, Any]:
    """Average every named feature within each acquisition without crossing identities."""
    sums: dict[tuple[str, str | None, str | None], np.ndarray] = {}
    counts: dict[tuple[str, str | None, str | None], int] = {}
    for chunk in store.iter_chunks(mmap=True):
        identity = (chunk.subject, chunk.session, chunk.acquisition_id)
        if identity not in sums:
            sums[identity] = np.zeros(store.n_features, dtype=np.float64)
            counts[identity] = 0
        values = np.asarray(chunk.values, dtype=np.float64)
        sums[identity] += values.sum(axis=0)
        counts[identity] += len(values)
    if not sums:
        raise ValueError("cannot summarize an empty FeatureStore")
    rows = []
    for identity in sorted(sums, key=lambda item: tuple("" if value is None else value for value in item)):
        means = sums[identity] / counts[identity]
        for index, (feature, value) in enumerate(zip(store.feature_keys, means, strict=True)):
            rows.append(
                {
                    "subject": identity[0],
                    "session": identity[1],
                    "acquisition_id": identity[2],
                    "endpoint": f"feature_{index}",
                    "feature": list(feature),
                    "value": float(value),
                    "n_samples": counts[identity],
                }
            )
    feature_type = "edge" if all(len(key) == 2 for key in store.feature_keys) else "node"
    expected_arity = 2 if feature_type == "edge" else 1
    if not all(len(key) == expected_arity for key in store.feature_keys):
        raise ValueError("mixed feature-key arities cannot be summarized as one endpoint family")
    return {
        "format": "dfc-kit-store-endpoints",
        "format_version": 1,
        "source_contract": store.source_contract,
        "summary": "within-acquisition arithmetic mean",
        "feature_type": feature_type,
        "n_features": store.n_features,
        "n_acquisitions": len(sums),
        "rows": rows,
    }


def summarize_store_file(path: str | Path) -> dict[str, Any]:
    return summarize_store_means(FeatureStore.open(path))


def write_store_summary(payload: dict[str, Any], path: str | Path) -> Path:
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


__all__ = ["summarize_store_file", "summarize_store_means", "write_store_summary"]
