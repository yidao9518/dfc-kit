"""Study-level FDR correction over explicitly labelled result families."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .multiple_testing import benjamini_hochberg


def adjust_result_families(
    records: list[dict[str, Any]],
    *,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Return copied result records with BH q values calculated within each named family."""
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    output = [dict(record) for record in records]
    identifiers: set[str] = set()
    families: dict[str, list[int]] = {}
    for index, record in enumerate(output):
        result_id = record.get("result_id")
        if not isinstance(result_id, str) or not result_id.strip():
            raise TypeError("every result requires a non-empty result_id")
        if result_id in identifiers:
            raise ValueError(f"duplicate result_id: {result_id}")
        identifiers.add(result_id)
        pvalue = record.get("p")
        if pvalue is None:
            if record.get("result_status") not in {"not_testable", "qc_failed"}:
                raise ValueError("results without p require not_testable or qc_failed status")
            record["q"] = None
            continue
        if isinstance(pvalue, bool) or not isinstance(pvalue, (int, float)):
            raise TypeError("result p values must be numeric or null")
        if not np.isfinite(pvalue) or not 0.0 <= pvalue <= 1.0:
            raise ValueError("result p values must lie within [0, 1]")
        family = record.get("fdr_family")
        if not isinstance(family, str) or not family.strip():
            raise ValueError("every tested result requires a non-empty fdr_family")
        families.setdefault(family, []).append(index)
    for family, indices in families.items():
        correction = benjamini_hochberg(
            [float(output[index]["p"]) for index in indices],
            family=family,
        )
        for index, qvalue in zip(indices, correction.adjusted_pvalues, strict=True):
            output[index]["q"] = float(qvalue)
            output[index]["result_status"] = "positive" if qvalue < alpha else "negative"
    return output


def adjust_result_families_file(path: str | Path, *, alpha: float = 0.05) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"results"}:
        raise TypeError("result-family input must contain exactly one results array")
    if not isinstance(raw["results"], list) or any(
        not isinstance(record, dict) for record in raw["results"]
    ):
        raise TypeError("result-family results must be an array of JSON objects")
    adjusted = adjust_result_families(raw["results"], alpha=alpha)
    return {
        "format": "dfc-kit-adjusted-result-families",
        "format_version": 1,
        "method": "benjamini-hochberg",
        "alpha": alpha,
        "families": sorted(
            {
                record["fdr_family"]
                for record in adjusted
                if record.get("p") is not None
            }
        ),
        "results": adjusted,
    }


def write_adjusted_result_families(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"adjusted result output already exists: {target}")
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
    "adjust_result_families",
    "adjust_result_families_file",
    "write_adjusted_result_families",
]
