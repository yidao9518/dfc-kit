"""Paired inference for generic named acquisition-level endpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .multiple_testing import benjamini_hochberg
from .paired import paired_bootstrap_mean_ci, paired_sign_flip


def infer_paired_endpoints(
    payload: dict[str, Any],
    *,
    condition_a: str,
    condition_b: str,
    fdr_family: str,
    alpha: float = 0.05,
    n_permutations: int = 10_000,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    exact: bool = False,
) -> dict[str, Any]:
    """Test condition A minus B for every endpoint in a strict long-form payload."""
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise TypeError("endpoint payload rows must be a non-empty array")
    if not condition_a or not condition_b or condition_a == condition_b:
        raise ValueError("paired conditions must be distinct non-empty labels")
    if not fdr_family.strip():
        raise ValueError("fdr_family must be a non-empty prespecified family")
    by_endpoint: dict[str, dict[str, dict[str, float]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("endpoint rows must contain JSON objects")
        subject = row.get("subject")
        session = row.get("session")
        endpoint = row.get("endpoint")
        value = row.get("value")
        if not all(isinstance(item, str) and item for item in (subject, session, endpoint)):
            raise TypeError("endpoint rows require subject, session, and endpoint strings")
        if session not in {condition_a, condition_b}:
            continue
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError("endpoint row values must be finite numbers")
        subjects = by_endpoint.setdefault(endpoint, {})
        if value is not None:
            cell = subjects.setdefault(subject, {})
            if session in cell:
                raise ValueError(
                    f"endpoint {endpoint!r} subject {subject!r} has duplicate condition {session!r}"
                )
            cell[session] = float(value)
        current = {
            key: row[key]
            for key in ("feature", "statistic", "measure", "length")
            if row.get(key) is not None
        }
        if endpoint in metadata and metadata[endpoint] != current:
            raise ValueError(f"endpoint {endpoint!r} metadata changes across rows")
        metadata[endpoint] = current
    results = []
    tested_indices = []
    pvalues = []
    for endpoint_index, endpoint in enumerate(sorted(by_endpoint)):
        subjects = tuple(
            subject
            for subject in sorted(by_endpoint[endpoint])
            if condition_a in by_endpoint[endpoint][subject]
            and condition_b in by_endpoint[endpoint][subject]
        )
        base = {"endpoint": endpoint, **metadata.get(endpoint, {})}
        if len(subjects) < 2:
            results.append(
                {
                    **base,
                    "n": len(subjects),
                    "estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "p": None,
                    "q": None,
                    "direction": None,
                    "result_status": "not_testable",
                    "reason": "fewer than two complete participant pairs",
                }
            )
            continue
        differences = np.asarray(
            [
                by_endpoint[endpoint][subject][condition_a]
                - by_endpoint[endpoint][subject][condition_b]
                for subject in subjects
            ]
        )
        test = paired_sign_flip(
            differences,
            subjects,
            n_permutations=n_permutations,
            seed=None if exact else seed + endpoint_index,
            exact=exact,
        )
        interval = paired_bootstrap_mean_ci(
            differences,
            subjects,
            n_resamples=n_bootstrap,
            seed=seed + 100_000 + endpoint_index,
        )
        results.append(
            {
                **base,
                "n": len(subjects),
                "estimate": test.estimate,
                "ci_low": interval.lower,
                "ci_high": interval.upper,
                "p": test.pvalue,
                "q": None,
                "direction": (
                    f"{condition_a}_higher"
                    if test.estimate > 0
                    else f"{condition_a}_lower"
                    if test.estimate < 0
                    else "no_difference"
                ),
                "result_status": None,
                "reason": None,
            }
        )
        tested_indices.append(len(results) - 1)
        pvalues.append(test.pvalue)
    if pvalues:
        adjusted = benjamini_hochberg(pvalues, family=fdr_family).adjusted_pvalues
        for index, qvalue in zip(tested_indices, adjusted, strict=True):
            results[index]["q"] = float(qvalue)
            results[index]["result_status"] = "positive" if qvalue < alpha else "negative"
    return {
        "format": "dfc-kit-paired-endpoint-inference",
        "format_version": 1,
        "source_format": payload.get("format"),
        "source_contract": payload.get("source_contract"),
        "contrast": f"{condition_a} - {condition_b}",
        "condition_a": condition_a,
        "condition_b": condition_b,
        "test": "paired participant-level sign flip",
        "confidence_interval": "participant bootstrap percentile",
        "fdr_family": fdr_family,
        "alpha": alpha,
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "exact": exact,
        "n_endpoints": len(results),
        "n_tested": len(tested_indices),
        "results": results,
    }


def infer_paired_endpoints_file(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("endpoint input must be a JSON object")
    return infer_paired_endpoints(raw, **kwargs)


def write_paired_endpoint_inference(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"endpoint-inference output already exists: {target}")
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
    "infer_paired_endpoints",
    "infer_paired_endpoints_file",
    "write_paired_endpoint_inference",
]
