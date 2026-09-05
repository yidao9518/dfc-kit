"""Paired inference for generic named acquisition-level endpoints."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts._json import write_json_atomic
from .multiple_testing import benjamini_hochberg
from .paired import paired_bootstrap_mean_ci, paired_sign_flip


def _paired_endpoint_result(
    cells: dict[str, dict[str, dict[str, float]]],
    *,
    endpoint: str,
    condition_a: str,
    condition_b: str,
    endpoint_index: int,
    seed: int,
    exact: bool,
    n_permutations: int,
    n_bootstrap: int,
    standardized: bool = False,
    errors_as_not_testable: bool = False,
) -> dict[str, Any]:
    """Test one participant-paired endpoint; callers own its seed index and schema."""
    subjects = tuple(
        subject
        for subject in sorted(cells)
        if cells[subject].get(condition_a) and cells[subject].get(condition_b)
    )
    result = {
        "endpoint": endpoint,
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
    if standardized:
        result.update(
            standardized_estimate=None, standardized_estimate_definition="paired Cohen dz"
        )
    if len(subjects) < 2:
        return result
    differences = np.asarray(
        [
            np.mean(tuple(cells[subject][condition_a].values()))
            - np.mean(tuple(cells[subject][condition_b].values()))
            for subject in subjects
        ],
        dtype=float,
    )
    try:
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
    except ValueError as error:
        if not errors_as_not_testable:
            raise
        result["reason"] = str(error)
        return result
    if standardized:
        scale = float(differences.std(ddof=1))
        result["standardized_estimate"] = (
            None if not np.isfinite(scale) or scale <= 0.0 else float(test.estimate / scale)
        )
    result.update(
        estimate=test.estimate,
        ci_low=interval.lower,
        ci_high=interval.upper,
        p=test.pvalue,
        direction=(
            f"{condition_a}_higher"
            if test.estimate > 0
            else f"{condition_a}_lower"
            if test.estimate < 0
            else "no_difference"
        ),
        result_status=None,
        reason=None,
    )
    for label, condition in (("a", condition_a), ("b", condition_b)):
        counts = [len(cells[subject][condition]) for subject in subjects]
        result[f"condition_{label}_acquisitions_min"] = min(counts)
        result[f"condition_{label}_acquisitions_max"] = max(counts)
    return result


def _adjust_endpoint_pvalues(results: list[dict[str, Any]], *, family: str, alpha: float) -> int:
    """Adjust only testable endpoints, preserving their output order."""
    tested = [result for result in results if result["p"] is not None]
    if tested:
        adjusted = benjamini_hochberg([result["p"] for result in tested], family=family)
        for result, qvalue in zip(tested, adjusted.adjusted_pvalues, strict=True):
            result["q"] = float(qvalue)
            result["result_status"] = "positive" if qvalue < alpha else "negative"
    return len(tested)


def _validated_endpoint_names(
    endpoint_names: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if endpoint_names is None:
        return None
    if isinstance(endpoint_names, (str, bytes)) or not isinstance(endpoint_names, Sequence):
        raise TypeError("endpoint_names must be a sequence of strings or None")
    names = tuple(endpoint_names)
    if not names:
        raise ValueError("endpoint_names must not be empty when provided")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("endpoint_names must contain only strings")
    if any(not name.strip() for name in names):
        raise ValueError("endpoint_names must contain only non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("endpoint_names must not contain duplicates")
    return names


def infer_paired_endpoints(
    payload: dict[str, Any],
    *,
    condition_a: str,
    condition_b: str,
    fdr_family: str,
    endpoint_names: Sequence[str] | None = None,
    alpha: float = 0.05,
    n_permutations: int = 10_000,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    exact: bool = False,
    within_condition_aggregation: str = "error",
) -> dict[str, Any]:
    """Test condition A minus B for selected endpoints in a strict long-form payload."""
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise TypeError("endpoint payload rows must be a non-empty array")
    if not condition_a or not condition_b or condition_a == condition_b:
        raise ValueError("paired conditions must be distinct non-empty labels")
    if not fdr_family.strip():
        raise ValueError("fdr_family must be a non-empty prespecified family")
    if within_condition_aggregation not in {"error", "mean"}:
        raise ValueError("within_condition_aggregation must be 'error' or 'mean'")
    requested_endpoints = _validated_endpoint_names(endpoint_names)
    requested_set = None if requested_endpoints is None else set(requested_endpoints)
    by_endpoint: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    available_endpoints: set[str] = set()
    contrast_endpoints: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("endpoint rows must contain JSON objects")
        subject = row.get("subject")
        session = row.get("session")
        endpoint = row.get("endpoint")
        value = row.get("value")
        if not all(isinstance(item, str) and item for item in (subject, session, endpoint)):
            raise TypeError("endpoint rows require subject, session, and endpoint strings")
        available_endpoints.add(endpoint)
        if session not in {condition_a, condition_b}:
            continue
        contrast_endpoints.add(endpoint)
        if requested_set is not None and endpoint not in requested_set:
            continue
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
        ):
            raise ValueError("endpoint row values must be finite numbers")
        subjects = by_endpoint.setdefault(endpoint, {})
        if value is not None:
            cell = subjects.setdefault(subject, {})
            acquisitions = cell.setdefault(session, {})
            acquisition_id = row.get("acquisition_id")
            acquisition_key = (
                acquisition_id
                if isinstance(acquisition_id, str) and acquisition_id
                else "<unspecified>"
            )
            if acquisitions and within_condition_aggregation == "error":
                raise ValueError(
                    f"endpoint {endpoint!r} subject {subject!r} has duplicate condition {session!r}"
                )
            if acquisition_key in acquisitions:
                raise ValueError(
                    f"endpoint {endpoint!r} subject {subject!r} condition {session!r} "
                    f"has duplicate acquisition {acquisition_key!r}"
                )
            acquisitions[acquisition_key] = float(value)
        current = {
            key: row[key]
            for key in ("feature", "statistic", "measure", "length")
            if row.get(key) is not None
        }
        if endpoint in metadata and metadata[endpoint] != current:
            raise ValueError(f"endpoint {endpoint!r} metadata changes across rows")
        metadata[endpoint] = current
    if requested_endpoints is not None:
        unknown = tuple(name for name in requested_endpoints if name not in available_endpoints)
        if unknown:
            rendered = ", ".join(repr(name) for name in unknown)
            raise ValueError(f"unknown endpoint_names: {rendered}")
        for endpoint in requested_endpoints:
            by_endpoint.setdefault(endpoint, {})
    seed_indices = {endpoint: index for index, endpoint in enumerate(sorted(contrast_endpoints))}
    results = []
    for endpoint in sorted(by_endpoint):
        result = _paired_endpoint_result(
            by_endpoint[endpoint],
            endpoint=endpoint,
            condition_a=condition_a,
            condition_b=condition_b,
            endpoint_index=seed_indices.get(endpoint, len(seed_indices)),
            seed=seed,
            exact=exact,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
            standardized=True,
        )
        results.append({**result, **metadata.get(endpoint, {})})
    n_tested = _adjust_endpoint_pvalues(results, family=fdr_family, alpha=alpha)
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
        "within_condition_aggregation": within_condition_aggregation,
        "endpoint_selection": {
            "mode": "all" if requested_endpoints is None else "explicit",
            "requested_endpoint_names": (
                None if requested_endpoints is None else list(requested_endpoints)
            ),
            "selected_endpoint_names": sorted(by_endpoint),
            "n_source_endpoints": len(available_endpoints),
        },
        "n_endpoints": len(results),
        "n_tested": n_tested,
        "results": results,
    }


def _infer_paired_endpoints_file(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("endpoint input must be a JSON object")
    return infer_paired_endpoints(raw, **kwargs)


def _write_paired_endpoint_inference(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"endpoint-inference output already exists: {target}")
    return write_json_atomic(target, payload)


__all__ = [
    "infer_paired_endpoints",
]
