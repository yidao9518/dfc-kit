"""HC3 group comparisons for two independent endpoint cohorts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .endpoints import _adjust_endpoint_pvalues, _write_paired_endpoint_inference
from .hc3 import hc3_confidence_interval, ols_hc3


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in ("feature", "statistic", "measure", "length")
        if row.get(key) is not None
    }


def _collect_subject_endpoints(
    payload: dict[str, Any],
    *,
    aggregation: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]], dict[str, int]]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise TypeError("endpoint payload rows must be a non-empty array")
    if aggregation not in {"error", "mean"}:
        raise ValueError("within_group_aggregation must be 'error' or 'mean'")
    acquisitions: dict[str, dict[str, dict[tuple[str, str], float]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("endpoint rows must contain JSON objects")
        subject = row.get("subject")
        session = row.get("session")
        endpoint = row.get("endpoint")
        value = row.get("value")
        if not isinstance(subject, str) or not subject:
            raise TypeError("endpoint rows require a non-empty subject string")
        if not isinstance(endpoint, str) or not endpoint:
            raise TypeError("endpoint rows require a non-empty endpoint string")
        if session is not None and (not isinstance(session, str) or not session):
            raise TypeError("endpoint session must be null or a non-empty string")
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
        ):
            raise ValueError("endpoint row values must be finite numbers")
        current = _metadata(row)
        if endpoint in metadata and metadata[endpoint] != current:
            raise ValueError(f"endpoint {endpoint!r} metadata changes across rows")
        metadata[endpoint] = current
        if value is None:
            acquisitions.setdefault(endpoint, {}).setdefault(subject, {})
            continue
        acquisition_id = row.get("acquisition_id")
        if acquisition_id is not None and (
            not isinstance(acquisition_id, str) or not acquisition_id
        ):
            raise TypeError("acquisition_id must be null or a non-empty string")
        key = (session or "<unspecified-session>", acquisition_id or "<unspecified-run>")
        subject_runs = acquisitions.setdefault(endpoint, {}).setdefault(subject, {})
        if key in subject_runs:
            raise ValueError(
                f"endpoint {endpoint!r} subject {subject!r} has duplicate acquisition {key!r}"
            )
        subject_runs[key] = float(value)
    values: dict[str, dict[str, float]] = {}
    maximum_counts: dict[str, int] = {}
    for endpoint, subjects in acquisitions.items():
        values[endpoint] = {}
        counts = []
        for subject, runs in subjects.items():
            if not runs:
                continue
            if aggregation == "error" and len(runs) != 1:
                raise ValueError(
                    f"endpoint {endpoint!r} subject {subject!r} has repeated acquisitions"
                )
            values[endpoint][subject] = float(np.mean(tuple(runs.values())))
            counts.append(len(runs))
        maximum_counts[endpoint] = max(counts, default=0)
    return values, metadata, maximum_counts


def load_group_covariates(
    path: str | Path,
    *,
    covariate_names: tuple[str, ...],
    group_column: str = "group",
    subject_column: str = "subject",
) -> dict[tuple[str, str], tuple[float, ...]]:
    """Load numeric covariates keyed by group and subject from a strict TSV."""
    names = tuple(str(name) for name in covariate_names)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("covariate_names must be a non-empty unique sequence")
    if group_column in names or subject_column in names or group_column == subject_column:
        raise ValueError("group, subject, and covariate columns must be distinct")
    output: dict[tuple[str, str], tuple[float, ...]] = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = {group_column, subject_column, *names}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(f"covariate TSV must contain columns {sorted(expected)}")
        for row in reader:
            group = row[group_column].strip()
            subject = row[subject_column].strip()
            if not group or not subject:
                raise ValueError("covariate TSV group and subject values must be non-empty")
            key = (group, subject)
            if key in output:
                raise ValueError(f"duplicate covariate row for group/subject {key!r}")
            try:
                values = tuple(float(row[name]) for name in names)
            except (TypeError, ValueError) as error:
                raise ValueError(f"non-numeric covariate value for {key!r}") from error
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite covariate value for {key!r}")
            output[key] = values
    if not output:
        raise ValueError("covariate TSV contains no rows")
    return output


def _hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> float | None:
    degrees = len(group_a) + len(group_b) - 2
    if degrees <= 0:
        return None
    pooled_variance = (
        (len(group_a) - 1) * group_a.var(ddof=1) + (len(group_b) - 1) * group_b.var(ddof=1)
    ) / degrees
    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        return None
    correction = 1.0 - 3.0 / (4.0 * (len(group_a) + len(group_b)) - 9.0)
    return float(correction * (group_a.mean() - group_b.mean()) / np.sqrt(pooled_variance))


def infer_independent_endpoints(
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    *,
    group_a: str,
    group_b: str,
    fdr_family: str,
    alpha: float = 0.05,
    within_group_aggregation: str = "mean",
    covariates: dict[tuple[str, str], tuple[float, ...]] | None = None,
    covariate_names: tuple[str, ...] = (),
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compare group A minus group B for every shared endpoint with HC3 OLS."""
    if not group_a or not group_b or group_a == group_b:
        raise ValueError("group labels must be distinct non-empty strings")
    if not fdr_family.strip():
        raise ValueError("fdr_family must be a non-empty prespecified family")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be a number")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and strictly between zero and one")
    names = tuple(str(name) for name in covariate_names)
    if (covariates is None) != (not names):
        raise ValueError("covariates and covariate_names must be supplied together")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("covariate_names must be unique and non-empty")
    values_a, metadata_a, counts_a = _collect_subject_endpoints(
        payload_a, aggregation=within_group_aggregation
    )
    values_b, metadata_b, counts_b = _collect_subject_endpoints(
        payload_b, aggregation=within_group_aggregation
    )
    endpoints = sorted(set(metadata_a) | set(metadata_b))
    results: list[dict[str, Any]] = []
    for endpoint in endpoints:
        if endpoint in metadata_a and endpoint in metadata_b:
            if metadata_a[endpoint] != metadata_b[endpoint]:
                raise ValueError(f"endpoint {endpoint!r} metadata differs between groups")
            metadata = metadata_a[endpoint]
        else:
            metadata = metadata_a.get(endpoint, metadata_b.get(endpoint, {}))
        subjects_a = sorted(values_a.get(endpoint, {}))
        subjects_b = sorted(values_b.get(endpoint, {}))
        if covariates is not None:
            subjects_a = [s for s in subjects_a if (group_a, s) in covariates]
            subjects_b = [s for s in subjects_b if (group_b, s) in covariates]
        base = {
            "endpoint": endpoint,
            **metadata,
            "n": len(subjects_a) + len(subjects_b),
            "n_group_a": len(subjects_a),
            "n_group_b": len(subjects_b),
            "estimate": None,
            "standardized_estimate": None,
            "standard_error": None,
            "statistic": None,
            "residual_degrees_of_freedom": None,
            "ci_low": None,
            "ci_high": None,
            "p": None,
            "q": None,
            "direction": None,
            "result_status": "not_testable",
            "reason": None,
        }
        minimum_per_group = max(2, len(names) + 1)
        if len(subjects_a) < minimum_per_group or len(subjects_b) < minimum_per_group:
            results.append(
                {
                    **base,
                    "reason": "insufficient complete participants in at least one group",
                }
            )
            continue
        group_a_values = np.asarray([values_a[endpoint][s] for s in subjects_a])
        group_b_values = np.asarray([values_b[endpoint][s] for s in subjects_b])
        outcome = np.concatenate((group_a_values, group_b_values))
        indicator = np.concatenate((np.ones(len(subjects_a)), np.zeros(len(subjects_b))))
        design_parts = [np.ones(len(outcome)), indicator]
        if covariates is not None:
            nuisance = np.asarray(
                [covariates[(group_a, s)] for s in subjects_a]
                + [covariates[(group_b, s)] for s in subjects_b],
                dtype=float,
            )
            nuisance_sd = nuisance.std(axis=0, ddof=1)
            if np.any(nuisance_sd <= 0.0) or not np.isfinite(nuisance_sd).all():
                raise ValueError("covariates must vary across the combined analysis sample")
            nuisance = (nuisance - nuisance.mean(axis=0)) / nuisance_sd
            design_parts.extend(nuisance.T)
        design = np.column_stack(design_parts)
        column_names = ("intercept", "group", *names)
        try:
            model = ols_hc3(outcome, design, column_names=column_names)
            reduced = ols_hc3(
                outcome,
                design[:, [0, *range(2, design.shape[1])]],
                column_names=("intercept", *names),
            )
        except ValueError as error:
            results.append({**base, "reason": str(error)})
            continue
        group_index = model.column_names.index("group")
        estimate = float(model.coefficients[group_index])
        interval_low, interval_high = hc3_confidence_interval(
            model,
            confidence=confidence,
        )
        outcome_sd = float(outcome.std(ddof=1))
        standardized = None if outcome_sd <= 0.0 else estimate / outcome_sd
        pvalue = float(model.pvalues[group_index])
        results.append(
            {
                **base,
                "group_a_mean": float(group_a_values.mean()),
                "group_b_mean": float(group_b_values.mean()),
                "estimate": estimate,
                "standardized_estimate": standardized,
                "standard_error": float(model.standard_errors[group_index]),
                "statistic": float(model.statistics[group_index]),
                "residual_degrees_of_freedom": model.degrees_of_freedom,
                "hedges_g": _hedges_g(group_a_values, group_b_values),
                "full_model_r2": model.r_squared,
                "group_incremental_r2": (
                    None
                    if model.r_squared is None or reduced.r_squared is None
                    else float(model.r_squared - reduced.r_squared)
                ),
                "ci_low": float(interval_low[group_index]),
                "ci_high": float(interval_high[group_index]),
                "p": pvalue,
                "q": None,
                "direction": (
                    f"{group_a}_higher"
                    if estimate > 0
                    else f"{group_a}_lower"
                    if estimate < 0
                    else "no_difference"
                ),
                "result_status": None,
                "reason": None,
                "group_a_acquisitions_max": counts_a.get(endpoint, 0),
                "group_b_acquisitions_max": counts_b.get(endpoint, 0),
            }
        )
    n_tested = _adjust_endpoint_pvalues(results, family=fdr_family, alpha=alpha)
    return {
        "format": "dfc-kit-independent-endpoint-inference",
        "format_version": 1,
        "source_format_a": payload_a.get("format"),
        "source_format_b": payload_b.get("format"),
        "source_contract_a": payload_a.get("source_contract"),
        "source_contract_b": payload_b.get("source_contract"),
        "contrast": f"{group_a} - {group_b}",
        "group_a": group_a,
        "group_b": group_b,
        "test": "independent participant-level OLS with HC3 standard errors",
        "covariate_names": list(names),
        "covariate_scaling": "combined-sample mean and sample standard deviation",
        "fdr_family": fdr_family,
        "alpha": alpha,
        "confidence": float(confidence),
        "confidence_interval": "two-sided Student-t interval using HC3 standard error",
        "within_group_aggregation": within_group_aggregation,
        "n_endpoints": len(results),
        "n_tested": n_tested,
        "results": results,
    }


def infer_independent_endpoints_file(
    path_a: str | Path,
    path_b: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payloads = []
    for path in (path_a, path_b):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("endpoint input must be a JSON object")
        payloads.append(raw)
    return infer_independent_endpoints(payloads[0], payloads[1], **kwargs)


def write_independent_endpoint_inference(payload: dict[str, Any], path: str | Path) -> Path:
    return _write_paired_endpoint_inference(payload, path)


__all__ = [
    "infer_independent_endpoints",
    "infer_independent_endpoints_file",
    "load_group_covariates",
    "write_independent_endpoint_inference",
]
