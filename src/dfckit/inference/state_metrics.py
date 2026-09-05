"""Paired inference over machine-readable state-dynamics metrics."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .endpoints import _adjust_endpoint_pvalues, _paired_endpoint_result

SUPPORTED_STATE_METRICS = frozenset(
    {"occupancy", "mean_dwell_seconds", "switch_rate", "transition_probabilities"}
)


def _load_metrics(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("runs"), list):
        raise TypeError("state metrics must be a dfc-kit state-metrics JSON object")
    n_states = raw.get("n_states")
    if isinstance(n_states, bool) or not isinstance(n_states, int) or n_states < 2:
        raise ValueError("state metrics n_states is invalid")
    return raw


def _endpoints(record: dict[str, Any], metrics: tuple[str, ...], n_states: int) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric in metrics:
        value = record.get(metric)
        if metric == "switch_rate":
            if value is not None:
                output[metric] = float(value)
            continue
        if metric == "transition_probabilities":
            if not isinstance(value, list) or len(value) != n_states:
                raise ValueError("transition_probabilities has an invalid state axis")
            for source, row in enumerate(value):
                if not isinstance(row, list) or len(row) != n_states:
                    raise ValueError("transition_probabilities has an invalid target axis")
                for target, item in enumerate(row):
                    if item is not None:
                        output[f"{metric}.state_{source}_to_{target}"] = float(item)
            continue
        if not isinstance(value, list) or len(value) != n_states:
            raise ValueError(f"{metric} has an invalid state axis")
        for state, item in enumerate(value):
            if item is not None:
                output[f"{metric}.state_{state}"] = float(item)
    if any(not np.isfinite(value) for value in output.values()):
        raise ValueError("state metrics contain non-finite endpoint values")
    return output


def infer_paired_state_metrics(
    metrics_payload: dict[str, Any],
    *,
    condition_a: str,
    condition_b: str,
    metrics: tuple[str, ...],
    fdr_family: str,
    alpha: float = 0.05,
    n_permutations: int = 10_000,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    exact: bool = False,
    within_condition_aggregation: str = "error",
) -> dict[str, Any]:
    """Test condition A minus condition B for each selected state endpoint."""
    if not condition_a or not condition_b or condition_a == condition_b:
        raise ValueError("paired conditions must be distinct non-empty labels")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be a non-empty unique sequence")
    unsupported = sorted(set(metrics) - SUPPORTED_STATE_METRICS)
    if unsupported:
        raise ValueError(f"unsupported state metrics: {unsupported}")
    if not fdr_family.strip():
        raise ValueError("fdr_family must be a non-empty prespecified family")
    if within_condition_aggregation not in {"error", "mean"}:
        raise ValueError("within_condition_aggregation must be 'error' or 'mean'")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    n_states = int(metrics_payload["n_states"])
    by_endpoint: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for raw in metrics_payload["runs"]:
        if not isinstance(raw, dict):
            raise TypeError("state metrics runs must contain JSON objects")
        subject = raw.get("subject")
        session = raw.get("session")
        if not isinstance(subject, str) or not isinstance(session, str):
            raise TypeError("paired state inference requires subject and session labels")
        if session not in {condition_a, condition_b}:
            continue
        acquisition_id = raw.get("acquisition_id")
        acquisition_key = (
            acquisition_id
            if isinstance(acquisition_id, str) and acquisition_id
            else "<unspecified>"
        )
        for endpoint, value in _endpoints(raw, metrics, n_states).items():
            acquisitions = (
                by_endpoint.setdefault(endpoint, {}).setdefault(subject, {}).setdefault(session, {})
            )
            if acquisitions and within_condition_aggregation == "error":
                raise ValueError(
                    f"subject {subject!r} has duplicate runs for condition {session!r}"
                )
            if acquisition_key in acquisitions:
                raise ValueError(
                    f"subject {subject!r} condition {session!r} endpoint {endpoint!r} "
                    f"has duplicate acquisition {acquisition_key!r}"
                )
            acquisitions[acquisition_key] = value

    endpoint_names = []
    for metric in metrics:
        if metric == "switch_rate":
            endpoint_names.append(metric)
        elif metric == "transition_probabilities":
            endpoint_names.extend(
                f"{metric}.state_{source}_to_{target}"
                for source in range(n_states)
                for target in range(n_states)
            )
        else:
            endpoint_names.extend(f"{metric}.state_{state}" for state in range(n_states))
    endpoint_names.sort()
    # Include absent canonical endpoints in the seed axis, as in the state contract.
    results = [
        _paired_endpoint_result(
            by_endpoint.get(endpoint, {}),
            endpoint=endpoint,
            condition_a=condition_a,
            condition_b=condition_b,
            endpoint_index=endpoint_index,
            seed=seed,
            exact=exact,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
            errors_as_not_testable=True,
        )
        for endpoint_index, endpoint in enumerate(endpoint_names)
    ]
    n_tested = _adjust_endpoint_pvalues(results, family=fdr_family, alpha=alpha)
    return {
        "format": "dfc-kit-paired-state-inference",
        "format_version": 1,
        "model_kind": metrics_payload.get("model_kind"),
        "source_contract": metrics_payload.get("source_contract"),
        "contrast": f"{condition_a} - {condition_b}",
        "condition_a": condition_a,
        "condition_b": condition_b,
        "metrics": list(metrics),
        "test": "paired participant-level sign flip",
        "confidence_interval": "participant bootstrap percentile",
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "exact": exact,
        "within_condition_aggregation": within_condition_aggregation,
        "fdr_family": fdr_family,
        "fdr_method": "benjamini-hochberg",
        "alpha": alpha,
        "n_endpoints": len(results),
        "n_tested": n_tested,
        "results": results,
    }


def _infer_paired_state_metrics_file(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    return infer_paired_state_metrics(_load_metrics(path), **kwargs)


def _write_paired_state_inference(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"state-inference output already exists: {target}")
    target.mkdir(parents=True)
    try:
        (target / "results.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        fields = (
            "endpoint",
            "n",
            "estimate",
            "ci_low",
            "ci_high",
            "p",
            "q",
            "direction",
            "condition_a_acquisitions_min",
            "condition_a_acquisitions_max",
            "condition_b_acquisitions_min",
            "condition_b_acquisitions_max",
            "result_status",
            "reason",
        )
        temporary = target / f".results.tsv.tmp-{os.getpid()}"
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(
                {field: row.get(field) for field in fields} for row in payload["results"]
            )
        os.replace(temporary, target / "results.tsv")
    except BaseException:
        for child in target.iterdir():
            child.unlink(missing_ok=True)
        target.rmdir()
        raise
    return target


__all__ = [
    "SUPPORTED_STATE_METRICS",
    "infer_paired_state_metrics",
]
