"""Paired inference over machine-readable state-dynamics metrics."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .multiple_testing import benjamini_hochberg
from .paired import paired_bootstrap_mean_ci, paired_sign_flip

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
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    n_states = int(metrics_payload["n_states"])
    by_subject: dict[str, dict[str, dict[str, float]]] = {}
    for raw in metrics_payload["runs"]:
        if not isinstance(raw, dict):
            raise TypeError("state metrics runs must contain JSON objects")
        subject = raw.get("subject")
        session = raw.get("session")
        if not isinstance(subject, str) or not isinstance(session, str):
            raise TypeError("paired state inference requires subject and session labels")
        if session not in {condition_a, condition_b}:
            continue
        if session in by_subject.setdefault(subject, {}):
            raise ValueError(f"subject {subject!r} has duplicate runs for condition {session!r}")
        by_subject[subject][session] = _endpoints(raw, metrics, n_states)

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
    results: list[dict[str, Any]] = []
    tested_indices: list[int] = []
    pvalues: list[float] = []
    for endpoint_index, endpoint in enumerate(endpoint_names):
        subjects = tuple(
            subject
            for subject in sorted(by_subject)
            if endpoint in by_subject[subject].get(condition_a, {})
            and endpoint in by_subject[subject].get(condition_b, {})
        )
        if len(subjects) < 2:
            results.append(
                {
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
            )
            continue
        differences = np.asarray(
            [
                by_subject[subject][condition_a][endpoint]
                - by_subject[subject][condition_b][endpoint]
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
            results.append(
                {
                    "endpoint": endpoint,
                    "n": len(subjects),
                    "estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "p": None,
                    "q": None,
                    "direction": None,
                    "result_status": "not_testable",
                    "reason": str(error),
                }
            )
            continue
        estimate = test.estimate
        results.append(
            {
                "endpoint": endpoint,
                "n": len(subjects),
                "estimate": estimate,
                "ci_low": interval.lower,
                "ci_high": interval.upper,
                "p": test.pvalue,
                "q": None,
                "direction": (
                    f"{condition_a}_higher"
                    if estimate > 0.0
                    else f"{condition_a}_lower"
                    if estimate < 0.0
                    else "no_difference"
                ),
                "result_status": None,
                "reason": None,
            }
        )
        tested_indices.append(len(results) - 1)
        pvalues.append(test.pvalue)
    if pvalues:
        correction = benjamini_hochberg(pvalues, family=fdr_family)
        for result_index, qvalue in zip(
            tested_indices, correction.adjusted_pvalues, strict=True
        ):
            results[result_index]["q"] = float(qvalue)
            results[result_index]["result_status"] = (
                "positive" if qvalue < alpha else "negative"
            )
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
        "fdr_family": fdr_family,
        "fdr_method": "benjamini-hochberg",
        "alpha": alpha,
        "n_endpoints": len(results),
        "n_tested": len(tested_indices),
        "results": results,
    }


def infer_paired_state_metrics_file(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    return infer_paired_state_metrics(_load_metrics(path), **kwargs)


def write_paired_state_inference(payload: dict[str, Any], path: str | Path) -> Path:
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
    "infer_paired_state_metrics_file",
    "write_paired_state_inference",
]
