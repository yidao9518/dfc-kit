"""Machine-readable paired NBS over edge FeatureStore summaries."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .nbs import PairedNBSResult, paired_nbs


def _edge_index(endpoint: str, statistic: str) -> int:
    prefix = "feature_"
    suffix = f".{statistic}"
    if not endpoint.startswith(prefix) or not endpoint.endswith(suffix):
        raise ValueError(
            f"NBS endpoint {endpoint!r} does not match feature_<index>.{statistic}"
        )
    raw = endpoint[len(prefix) : -len(suffix)]
    if not raw.isdigit() or str(int(raw)) != raw:
        raise ValueError(f"NBS endpoint {endpoint!r} has a non-canonical edge index")
    return int(raw)


def _edge_definition(
    payload: Mapping[str, Any], statistic: str
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    if payload.get("format") != "dfc-kit-store-endpoints":
        raise ValueError("NBS requires a dfc-kit-store-endpoints artifact")
    if payload.get("feature_type") != "edge":
        raise ValueError("NBS requires edge features")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise TypeError("NBS endpoint rows must be a non-empty array")
    indexed: dict[int, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("statistic") != statistic:
            continue
        endpoint = row.get("endpoint")
        feature = row.get("feature")
        if not isinstance(endpoint, str):
            raise TypeError("NBS endpoint names must be strings")
        if (
            not isinstance(feature, list)
            or len(feature) != 2
            or any(not isinstance(node, str) or not node for node in feature)
            or feature[0] == feature[1]
        ):
            raise ValueError("every NBS feature must name two distinct nodes")
        index = _edge_index(endpoint, statistic)
        edge = (feature[0], feature[1])
        if index in indexed and indexed[index] != edge:
            raise ValueError(f"NBS edge definition changes at feature index {index}")
        indexed[index] = edge
    n_features = payload.get("n_features")
    if (
        isinstance(n_features, bool)
        or not isinstance(n_features, int)
        or n_features < 1
        or set(indexed) != set(range(n_features))
    ):
        raise ValueError("NBS summary must contain every indexed edge for the statistic")
    edges = tuple(indexed[index] for index in range(n_features))
    nodes = tuple(dict.fromkeys(node for edge in edges for node in edge))
    expected = tuple(combinations(nodes, 2))
    if edges != expected:
        raise ValueError(
            "NBS edge features must be one complete upper-triangle graph in ROI order"
        )
    return nodes, edges


def _paired_edge_matrix(
    payload: Mapping[str, Any],
    *,
    statistic: str,
    condition_a: str,
    condition_b: str,
    within_condition_aggregation: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], np.ndarray, tuple[int, int, int, int]]:
    if not condition_a or not condition_b or condition_a == condition_b:
        raise ValueError("paired NBS conditions must be distinct non-empty labels")
    if within_condition_aggregation not in {"error", "mean"}:
        raise ValueError("within_condition_aggregation must be 'error' or 'mean'")
    _nodes, edges = _edge_definition(payload, statistic)
    rows = payload["rows"]
    cells: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("statistic") != statistic:
            continue
        subject = row.get("subject")
        session = row.get("session")
        endpoint = row.get("endpoint")
        acquisition = row.get("acquisition_id")
        value = row.get("value")
        if session not in {condition_a, condition_b}:
            continue
        if not isinstance(subject, str) or not subject:
            raise TypeError("NBS endpoint rows require subject strings")
        if not isinstance(session, str) or not isinstance(endpoint, str):
            raise TypeError("NBS endpoint rows require session and endpoint strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("NBS endpoint values must be finite numbers")
        acquisition_key = acquisition if isinstance(acquisition, str) and acquisition else "<unspecified>"
        index = _edge_index(endpoint, statistic)
        per_condition = cells.setdefault(subject, {}).setdefault(session, {})
        per_acquisition = per_condition.setdefault(acquisition_key, {})
        if index in per_acquisition:
            raise ValueError(
                f"duplicate NBS edge {index} for {subject!r} {session!r} {acquisition_key!r}"
            )
        per_acquisition[index] = float(value)
    subjects = tuple(
        subject
        for subject in sorted(cells)
        if condition_a in cells[subject] and condition_b in cells[subject]
    )
    if len(subjects) < 2:
        raise ValueError("paired NBS requires at least two complete participant pairs")
    n_edges = len(edges)
    differences = []
    counts_a = []
    counts_b = []
    for subject in subjects:
        condition_vectors: dict[str, np.ndarray] = {}
        for condition, counts in ((condition_a, counts_a), (condition_b, counts_b)):
            acquisitions = cells[subject][condition]
            if len(acquisitions) > 1 and within_condition_aggregation == "error":
                raise ValueError(
                    f"subject {subject!r} has duplicate acquisitions for condition {condition!r}"
                )
            vectors = []
            for acquisition, values in sorted(acquisitions.items()):
                if set(values) != set(range(n_edges)):
                    raise ValueError(
                        f"subject {subject!r} acquisition {acquisition!r} does not contain the complete edge graph"
                    )
                vectors.append(np.asarray([values[index] for index in range(n_edges)]))
            counts.append(len(vectors))
            condition_vectors[condition] = np.mean(vectors, axis=0)
        differences.append(condition_vectors[condition_a] - condition_vectors[condition_b])
    return (
        subjects,
        edges,
        np.asarray(differences, dtype=float),
        (min(counts_a), max(counts_a), min(counts_b), max(counts_b)),
    )


def load_nbs_confounds(
    path: str | Path,
    names: Sequence[str],
) -> dict[str, tuple[float, ...]]:
    """Load a strict subject-keyed TSV of numeric nuisance variables."""
    selected = tuple(str(name) for name in names)
    if not selected or any(not name or name == "subject" for name in selected):
        raise ValueError("confound names must be non-empty and cannot be 'subject'")
    if len(set(selected)) != len(selected):
        raise ValueError("confound names must be unique")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["subject", *selected]:
            raise ValueError(
                f"NBS confounds header must be exactly {['subject', *selected]}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("NBS confounds table is empty")
    output: dict[str, tuple[float, ...]] = {}
    for row_number, row in enumerate(rows, start=2):
        subject = row["subject"]
        if not subject or subject in output:
            raise ValueError(f"invalid or duplicate subject at confounds row {row_number}")
        try:
            values = tuple(float(row[name]) for name in selected)
        except (TypeError, ValueError) as error:
            raise ValueError(f"non-numeric confound at row {row_number}") from error
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite confound at row {row_number}")
        output[subject] = values
    return output


def _component_payload(
    result: PairedNBSResult,
    *,
    alpha: float,
    node_names: tuple[str, ...],
    edge_names: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold_result in result.threshold_results:
        components = threshold_result.components
        if not components:
            rows.append(
                {
                    "endpoint": f"threshold_{threshold_result.threshold:g}.no_component",
                    "threshold": threshold_result.threshold,
                    "component_index": None,
                    "component_direction": "none",
                    "component_nodes": [],
                    "component_edges": [],
                    "n": len(result.subject_ids),
                    "estimate": 0.0,
                    "ci_low": None,
                    "ci_high": None,
                    "p": 1.0,
                    "q": 1.0,
                    "direction": "no_suprathreshold_component",
                    "result_status": "negative",
                    "reason": "no supra-threshold component",
                }
            )
            continue
        for component_index, component in enumerate(components):
            assert component.fwe_pvalue is not None
            rows.append(
                {
                    "endpoint": (
                        f"threshold_{threshold_result.threshold:g}."
                        f"{component.direction}.component_{component_index}"
                    ),
                    "threshold": threshold_result.threshold,
                    "component_index": component_index,
                    "component_direction": component.direction,
                    "component_nodes": [node_names[index] for index in component.node_indices],
                    "component_edges": [list(edge_names[index]) for index in component.edge_indices],
                    "edge_indices": list(component.edge_indices),
                    "n": len(result.subject_ids),
                    "estimate": component.statistic_value,
                    "ci_low": None,
                    "ci_high": None,
                    "p": component.fwe_pvalue,
                    "q": component.fwe_pvalue,
                    "direction": component.direction,
                    "result_status": (
                        "positive" if component.fwe_pvalue < alpha else "negative"
                    ),
                    "reason": None,
                }
            )
    return rows


def infer_paired_nbs_endpoints(
    payload: Mapping[str, Any],
    *,
    condition_a: str,
    condition_b: str,
    statistic: str,
    thresholds: Sequence[float],
    n_permutations: int,
    seed: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    component_statistic: str = "edge_extent",
    component_sign_mode: str = "separate",
    within_condition_aggregation: str = "error",
    confounds_by_subject: Mapping[str, Sequence[float]] | None = None,
    confound_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Convert edge summaries into paired differences and run component-level NBS."""
    if not isinstance(statistic, str) or not statistic:
        raise ValueError("statistic must be a non-empty name")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    subjects, edges, differences, acquisition_counts = _paired_edge_matrix(
        payload,
        statistic=statistic,
        condition_a=condition_a,
        condition_b=condition_b,
        within_condition_aggregation=within_condition_aggregation,
    )
    node_names = tuple(dict.fromkeys(node for edge in edges for node in edge))
    confounds = None
    names = tuple(str(name) for name in confound_names)
    if confounds_by_subject is not None:
        missing = [subject for subject in subjects if subject not in confounds_by_subject]
        if missing:
            raise ValueError(f"NBS confounds are missing paired subjects: {missing[:10]}")
        confounds = np.asarray([confounds_by_subject[subject] for subject in subjects], dtype=float)
    result = paired_nbs(
        differences,
        subjects,
        np.asarray([node_names.index(edge[0]) for edge in edges]),
        np.asarray([node_names.index(edge[1]) for edge in edges]),
        len(node_names),
        thresholds=thresholds,
        n_permutations=n_permutations,
        seed=seed,
        difference_direction=f"{condition_a} - {condition_b}",
        alternative=alternative,
        component_statistic=component_statistic,
        component_sign_mode=component_sign_mode,
        confounds=confounds,
        confound_names=names,
    )
    threshold_payload = []
    for item in result.threshold_results:
        threshold_payload.append(
            {
                "threshold": item.threshold,
                "observed_t": item.observed_t.tolist(),
                "observed_mean_difference": item.observed_mean_difference.tolist(),
                "null_maximum": item.null_maximum.tolist(),
                "fwe_scope": item.fwe_scope,
            }
        )
    return {
        "format": "dfc-kit-paired-nbs",
        "format_version": 1,
        "source_format": payload.get("format"),
        "source_contract": payload.get("source_contract"),
        "contrast": f"{condition_a} - {condition_b}",
        "condition_a": condition_a,
        "condition_b": condition_b,
        "statistic": statistic,
        "node_names": list(node_names),
        "n_nodes": len(node_names),
        "edges": [list(edge) for edge in edges],
        "n_edges": len(edges),
        "subject_ids": list(subjects),
        "n_subjects": len(subjects),
        "thresholds": [item.threshold for item in result.threshold_results],
        "alternative": result.alternative,
        "component_statistic": result.component_statistic,
        "component_sign_mode": result.component_sign_mode,
        "n_permutations": result.n_permutations,
        "seed": result.seed,
        "permutation_unit": result.permutation_unit,
        "permutation_method": result.permutation_method,
        "threshold_correction": result.threshold_correction,
        "confound_names": list(result.confound_names),
        "within_condition_aggregation": within_condition_aggregation,
        "condition_a_acquisitions_min": acquisition_counts[0],
        "condition_a_acquisitions_max": acquisition_counts[1],
        "condition_b_acquisitions_min": acquisition_counts[2],
        "condition_b_acquisitions_max": acquisition_counts[3],
        "alpha": float(alpha),
        "correction_method": "network-based-statistic component FWE within each threshold",
        "threshold_results": threshold_payload,
        "results": _component_payload(
            result,
            alpha=float(alpha),
            node_names=node_names,
            edge_names=edges,
        ),
    }


def infer_paired_nbs_file(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("NBS endpoint input must be a JSON object")
    return infer_paired_nbs_endpoints(raw, **kwargs)


__all__ = [
    "infer_paired_nbs_endpoints",
    "infer_paired_nbs_file",
    "load_nbs_confounds",
]
