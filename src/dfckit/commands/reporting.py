"""Command handlers that summarize, interpret, or infer from existing artifacts."""

from __future__ import annotations

import argparse

from ..artifacts import load_state_model_scores
from ..inference.endpoints import (
    _infer_paired_endpoints_file,
    _write_paired_endpoint_inference,
)
from ..inference.state_metrics import (
    _infer_paired_state_metrics_file,
    _write_paired_state_inference,
)
from ..information.summary import summarize_information_artifact, write_information_summary
from ..states import compare_state_model_scores
from ..states.interpretation import (
    describe_state_artifacts,
    write_state_description,
)
from ..states.selection import _write_state_count_comparison
from ..storage.summary import _summarize_store_file, _write_store_summary


def describe_states(namespace: argparse.Namespace) -> dict[str, object]:
    """Describe named features that distinguish fitted state patterns."""
    payload = describe_state_artifacts(
        namespace.store,
        namespace.model,
        top_features=namespace.top_features,
        network_map_path=namespace.network_map,
    )
    output = write_state_description(payload, namespace.output)
    return {
        "output": str(output),
        "model_kind": payload["model_kind"],
        "feature_type": payload["feature_type"],
        "n_states": payload["n_states"],
        "n_features": payload["n_features"],
        "network_map_applied": payload["network_map_applied"],
    }


def infer_state_metrics(namespace: argparse.Namespace) -> dict[str, object]:
    """Run paired inference over selected state metrics."""
    payload = _infer_paired_state_metrics_file(
        namespace.metrics,
        condition_a=namespace.condition_a,
        condition_b=namespace.condition_b,
        metrics=tuple(namespace.metric),
        fdr_family=namespace.fdr_family,
        alpha=namespace.alpha,
        n_permutations=namespace.permutations,
        n_bootstrap=namespace.bootstrap,
        seed=namespace.seed,
        exact=namespace.exact,
    )
    output = _write_paired_state_inference(payload, namespace.output)
    counts = {
        status: sum(result["result_status"] == status for result in payload["results"])
        for status in ("positive", "negative", "not_testable")
    }
    return {
        "output": str(output),
        "contrast": payload["contrast"],
        "fdr_family": payload["fdr_family"],
        "n_endpoints": payload["n_endpoints"],
        "n_tested": payload["n_tested"],
        "status_counts": counts,
    }


def summarize_store(namespace: argparse.Namespace) -> dict[str, object]:
    """Write acquisition-level feature statistics from a FeatureStore."""
    statistics = tuple(namespace.statistic or ("mean",))
    payload = _summarize_store_file(namespace.store, statistics)
    output = _write_store_summary(payload, namespace.output)
    return {
        "output": str(output),
        "source_contract": payload["source_contract"],
        "feature_type": payload["feature_type"],
        "n_features": payload["n_features"],
        "n_acquisitions": payload["n_acquisitions"],
        "statistics": payload["statistics"],
    }


def infer_endpoints(namespace: argparse.Namespace) -> dict[str, object]:
    """Run paired inference over named acquisition-level endpoints."""
    payload = _infer_paired_endpoints_file(
        namespace.endpoints,
        condition_a=namespace.condition_a,
        condition_b=namespace.condition_b,
        fdr_family=namespace.fdr_family,
        alpha=namespace.alpha,
        n_permutations=namespace.permutations,
        n_bootstrap=namespace.bootstrap,
        seed=namespace.seed,
        exact=namespace.exact,
    )
    output = _write_paired_endpoint_inference(payload, namespace.output)
    return {
        "output": str(output),
        "contrast": payload["contrast"],
        "fdr_family": payload["fdr_family"],
        "n_endpoints": payload["n_endpoints"],
        "n_tested": payload["n_tested"],
    }


def summarize_information(namespace: argparse.Namespace) -> dict[str, object]:
    """Extract acquisition-level endpoints from a fixed-information artifact."""
    payload = summarize_information_artifact(namespace.artifact)
    output = write_information_summary(payload, namespace.output)
    return {
        "output": str(output),
        "lengths": payload["lengths"],
        "groups": payload["groups"],
        "n_endpoints": len(payload["rows"]),
    }


def compare_state_counts(namespace: argparse.Namespace) -> dict[str, object]:
    """Aggregate complete held-out score grids into a state-count comparison."""
    if namespace.output.exists() or namespace.output.is_symlink():
        raise FileExistsError(
            f"state-count comparison output path already exists: {namespace.output}"
        )
    reports = tuple(load_state_model_scores(path) for path in namespace.score)
    comparison = compare_state_model_scores(
        reports,
    )
    output = _write_state_count_comparison(comparison, namespace.output)
    return {
        "output": str(output),
        "model_kind": comparison.model_kind,
        "n_folds": comparison.selection.n_folds,
        "n_candidates": comparison.selection.n_candidates,
        "candidate_n_states": comparison.selection.candidate_n_states.tolist(),
        "n_seeds": len(comparison.model_seeds),
        "model_seeds": list(comparison.model_seeds),
        "n_subjects": len(comparison.subjects),
        "best_n_states": comparison.selection.best_n_states,
        "one_standard_error_n_states": comparison.selection.one_standard_error_n_states,
    }
