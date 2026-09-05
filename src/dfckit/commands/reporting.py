"""Command handlers that summarize, interpret, or infer from existing artifacts."""

from __future__ import annotations

import argparse

from ..artifacts import load_state_model_scores
from ..inference.endpoints import (
    _infer_paired_endpoints_file,
    _write_paired_endpoint_inference,
)
from ..inference.independent_endpoints import (
    infer_independent_endpoints_file,
    load_group_covariates,
    write_independent_endpoint_inference,
)
from ..inference.nbs_endpoints import (
    infer_paired_nbs_file,
    load_nbs_confounds,
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
        within_condition_aggregation=namespace.within_condition_aggregation,
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
        endpoint_names=namespace.endpoint,
        alpha=namespace.alpha,
        n_permutations=namespace.permutations,
        n_bootstrap=namespace.bootstrap,
        seed=namespace.seed,
        exact=namespace.exact,
        within_condition_aggregation=namespace.within_condition_aggregation,
    )
    output = _write_paired_endpoint_inference(payload, namespace.output)
    return {
        "output": str(output),
        "contrast": payload["contrast"],
        "fdr_family": payload["fdr_family"],
        "endpoint_selection": payload["endpoint_selection"],
        "n_endpoints": payload["n_endpoints"],
        "n_tested": payload["n_tested"],
    }


def infer_independent_endpoints(namespace: argparse.Namespace) -> dict[str, object]:
    """Run HC3 group models over two independent endpoint cohorts."""
    if (namespace.covariates is None) != (not namespace.covariate):
        raise ValueError("--covariates and at least one --covariate must be supplied together")
    names = tuple(namespace.covariate)
    covariates = (
        None
        if namespace.covariates is None
        else load_group_covariates(
            namespace.covariates,
            covariate_names=names,
            group_column=namespace.group_column,
            subject_column=namespace.subject_column,
        )
    )
    payload = infer_independent_endpoints_file(
        namespace.endpoints_a,
        namespace.endpoints_b,
        group_a=namespace.group_a,
        group_b=namespace.group_b,
        fdr_family=namespace.fdr_family,
        alpha=namespace.alpha,
        within_group_aggregation=namespace.within_group_aggregation,
        covariates=covariates,
        covariate_names=names,
    )
    output = write_independent_endpoint_inference(payload, namespace.output)
    return {
        "output": str(output),
        "contrast": payload["contrast"],
        "fdr_family": payload["fdr_family"],
        "n_endpoints": payload["n_endpoints"],
        "n_tested": payload["n_tested"],
    }


def infer_nbs(namespace: argparse.Namespace) -> dict[str, object]:
    """Run paired NBS over one edge statistic from a store summary."""
    if (namespace.confounds is None) != (not namespace.confound):
        raise ValueError("--confounds and at least one --confound must be supplied together")
    confounds = (
        None
        if namespace.confounds is None
        else load_nbs_confounds(namespace.confounds, tuple(namespace.confound))
    )
    payload = infer_paired_nbs_file(
        namespace.endpoints,
        condition_a=namespace.condition_a,
        condition_b=namespace.condition_b,
        statistic=namespace.statistic,
        thresholds=tuple(namespace.threshold),
        n_permutations=namespace.permutations,
        seed=namespace.seed,
        alpha=namespace.alpha,
        alternative=namespace.alternative,
        component_statistic=namespace.component_statistic,
        component_sign_mode=namespace.component_sign_mode,
        within_condition_aggregation=namespace.within_condition_aggregation,
        confounds_by_subject=confounds,
        confound_names=tuple(namespace.confound),
    )
    output = _write_paired_endpoint_inference(payload, namespace.output)
    return {
        "output": str(output),
        "contrast": payload["contrast"],
        "n_subjects": payload["n_subjects"],
        "n_nodes": len(payload["node_names"]),
        "n_edges": len(payload["edges"]),
        "n_components": len(payload["results"]),
        "thresholds": payload["thresholds"],
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
