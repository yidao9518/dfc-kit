"""Command handlers for fitting, decoding, summarizing, and scoring state models."""

from __future__ import annotations

import argparse

from ..artifacts import (
    StatePredictions,
    load_fitted_model,
    load_state_predictions,
    save_fitted_model,
    save_state_predictions,
    write_state_metrics,
    write_state_model_scores,
)
from ..artifacts.state_scoring import _state_model_specification
from ..states.hmm import GaussianHMMStateModel
from ..states.kmeans import KMeansStateModel
from ..states.metrics import summarize_state_assignments
from ..states.streaming import (
    fit_kmeans_store_materialized,
    fit_minibatch_kmeans_store,
    predict_kmeans_store,
    score_kmeans_store,
)
from ..states.streaming_hmm import (
    fit_gaussian_hmm_store,
    predict_gaussian_hmm_store,
    score_gaussian_hmm_store,
)
from ..storage import FeatureStore


def fit_states(namespace: argparse.Namespace) -> dict[str, object]:
    """Fit one state model from a disk-backed feature store."""
    store = FeatureStore.open(namespace.store)
    if namespace.output.exists() or namespace.output.is_symlink():
        raise FileExistsError(f"model artifact path already exists: {namespace.output}")
    subjects = None if namespace.subject is None else tuple(namespace.subject)
    if subjects is not None:
        missing = sorted(set(subjects) - set(store.subjects))
        if missing:
            raise ValueError(f"requested fit subjects are absent from the store: {missing}")

    if namespace.method == "kmeans":
        if namespace.fitting_mode == "streaming":
            if namespace.algorithm != "minibatch":
                raise ValueError("streaming KMeans requires --algorithm minibatch")
            fit = fit_minibatch_kmeans_store(
                store,
                n_states=namespace.n_states,
                seed=namespace.seed,
                n_init=10 if namespace.n_init is None else namespace.n_init,
                max_iter=namespace.max_iter,
                batch_size=namespace.batch_size,
                standardize_features=namespace.standardize_features,
                reassignment_ratio=namespace.reassignment_ratio,
                init_sample_size=namespace.init_sample_size,
                n_pca_components=namespace.n_pca_components,
                pca_batch_size=namespace.pca_batch_size,
                subjects=subjects,
                convergence_tol=namespace.streaming_tol,
                convergence_patience=namespace.streaming_patience,
                minimum_passes=namespace.streaming_min_passes,
            )
        else:
            if namespace.init_sample_size is not None:
                raise ValueError("materialized KMeans does not use --init-sample-size")
            fit = fit_kmeans_store_materialized(
                store,
                n_states=namespace.n_states,
                seed=namespace.seed,
                n_init=10 if namespace.n_init is None else namespace.n_init,
                max_iter=namespace.max_iter,
                algorithm=namespace.algorithm,
                batch_size=namespace.batch_size,
                standardize_features=namespace.standardize_features,
                reassignment_ratio=namespace.reassignment_ratio,
                n_pca_components=namespace.n_pca_components,
                subjects=subjects,
            )
        model = fit.model
        fit_sequence_count = len(fit.assignments.sequences)
        artifact = save_fitted_model(model, namespace.output)
        return {
            "model_kind": "kmeans-state",
            "method": namespace.method,
            "fitting_mode": namespace.fitting_mode,
            "output": str(artifact),
            "fit_subjects": list(model.fit_subjects),
            "fit_sample_count": model.fit_sample_count,
            "fit_sequence_count": fit_sequence_count,
            "n_states": model.n_states,
            "seed": model.seed,
            "converged": fit.converged,
            "passes_completed": fit.passes_completed,
            "initialization_passes": list(fit.initialization_passes),
            "log_likelihood": None,
            "inertia": model.inertia,
            "init_sample_size": model.init_sample_size,
            "n_pca_components": model.n_pca_components,
            "pca_batch_size": model.pca_batch_size,
        }

    if namespace.n_pca_components is None:
        raise ValueError("--n-pca-components is required when --method hmm")
    fit = fit_gaussian_hmm_store(
        store,
        n_states=namespace.n_states,
        seed=namespace.seed,
        n_pca_components=namespace.n_pca_components,
        covariance_type=namespace.covariance_type,
        n_init=1 if namespace.n_init is None else namespace.n_init,
        n_iter=namespace.n_iter,
        tol=namespace.tol,
        pca_batch_size=namespace.pca_batch_size,
        subjects=subjects,
        minimum_sequence_length=namespace.minimum_sequence_length,
    )
    model = fit.model
    artifact = save_fitted_model(model, namespace.output)
    return {
        "model_kind": "gaussian-hmm-state",
        "method": namespace.method,
        "output": str(artifact),
        "fit_subjects": list(model.fit_subjects),
        "fit_sample_count": model.fit_sample_count,
        "fit_sequence_count": model.fit_sequence_count,
        "n_states": model.n_states,
        "seed": model.seed,
        "converged": model.converged,
        "log_likelihood": model.log_likelihood,
        "n_pca_components": model.n_pca_components,
        "omitted_short_sequence_count": model.omitted_short_sequence_count,
        "pca_batch_size": model.pca_batch_size,
    }


def selected_store_subjects(
    store: FeatureStore,
    requested: list[str] | None,
) -> tuple[str, ...] | None:
    """Validate and retain an optional store-subject selection."""
    if requested is None:
        return None
    missing = sorted(set(requested) - set(store.subjects))
    if missing:
        raise ValueError(f"requested prediction subjects are absent from the store: {missing}")
    return tuple(requested)


def predict_states(namespace: argparse.Namespace) -> dict[str, object]:
    """Decode a feature store with a fitted state model."""
    store = FeatureStore.open(namespace.store)
    model = load_fitted_model(namespace.model)
    subjects = selected_store_subjects(store, namespace.subject)
    if isinstance(model, KMeansStateModel):
        assignments = predict_kmeans_store(
            model,
            store,
            subjects=subjects,
            allow_fit_subjects=namespace.allow_fit_subjects,
        )
        predictions = StatePredictions(
            assignments=assignments,
            model_kind="kmeans-state",
            model_seed=model.seed,
        )
    elif isinstance(model, GaussianHMMStateModel):
        result = predict_gaussian_hmm_store(
            model,
            store,
            subjects=subjects,
            allow_fit_subjects=namespace.allow_fit_subjects,
        )
        predictions = StatePredictions(
            assignments=result.assignments,
            model_kind="gaussian-hmm-state",
            model_seed=model.seed,
            posterior_probabilities=result.posterior_probabilities,
            log_likelihood=result.log_likelihood,
        )
    else:
        raise TypeError("predict-states requires a KMeans or Gaussian HMM model artifact")
    artifact = save_state_predictions(predictions, namespace.output)
    return {
        "model_kind": predictions.model_kind,
        "output": str(artifact),
        "subjects": list(predictions.subjects),
        "n_sequences": predictions.n_sequences,
        "n_samples": predictions.n_samples,
        "allow_fit_subjects": bool(namespace.allow_fit_subjects),
        "log_likelihood": predictions.log_likelihood,
    }


def summarize_states(namespace: argparse.Namespace) -> dict[str, object]:
    """Write acquisition-level state metrics."""
    predictions = load_state_predictions(namespace.predictions)
    output = write_state_metrics(predictions, namespace.output)
    metrics = summarize_state_assignments(predictions.assignments)
    return {
        "output": str(output),
        "model_kind": predictions.model_kind,
        "n_states": predictions.assignments.n_states,
        "n_runs": len(metrics),
        "n_sequences": predictions.n_sequences,
        "n_samples": predictions.n_samples,
        "subjects": list(predictions.subjects),
    }


def score_states(namespace: argparse.Namespace) -> dict[str, object]:
    """Score a fitted state model on selected acquisitions."""
    if namespace.output.exists() or namespace.output.is_symlink():
        raise FileExistsError(f"state-model score output path already exists: {namespace.output}")
    store = FeatureStore.open(namespace.store)
    model = load_fitted_model(namespace.model)
    subjects = selected_store_subjects(store, namespace.subject)
    if isinstance(model, KMeansStateModel):
        scores = score_kmeans_store(
            model,
            store,
            subjects=subjects,
            allow_fit_subjects=namespace.allow_fit_subjects,
        )
        model_kind = "kmeans-state"
        minimum_sequence_length = 1
    elif isinstance(model, GaussianHMMStateModel):
        scores = score_gaussian_hmm_store(
            model,
            store,
            subjects=subjects,
            allow_fit_subjects=namespace.allow_fit_subjects,
        )
        model_kind = "gaussian-hmm-state"
        minimum_sequence_length = model.minimum_sequence_length
    else:
        raise TypeError("score-states requires a KMeans or Gaussian HMM model artifact")
    scored_subjects = tuple(dict.fromkeys(score.subject for score in scores))
    selected_subject_set = set(scored_subjects)
    omitted_short_sequence_count = sum(
        identity[0] in selected_subject_set and count < minimum_sequence_length
        for identity, count in store.sequence_sample_counts
    )
    output = write_state_model_scores(
        scores,
        namespace.output,
        model_kind=model_kind,
        model_seed=model.seed,
        n_states=model.n_states,
        fit_subjects=model.fit_subjects,
        source_contract=store.source_contract,
        sample_interval_seconds=store.sample_interval_seconds,
        minimum_sequence_length=minimum_sequence_length,
        omitted_short_sequence_count=omitted_short_sequence_count,
        allow_fit_subjects=namespace.allow_fit_subjects,
        model_specification=_state_model_specification(model),
    )
    return {
        "output": str(output),
        "model_kind": model_kind,
        "n_states": model.n_states,
        "subjects": list(scored_subjects),
        "n_runs": len(scores),
        "n_sequences": sum(score.n_sequences for score in scores),
        "n_samples": sum(score.n_samples for score in scores),
        "minimum_sequence_length": minimum_sequence_length,
        "omitted_short_sequence_count": omitted_short_sequence_count,
        "allow_fit_subjects": bool(namespace.allow_fit_subjects),
    }
