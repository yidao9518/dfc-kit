"""Command handlers for state alignment and repeated-fit stability."""

from __future__ import annotations

import argparse

from ..artifacts import (
    StatePredictions,
    load_fitted_model,
    load_state_predictions,
    save_fitted_model,
    save_state_alignment,
    save_state_predictions,
    write_state_stability,
)
from ..states.alignment import (
    align_gaussian_hmm_emissions,
    align_kmeans_centroids,
    apply_gaussian_hmm_alignment,
    apply_state_alignment,
    relabel_gaussian_hmm_model,
    relabel_kmeans_model,
)
from ..states.data import StateAssignments
from ..states.hmm import GaussianHMMStateModel, GaussianHMMStateResult
from ..states.kmeans import KMeansStateModel
from ..states.stability import summarize_state_stability
from ..states.streaming import predict_kmeans_store
from ..states.streaming_hmm import predict_gaussian_hmm_store
from ..storage import FeatureStore
from .states import selected_store_subjects


def _requested_alignment_metric(
    namespace: argparse.Namespace,
    model: KMeansStateModel | GaussianHMMStateModel,
) -> str:
    requested = getattr(namespace, "alignment_metric", "auto")
    if requested == "auto":
        return "pearson" if model.source_contract.startswith("cap:") else "euclidean"
    return str(requested)


def align_states(namespace: argparse.Namespace) -> dict[str, object]:
    """Align a candidate model and optional predictions to a reference model."""
    reference = load_fitted_model(namespace.reference_model)
    candidate = load_fitted_model(namespace.candidate_model)
    if type(reference) is not type(candidate):
        raise TypeError("reference and candidate models must use the same state-model family")
    alignment_metric = _requested_alignment_metric(namespace, reference)
    if isinstance(reference, KMeansStateModel):
        alignment = align_kmeans_centroids(
            reference,
            candidate,
            metric=alignment_metric,
        )
        aligned_model = relabel_kmeans_model(candidate, alignment)
        model_kind = "kmeans-state"
    elif isinstance(reference, GaussianHMMStateModel):
        alignment = align_gaussian_hmm_emissions(
            reference,
            candidate,
            metric=alignment_metric,
        )
        aligned_model = relabel_gaussian_hmm_model(candidate, alignment)
        model_kind = "gaussian-hmm-state"
    else:
        raise TypeError("align-states requires KMeans or Gaussian HMM model artifacts")

    prediction_output = namespace.predictions_output
    if (namespace.predictions is None) != (prediction_output is None):
        raise ValueError("--predictions and --predictions-output must be provided together")
    aligned_predictions = None
    if namespace.predictions is not None:
        predictions = load_state_predictions(namespace.predictions)
        if predictions.model_kind != model_kind:
            raise ValueError("prediction artifact and candidate model use different model families")
        if predictions.model_seed != candidate.seed:
            raise ValueError("prediction artifact seed does not match the candidate model")
        if model_kind == "kmeans-state":
            aligned_assignments = apply_state_alignment(predictions.assignments, alignment)
            aligned_predictions = StatePredictions(
                assignments=aligned_assignments,
                model_kind=model_kind,
                model_seed=candidate.seed,
            )
        else:
            if predictions.posterior_probabilities is None or predictions.log_likelihood is None:
                raise ValueError("HMM prediction artifact is missing posterior information")
            result = GaussianHMMStateResult(
                assignments=predictions.assignments,
                posterior_probabilities=predictions.posterior_probabilities,
                log_likelihood=predictions.log_likelihood,
            )
            aligned_result = apply_gaussian_hmm_alignment(result, alignment)
            aligned_predictions = StatePredictions(
                assignments=aligned_result.assignments,
                model_kind=model_kind,
                model_seed=candidate.seed,
                posterior_probabilities=aligned_result.posterior_probabilities,
                log_likelihood=aligned_result.log_likelihood,
            )

    output_paths = [namespace.output_model, namespace.alignment_output]
    if aligned_predictions is not None:
        output_paths.append(prediction_output)
    normalized_outputs = [path.resolve(strict=False) for path in output_paths]
    if len(set(normalized_outputs)) != len(normalized_outputs):
        raise ValueError("align-states output paths must be distinct")
    if any(path.exists() or path.is_symlink() for path in output_paths):
        raise FileExistsError("one or more align-states output paths already exist")
    alignment_path = save_state_alignment(alignment, namespace.alignment_output)
    model_path = save_fitted_model(aligned_model, namespace.output_model)
    prediction_path = None
    if aligned_predictions is not None:
        prediction_path = save_state_predictions(aligned_predictions, prediction_output)
    return {
        "model_kind": model_kind,
        "reference_seed": alignment.reference_seed,
        "candidate_seed": alignment.candidate_seed,
        "candidate_to_reference": alignment.candidate_to_reference.tolist(),
        "alignment_metric": alignment.metric,
        "matched_costs": alignment.matched_costs.tolist(),
        "output_model": str(model_path),
        "alignment_output": str(alignment_path),
        "predictions_output": None if prediction_path is None else str(prediction_path),
    }


def _repeated_fit_contract(
    model: KMeansStateModel | GaussianHMMStateModel,
) -> tuple[object, ...]:
    common = (
        type(model),
        model.feature_keys,
        model.source_contract,
        model.sample_interval_seconds,
        model.n_states,
        model.fit_subjects,
        model.fit_sample_count,
        model.implementation,
    )
    if isinstance(model, KMeansStateModel):
        return common + (
            model.n_init,
            model.max_iter,
            model.algorithm,
            model.standardize_features,
            model.batch_size,
            model.reassignment_ratio,
            model.init_sample_size,
        )
    return common + (
        model.fit_sequence_count,
        model.omitted_short_sequence_count,
        model.n_pca_components,
        model.covariance_type,
        model.n_init,
        model.n_iter,
        model.tol,
        model.minimum_sequence_length,
        model.pca_batch_size,
    )


def _require_complete_repeated_fit_provenance(
    model: KMeansStateModel | GaussianHMMStateModel,
) -> None:
    if (
        isinstance(model, KMeansStateModel)
        and model.algorithm == "minibatch"
        and "out-of-core" in model.implementation
        and model.init_sample_size is None
    ):
        raise ValueError("out-of-core KMeans stability requires a recorded init_sample_size")
    if (
        isinstance(model, GaussianHMMStateModel)
        and "IncrementalPCA" in model.implementation
        and model.pca_batch_size is None
    ):
        raise ValueError("out-of-core HMM stability requires a recorded pca_batch_size")

def _decode_assignments(
    model: KMeansStateModel | GaussianHMMStateModel,
    store: FeatureStore,
    *,
    subjects: tuple[str, ...] | None,
    allow_fit_subjects: bool,
) -> StateAssignments:
    if isinstance(model, KMeansStateModel):
        return predict_kmeans_store(
            model,
            store,
            subjects=subjects,
            allow_fit_subjects=allow_fit_subjects,
        )
    return predict_gaussian_hmm_store(
        model,
        store,
        subjects=subjects,
        allow_fit_subjects=allow_fit_subjects,
    ).assignments


def summarize_stability(namespace: argparse.Namespace) -> dict[str, object]:
    """Align repeated state fits and summarize their run-level agreement."""
    if namespace.output.exists() or namespace.output.is_symlink():
        raise FileExistsError(f"state-stability output path already exists: {namespace.output}")
    store = FeatureStore.open(namespace.store)
    subjects = selected_store_subjects(store, namespace.subject)
    reference = load_fitted_model(namespace.reference_model)
    candidates = tuple(load_fitted_model(path) for path in namespace.candidate_model)
    if not isinstance(reference, (KMeansStateModel, GaussianHMMStateModel)):
        raise TypeError("summarize-stability requires a KMeans or Gaussian HMM reference")
    if any(type(candidate) is not type(reference) for candidate in candidates):
        raise TypeError("all stability models must use the same state-model family")
    for model in (reference, *candidates):
        _require_complete_repeated_fit_provenance(model)
    if any(candidate.fit_subjects != reference.fit_subjects for candidate in candidates):
        raise ValueError("stability models must use the same training subjects")
    if any(candidate.fit_sample_count != reference.fit_sample_count for candidate in candidates):
        raise ValueError("stability models must use the same fitted sample count")
    reference_contract = _repeated_fit_contract(reference)
    if any(_repeated_fit_contract(candidate) != reference_contract for candidate in candidates):
        raise ValueError(
            "stability models must share training subjects, fitted samples, and hyperparameters"
        )
    seeds = tuple(model.seed for model in (reference, *candidates))
    if len(set(seeds)) != len(seeds):
        raise ValueError("stability models must use distinct seeds")

    reference_assignments = _decode_assignments(
        reference,
        store,
        subjects=subjects,
        allow_fit_subjects=namespace.allow_fit_subjects,
    )
    aligned_assignments = [reference_assignments]
    alignments = []
    alignment_metric = _requested_alignment_metric(namespace, reference)
    for candidate in candidates:
        candidate_assignments = _decode_assignments(
            candidate,
            store,
            subjects=subjects,
            allow_fit_subjects=namespace.allow_fit_subjects,
        )
        if isinstance(reference, KMeansStateModel):
            alignment = align_kmeans_centroids(
                reference,
                candidate,
                metric=alignment_metric,
            )
        else:
            alignment = align_gaussian_hmm_emissions(
                reference,
                candidate,
                metric=alignment_metric,
            )
        alignments.append(alignment)
        aligned_assignments.append(apply_state_alignment(candidate_assignments, alignment))
    stability = summarize_state_stability(aligned_assignments)
    model_kind = "kmeans-state" if isinstance(reference, KMeansStateModel) else "gaussian-hmm-state"
    output = write_state_stability(
        stability,
        namespace.output,
        model_kind=model_kind,
        reference_seed=reference.seed,
        candidate_seeds=tuple(candidate.seed for candidate in candidates),
        alignments=tuple(alignments),
        source_contract=store.source_contract,
        sample_interval_seconds=store.sample_interval_seconds,
        allow_fit_subjects=namespace.allow_fit_subjects,
    )
    return {
        "output": str(output),
        "model_kind": model_kind,
        "reference_seed": reference.seed,
        "candidate_seeds": [candidate.seed for candidate in candidates],
        "subjects": list(dict.fromkeys(run.subject for run in stability)),
        "n_fits": len(aligned_assignments),
        "n_states": reference.n_states,
        "n_runs": len(stability),
        "n_sequences": sum(run.n_sequences for run in stability),
        "n_samples": sum(run.n_samples for run in stability),
        "allow_fit_subjects": bool(namespace.allow_fit_subjects),
        "alignment_metric": alignment_metric,
    }
