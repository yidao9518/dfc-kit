"""Small command-line adapters for the XCP-D-first workflows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .connectivity import LEiDA, SlidingWindowFC
from .data import TimeSeriesDataset
from .io import (
    StatePredictions,
    compare_state_model_scores,
    compute_fixed_information,
    discover_xcpd_runs,
    fitted_model_fingerprint,
    inspect_nested_state_count_progress,
    load_fitted_model,
    load_fixed_window_schedule,
    load_information_groups,
    load_nested_state_count_cross_validation,
    load_selected_state_count_evaluation,
    load_state_count_cross_validation,
    load_state_model_scores,
    load_state_predictions,
    load_xcpd_dataset,
    nested_state_count_progress_payload,
    save_fitted_model,
    save_fixed_information,
    save_state_alignment,
    save_state_predictions,
    state_model_specification,
    write_selected_state_count_evaluation,
    write_state_count_comparison,
    write_state_metrics,
    write_state_model_scores,
    write_state_stability,
)
from .io.state_cross_validation_checkpoint import (
    MANIFEST_NAME as STATE_COUNT_CHECKPOINT_MANIFEST,
)
from .io.state_cross_validation_checkpoint import (
    inspect_state_count_checkpoint,
    load_state_count_checkpoint_cell,
    make_state_count_checkpoint_plan,
    state_count_checkpoint_cell_paths,
    validate_completed_state_count_checkpoint,
    validate_state_count_checkpoint_contract,
    write_state_count_checkpoint,
)
from .io.state_nested_checkpoint import (
    load_nested_state_count_checkpoint,
    write_nested_state_count_checkpoint,
)
from .io.state_nested_contract import (
    validate_nested_evaluation,
    validate_nested_selection,
)
from .io.state_nested_lock import acquire_nested_checkpoint_lock
from .outofcore import (
    fit_kmeans_store_materialized,
    fit_minibatch_kmeans_store,
    predict_kmeans_store,
    score_kmeans_store,
)
from .outofcore_hmm import (
    fit_gaussian_hmm_store,
    predict_gaussian_hmm_store,
    score_gaussian_hmm_store,
)
from .states.alignment import (
    align_gaussian_hmm_emissions,
    align_kmeans_centroids,
    apply_gaussian_hmm_alignment,
    apply_state_alignment,
    relabel_gaussian_hmm_model,
    relabel_kmeans_model,
)
from .states.cross_validation import make_subject_validation_folds
from .states.data import StateAssignments
from .states.hmm import GaussianHMMStateModel, GaussianHMMStateResult
from .states.kmeans import KMeansStateModel
from .states.metrics import summarize_state_assignments
from .states.stability import summarize_state_stability
from .storage import (
    FeatureStore,
    write_cap_store,
    write_ets_store,
    write_leida_store,
    write_mtd_store,
    write_window_fc_store,
)


def _add_xcpd_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("root", type=Path, help="XCP-D derivatives root")
    parser.add_argument(
        "--atlas",
        action="append",
        required=True,
        help="atlas name; repeat for multiple atlases",
    )
    parser.add_argument(
        "--subject",
        action="append",
        help="optional BIDS subject label or value; repeat to select multiple subjects",
    )
    parser.add_argument("--session", help="optional BIDS session label or value")
    parser.add_argument("--task", help="optional BIDS task label or value")
    parser.add_argument("--space", help="optional XCP-D standard-space label")


def _add_load_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--roi-selection",
        type=Path,
        help="JSON object mapping atlas names to ordered ROI-name arrays",
    )
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=None,
        help="reject selected ROIs below this XCP-D coverage fraction",
    )
    parser.add_argument("--tr", type=float, default=None, help="run TR in seconds")


def _load_roi_selection(path: Path | None) -> dict[str, tuple[str, ...]] | None:
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read ROI selection JSON {path}: {error}") from error
    if not isinstance(raw, dict) or not raw:
        raise ValueError("ROI selection must be a non-empty JSON object")
    output: dict[str, tuple[str, ...]] = {}
    for atlas, names in raw.items():
        if not isinstance(atlas, str) or not atlas.strip():
            raise ValueError("ROI selection atlas names must be non-empty strings")
        if isinstance(names, str) or not isinstance(names, Sequence) or not names:
            raise ValueError(f"ROI selection for {atlas!r} must be a non-empty array")
        selected = tuple(str(name) for name in names)
        if any(not name.strip() for name in selected) or len(set(selected)) != len(selected):
            raise ValueError(f"ROI selection for {atlas!r} must contain unique non-empty names")
        output[atlas] = selected
    return output


def _filters(namespace: argparse.Namespace, *, subject: str | None = None) -> dict[str, object]:
    return {
        "atlases": tuple(namespace.atlas),
        "subject": subject,
        "session": namespace.session,
        "task": namespace.task,
        "space": namespace.space,
    }


def _selected_xcpd_subjects(namespace: argparse.Namespace) -> tuple[str | None, ...]:
    subjects = namespace.subject
    if subjects is None:
        return (None,)
    values = tuple(str(subject).strip().removeprefix("sub-") for subject in subjects)
    if any(not subject for subject in values):
        raise ValueError("--subject values must be non-empty")
    normalized = tuple(f"sub-{subject}" for subject in values)
    unique = tuple(dict.fromkeys(normalized))
    if len(unique) != len(subjects):
        raise ValueError("--subject values must be unique")
    return unique


def _discover_selected_xcpd_runs(namespace: argparse.Namespace):
    discovered = []
    for subject in _selected_xcpd_subjects(namespace):
        discovered.extend(
            discover_xcpd_runs(namespace.root, **_filters(namespace, subject=subject))
        )
    return tuple(discovered)


def _inspect(namespace: argparse.Namespace) -> dict[str, object]:
    files = _discover_selected_xcpd_runs(namespace)
    acquisitions = []
    for item in files:
        stem = item.outliers.name.removesuffix("_outliers.tsv")
        entities = {
            token.split("-", 1)[0]: token.split("-", 1)[1]
            for token in stem.split("_")
            if "-" in token
        }
        acquisitions.append(
            {
                "acquisition_id": stem,
                "subject": entities.get("sub"),
                "session": entities.get("ses"),
                "task": entities.get("task"),
                "atlases": [atlas.atlas for atlas in item.atlases],
                "files": {
                    atlas.atlas: {
                        "timeseries": str(atlas.timeseries),
                        "coverage": str(atlas.coverage),
                    }
                    for atlas in item.atlases
                },
                "outliers": str(item.outliers),
                "motion": None if item.motion is None else str(item.motion),
            }
        )
    return {"n_acquisitions": len(acquisitions), "acquisitions": acquisitions}


def _build_store(namespace: argparse.Namespace) -> dict[str, object]:
    selection = _load_roi_selection(namespace.roi_selection)
    datasets = tuple(
        load_xcpd_dataset(
            namespace.root,
            roi_names=selection,
            minimum_coverage=namespace.minimum_coverage,
            tr=namespace.tr,
            **_filters(namespace, subject=subject),
        )
        for subject in _selected_xcpd_subjects(namespace)
    )
    dataset = TimeSeriesDataset(tuple(run for item in datasets for run in item.runs))
    if namespace.method == "window-fc":
        estimator = SlidingWindowFC(
            length=namespace.window_length,
            step=namespace.window_step,
            taper=namespace.taper,
        )
        store = write_window_fc_store(
            namespace.output,
            dataset.runs,
            estimator,
            chunk_size=namespace.chunk_size,
            dtype=namespace.dtype,
        )
    elif namespace.method == "cap":
        store = write_cap_store(
            namespace.output,
            dataset.runs,
            chunk_size=namespace.chunk_size,
            dtype=namespace.dtype,
        )
    elif namespace.method == "leida":
        store = write_leida_store(
            namespace.output,
            dataset.runs,
            LEiDA(minimum_segment_length=namespace.minimum_segment_length),
            chunk_size=namespace.chunk_size,
            dtype=namespace.dtype,
        )
    elif namespace.method == "ets":
        store = write_ets_store(
            namespace.output,
            dataset.runs,
            chunk_size=namespace.chunk_size,
            dtype=namespace.dtype,
        )
    else:
        store = write_mtd_store(
            namespace.output,
            dataset.runs,
            chunk_size=namespace.chunk_size,
            dtype=namespace.dtype,
        )
    return {
        "method": namespace.method,
        "output": str(store.root),
        "n_runs": dataset.n_runs,
        "subjects": list(dataset.subjects),
        "acquisition_ids": [run.acquisition_id for run in dataset.runs],
        "n_sequences": store.n_sequences,
        "n_samples": store.n_samples,
        "n_chunks": store.n_chunks,
        "n_features": store.n_features,
        "format_version": store.format_version,
    }


def _fixed_information(namespace: argparse.Namespace) -> dict[str, object]:
    if namespace.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    selection = _load_roi_selection(namespace.roi_selection)
    datasets = tuple(
        load_xcpd_dataset(
            namespace.root,
            roi_names=selection,
            minimum_coverage=namespace.minimum_coverage,
            tr=namespace.tr,
            **_filters(namespace, subject=subject),
        )
        for subject in _selected_xcpd_subjects(namespace)
    )
    dataset = TimeSeriesDataset(tuple(run for item in datasets for run in item.runs))
    groups = load_information_groups(namespace.information_groups)
    schedule = (
        None
        if namespace.window_schedule is None
        else load_fixed_window_schedule(namespace.window_schedule)
    )
    artifact = compute_fixed_information(
        dataset,
        groups,
        lengths=namespace.length,
        draws=namespace.draws,
        sample_seed=namespace.sample_seed,
        schedule=schedule,
        schedule_source=(
            None if namespace.window_schedule is None else str(namespace.window_schedule)
        ),
        k=namespace.k,
        jitter=namespace.jitter,
        jitter_seed=namespace.jitter_seed,
        standardize=namespace.standardize,
        jobs=namespace.jobs,
    )
    output = save_fixed_information(artifact, namespace.output)
    return {
        "acquisition_ids": [item.acquisition_id for item in artifact.acquisitions],
        "conditioning_rois": (
            [] if artifact.groups.conditioning is None else list(artifact.groups.conditioning)
        ),
        "draws_per_length": artifact.draws_per_length,
        "format_version": artifact.format_version,
        "has_cmi": artifact.has_cmi,
        "left_rois": list(artifact.groups.left),
        "lengths": list(artifact.lengths),
        "method": "fixed-information",
        "n_draws": artifact.n_draws,
        "n_runs": len(artifact.acquisitions),
        "n_cells": artifact.n_cells,
        "jobs": namespace.jobs,
        "output": str(output),
        "right_rois": list(artifact.groups.right),
        "schedule_mode": artifact.schedule_mode,
        "subjects": list(dataset.subjects),
    }


def _fit_states(namespace: argparse.Namespace) -> dict[str, object]:
    """Fit one state model from a disk-backed feature store.

    The command deliberately writes only the portable model artifact.  State
    labels remain a prediction result so callers can choose their own output
    format and held-out subject policy instead of silently mixing train and
    test observations in one file.
    """
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
                subjects=subjects,
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
            "converged": None,
            "log_likelihood": None,
            "inertia": model.inertia,
            "init_sample_size": model.init_sample_size,
            "training_data_fingerprint": model.training_data_fingerprint,
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
        "training_data_fingerprint": model.training_data_fingerprint,
    }


def _selected_store_subjects(store: FeatureStore, requested: list[str] | None) -> tuple[str, ...] | None:
    if requested is None:
        return None
    missing = sorted(set(requested) - set(store.subjects))
    if missing:
        raise ValueError(f"requested prediction subjects are absent from the store: {missing}")
    return tuple(requested)


def _predict_states(namespace: argparse.Namespace) -> dict[str, object]:
    store = FeatureStore.open(namespace.store)
    model = load_fitted_model(namespace.model)
    subjects = _selected_store_subjects(store, namespace.subject)
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
            model_fingerprint=fitted_model_fingerprint(model),
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
            model_fingerprint=fitted_model_fingerprint(model),
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


def _summarize_states(namespace: argparse.Namespace) -> dict[str, object]:
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


def _score_states(namespace: argparse.Namespace) -> dict[str, object]:
    if namespace.output.exists() or namespace.output.is_symlink():
        raise FileExistsError(
            f"state-model score output path already exists: {namespace.output}"
        )
    store = FeatureStore.open(namespace.store)
    model = load_fitted_model(namespace.model)
    subjects = _selected_store_subjects(store, namespace.subject)
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
    evaluation_fingerprint = store.data_fingerprint(
        subjects=scored_subjects,
        minimum_sequence_length=minimum_sequence_length,
    )
    output = write_state_model_scores(
        scores,
        namespace.output,
        model_kind=model_kind,
        model_fingerprint=fitted_model_fingerprint(model),
        model_seed=model.seed,
        n_states=model.n_states,
        fit_subjects=model.fit_subjects,
        training_data_fingerprint=model.training_data_fingerprint,
        evaluation_data_fingerprint=evaluation_fingerprint,
        feature_contract_fingerprint=store.feature_contract_fingerprint(),
        source_contract=store.source_contract,
        sample_interval_seconds=store.sample_interval_seconds,
        minimum_sequence_length=minimum_sequence_length,
        omitted_short_sequence_count=omitted_short_sequence_count,
        allow_fit_subjects=namespace.allow_fit_subjects,
        model_specification=state_model_specification(model),
    )
    return {
        "output": str(output),
        "model_kind": model_kind,
        "model_fingerprint": fitted_model_fingerprint(model),
        "n_states": model.n_states,
        "subjects": list(scored_subjects),
        "n_runs": len(scores),
        "n_sequences": sum(score.n_sequences for score in scores),
        "n_samples": sum(score.n_samples for score in scores),
        "minimum_sequence_length": minimum_sequence_length,
        "omitted_short_sequence_count": omitted_short_sequence_count,
        "evaluation_data_fingerprint": evaluation_fingerprint,
        "allow_fit_subjects": bool(namespace.allow_fit_subjects),
    }


def _compare_state_counts(namespace: argparse.Namespace) -> dict[str, object]:
    if namespace.output.exists() or namespace.output.is_symlink():
        raise FileExistsError(
            f"state-count comparison output path already exists: {namespace.output}"
        )
    reports = tuple(load_state_model_scores(path) for path in namespace.score)
    comparison = compare_state_model_scores(
        reports,
        score_artifacts=namespace.score,
    )
    output = write_state_count_comparison(comparison, namespace.output)
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
        "one_standard_error_n_states": (
            comparison.selection.one_standard_error_n_states
        ),
    }


def _integer_grid(
    values: Sequence[int],
    name: str,
    *,
    minimum: int,
    minimum_count: int,
) -> tuple[int, ...]:
    raw = tuple(values)
    if len(raw) < minimum_count:
        raise ValueError(f"{name} requires at least {minimum_count} value(s)")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise TypeError(f"{name} values must be integers")
    if any(value < minimum for value in raw):
        raise ValueError(f"{name} values must be at least {minimum}")
    if len(set(raw)) != len(raw):
        raise ValueError(f"{name} values must be unique")
    return tuple(sorted(raw))


def _cross_validation_subjects(
    store: FeatureStore,
    requested: list[str] | None,
) -> tuple[str, ...]:
    if requested is None:
        return store.subjects
    if len(set(requested)) != len(requested):
        raise ValueError("cross-validation subjects must be unique")
    missing = sorted(set(requested) - set(store.subjects))
    if missing:
        raise ValueError(f"requested cross-validation subjects are absent: {missing}")
    selected = set(requested)
    return tuple(subject for subject in store.subjects if subject in selected)


def _fit_cross_validation_model(
    namespace: argparse.Namespace,
    store: FeatureStore,
    *,
    n_states: int,
    seed: int,
    fit_subjects: tuple[str, ...],
) -> KMeansStateModel | GaussianHMMStateModel:
    if namespace.method == "kmeans":
        return fit_minibatch_kmeans_store(
            store,
            n_states=n_states,
            seed=seed,
            n_init=10 if namespace.n_init is None else namespace.n_init,
            max_iter=namespace.max_iter,
            batch_size=namespace.batch_size,
            standardize_features=namespace.standardize_features,
            reassignment_ratio=namespace.reassignment_ratio,
            init_sample_size=namespace.init_sample_size,
            subjects=fit_subjects,
        ).model
    assert namespace.n_pca_components is not None
    return fit_gaussian_hmm_store(
        store,
        n_states=n_states,
        seed=seed,
        n_pca_components=namespace.n_pca_components,
        covariance_type=namespace.covariance_type,
        n_init=1 if namespace.n_init is None else namespace.n_init,
        n_iter=namespace.n_iter,
        tol=namespace.tol,
        pca_batch_size=namespace.pca_batch_size,
        subjects=fit_subjects,
        minimum_sequence_length=namespace.minimum_sequence_length,
    ).model


def _cross_validation_scores(
    store: FeatureStore,
    model: KMeansStateModel | GaussianHMMStateModel,
    evaluation_subjects: tuple[str, ...],
) -> tuple[tuple[object, ...], int, int, str]:
    if isinstance(model, KMeansStateModel):
        scores = score_kmeans_store(model, store, subjects=evaluation_subjects)
        minimum_sequence_length = 1
    else:
        scores = score_gaussian_hmm_store(model, store, subjects=evaluation_subjects)
        minimum_sequence_length = model.minimum_sequence_length
    selected = set(evaluation_subjects)
    omitted = sum(
        identity[0] in selected and count < minimum_sequence_length
        for identity, count in store.sequence_sample_counts
    )
    fingerprint = store.data_fingerprint(
        subjects=evaluation_subjects,
        minimum_sequence_length=minimum_sequence_length,
    )
    return tuple(scores), minimum_sequence_length, omitted, fingerprint


def _cross_validation_configuration(namespace: argparse.Namespace) -> dict[str, object]:
    if namespace.method == "kmeans":
        return {
            "batch_size": namespace.batch_size,
            "init_sample_size": namespace.init_sample_size,
            "max_iter": namespace.max_iter,
            "n_init": 10 if namespace.n_init is None else namespace.n_init,
            "reassignment_ratio": namespace.reassignment_ratio,
            "standardize_features": namespace.standardize_features,
        }
    return {
        "covariance_type": namespace.covariance_type,
        "minimum_sequence_length": namespace.minimum_sequence_length,
        "n_init": 1 if namespace.n_init is None else namespace.n_init,
        "n_iter": namespace.n_iter,
        "n_pca_components": namespace.n_pca_components,
        "pca_batch_size": namespace.pca_batch_size,
        "tol": namespace.tol,
    }


def _cross_validate_state_counts(namespace: argparse.Namespace) -> dict[str, object]:
    output = namespace.output
    cell_checkpoint = getattr(namespace, "_cell_checkpoint", None)
    preserve_cells = cell_checkpoint is not None
    if (output.exists() or output.is_symlink()) and not preserve_cells:
        raise FileExistsError(f"cross-validation output path already exists: {output}")
    candidates = _integer_grid(
        namespace.n_states,
        "--n-states",
        minimum=2,
        minimum_count=2,
    )
    model_seeds = _integer_grid(
        namespace.seed,
        "--seed",
        minimum=0,
        minimum_count=1,
    )
    if namespace.method == "hmm" and namespace.n_pca_components is None:
        raise ValueError("--n-pca-components is required when --method hmm")

    store = FeatureStore.open(namespace.store)
    subjects = _cross_validation_subjects(store, namespace.subject)
    minimum_sequence_length = (
        1 if namespace.method == "kmeans" else namespace.minimum_sequence_length
    )
    if namespace.method == "hmm":
        eligible_subjects = {
            identity[0]
            for identity, count in store.sequence_sample_counts
            if count >= minimum_sequence_length
        }
        missing = sorted(set(subjects) - eligible_subjects)
        if missing:
            raise ValueError(
                "cross-validation subjects have no HMM-eligible sequence: "
                f"{missing}"
            )
    configuration = _cross_validation_configuration(namespace)
    plan = make_state_count_checkpoint_plan(
        store,
        method=namespace.method,
        subjects=subjects,
        candidate_n_states=candidates,
        model_seeds=model_seeds,
        n_folds=namespace.n_folds,
        split_seed=namespace.split_seed,
        configuration=configuration,
        minimum_sequence_length=minimum_sequence_length,
    )
    ordered_folds = plan.ordered_folds
    development_fingerprint = str(plan.payload["development_data_fingerprint"])

    output.parent.mkdir(parents=True, exist_ok=True)
    if preserve_cells:
        temporary = Path(cell_checkpoint)
        if temporary.resolve(strict=False) != output.resolve(strict=False):
            raise ValueError("internal cell checkpoint must be the selection output path")
        if temporary.is_symlink() or (temporary.exists() and not temporary.is_dir()):
            raise ValueError("state-count cell checkpoint must be a real directory")
        if temporary.exists():
            validate_state_count_checkpoint_contract(temporary, plan)
            inspect_state_count_checkpoint(store, plan, temporary)
        else:
            temporary.mkdir()
            write_state_count_checkpoint(
                plan.payload,
                temporary / STATE_COUNT_CHECKPOINT_MANIFEST,
            )
        _remove_checkpoint_temporaries(temporary, "comparison.json")
        _remove_checkpoint_temporaries(temporary, "workflow.json")
        comparison_path = temporary / "comparison.json"
        if comparison_path.exists():
            if not comparison_path.is_file() or comparison_path.is_symlink():
                raise ValueError("checkpoint comparison path must be a regular file")
            comparison_path.unlink()
    else:
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
        )
    reports = []
    score_artifacts: list[str] = []
    fit_records: list[dict[str, object]] = []
    fold_records: list[dict[str, object]] = []
    reused_models = 0
    reused_scores = 0
    try:
        for fold_index, (evaluation_fingerprint, fold) in enumerate(ordered_folds):
            fold_records.append(
                {
                    "fold_index": fold_index,
                    "assignment_index": fold.fold_index,
                    "fit_subjects": list(fold.fit_subjects),
                    "evaluation_subjects": list(fold.evaluation_subjects),
                    "evaluation_data_fingerprint": evaluation_fingerprint,
                }
            )
            for n_states in candidates:
                for seed in model_seeds:
                    stem = f"fold-{fold_index + 1:03d}_k-{n_states}_seed-{seed}"
                    model_relative = Path("models") / f"{stem}.model"
                    score_relative = Path("scores") / f"{stem}.json"
                    model = None
                    report = None
                    if preserve_cells:
                        model_path, score_path = state_count_checkpoint_cell_paths(
                            temporary,
                            fold_index,
                            n_states,
                            seed,
                        )
                        _remove_checkpoint_temporaries(
                            model_path.parent,
                            model_path.name,
                        )
                        _remove_checkpoint_temporaries(
                            score_path.parent,
                            score_path.name,
                        )
                        model, report = load_state_count_checkpoint_cell(
                            store,
                            plan,
                            temporary,
                            fold_index=fold_index,
                            n_states=n_states,
                            seed=seed,
                        )
                        reused_models += model is not None
                        reused_scores += report is not None
                    if model is None:
                        model = _fit_cross_validation_model(
                            namespace,
                            store,
                            n_states=n_states,
                            seed=seed,
                            fit_subjects=fold.fit_subjects,
                        )
                        if model.fit_subjects != fold.fit_subjects:
                            raise RuntimeError(
                                "fitted model subjects disagree with the validation fold"
                            )
                        save_fitted_model(model, temporary / model_relative)
                    if report is None:
                        (
                            scores,
                            observed_minimum,
                            omitted_short_sequence_count,
                            observed_fingerprint,
                        ) = _cross_validation_scores(
                            store,
                            model,
                            fold.evaluation_subjects,
                        )
                        if observed_fingerprint != evaluation_fingerprint:
                            raise RuntimeError(
                                "evaluation fingerprint changed during cross-validation"
                            )
                        write_state_model_scores(
                            scores,
                            temporary / score_relative,
                            model_kind=(
                                "kmeans-state"
                                if isinstance(model, KMeansStateModel)
                                else "gaussian-hmm-state"
                            ),
                            model_fingerprint=fitted_model_fingerprint(model),
                            model_seed=model.seed,
                            n_states=model.n_states,
                            fit_subjects=model.fit_subjects,
                            training_data_fingerprint=(
                                model.training_data_fingerprint
                            ),
                            evaluation_data_fingerprint=observed_fingerprint,
                            feature_contract_fingerprint=(
                                store.feature_contract_fingerprint()
                            ),
                            source_contract=store.source_contract,
                            sample_interval_seconds=store.sample_interval_seconds,
                            minimum_sequence_length=observed_minimum,
                            omitted_short_sequence_count=(
                                omitted_short_sequence_count
                            ),
                            allow_fit_subjects=False,
                            model_specification=state_model_specification(model),
                        )
                        report = load_state_model_scores(
                            temporary / score_relative
                        )
                    reports.append(report)
                    score_reference = score_relative.as_posix()
                    score_artifacts.append(score_reference)
                    fit_records.append(
                        {
                            "fold_index": fold_index,
                            "n_states": n_states,
                            "model_seed": seed,
                            "model_fingerprint": fitted_model_fingerprint(model),
                            "model_artifact": model_relative.as_posix(),
                            "score_artifact": score_reference,
                        }
                    )

        comparison = compare_state_model_scores(
            reports,
            score_artifacts=score_artifacts,
        )
        write_state_count_comparison(comparison, temporary / "comparison.json")
        workflow = {
            "format": "dfckit-state-count-cross-validation",
            "format_version": 1,
            "model_kind": comparison.model_kind,
            "method": namespace.method,
            "source_contract": store.source_contract,
            "sample_interval_seconds": store.sample_interval_seconds,
            "feature_contract_fingerprint": store.feature_contract_fingerprint(),
            "development_data_fingerprint": development_fingerprint,
            "development_subjects": list(subjects),
            "candidate_n_states": list(candidates),
            "model_seeds": list(model_seeds),
            "split": {
                "algorithm": "sha256-seed-subject-balanced-v1",
                "seed": namespace.split_seed,
                "n_folds": namespace.n_folds,
                "folds": fold_records,
            },
            "fit_configuration": configuration,
            "n_fits": len(fit_records),
            "fits": fit_records,
            "comparison_artifact": "comparison.json",
            "best_n_states": comparison.selection.best_n_states,
            "one_standard_error_n_states": (
                comparison.selection.one_standard_error_n_states
            ),
        }
        workflow_path = temporary / "workflow.json"
        if preserve_cells:
            _write_state_count_checkpoint_json(workflow, workflow_path)
        else:
            workflow_path.write_text(
                json.dumps(workflow, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        if preserve_cells:
            selection = load_state_count_cross_validation(temporary)
            if selection.development_data_fingerprint != development_fingerprint:
                raise RuntimeError("completed cell checkpoint changed development data")
            (temporary / STATE_COUNT_CHECKPOINT_MANIFEST).unlink()
        else:
            if output.exists() or output.is_symlink():
                raise FileExistsError(
                    f"cross-validation output path already exists: {output}"
                )
            os.rename(temporary, output)
    except BaseException:
        if not preserve_cells:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "workflow_artifact": str(output / "workflow.json"),
        "comparison_artifact": str(output / "comparison.json"),
        "model_kind": comparison.model_kind,
        "n_folds": comparison.selection.n_folds,
        "n_candidates": comparison.selection.n_candidates,
        "candidate_n_states": comparison.selection.candidate_n_states.tolist(),
        "n_seeds": len(comparison.model_seeds),
        "model_seeds": list(comparison.model_seeds),
        "n_subjects": len(comparison.subjects),
        "n_fits": len(fit_records),
        "reused_models": reused_models,
        "reused_scores": reused_scores,
        "best_n_states": comparison.selection.best_n_states,
        "one_standard_error_n_states": (
            comparison.selection.one_standard_error_n_states
        ),
    }


def _sample_intervals_match(left: float | None, right: float | None) -> bool:
    return (left is None and right is None) or (
        left is not None
        and right is not None
        and abs(left - right) <= 1e-9
    )


def _outer_test_subjects(
    store: FeatureStore,
    requested: list[str],
    development_subjects: tuple[str, ...],
) -> tuple[str, ...]:
    if len(set(requested)) != len(requested):
        raise ValueError("outer-test subjects must be unique")
    missing = sorted(set(requested) - set(store.subjects))
    if missing:
        raise ValueError(f"requested outer-test subjects are absent: {missing}")
    overlap = sorted(set(requested).intersection(development_subjects))
    if overlap:
        raise ValueError(
            "outer-test subjects overlap the inner development cohort: "
            f"{overlap}"
        )
    selected = set(requested)
    return tuple(subject for subject in store.subjects if subject in selected)


def _fit_selected_state_count_model(
    store: FeatureStore,
    *,
    method: str,
    configuration: dict[str, object],
    n_states: int,
    seed: int,
    development_subjects: tuple[str, ...],
) -> KMeansStateModel | GaussianHMMStateModel:
    if method == "kmeans":
        return fit_minibatch_kmeans_store(
            store,
            n_states=n_states,
            seed=seed,
            n_init=int(configuration["n_init"]),
            max_iter=int(configuration["max_iter"]),
            batch_size=int(configuration["batch_size"]),
            standardize_features=bool(configuration["standardize_features"]),
            reassignment_ratio=float(configuration["reassignment_ratio"]),
            init_sample_size=(
                None
                if configuration["init_sample_size"] is None
                else int(configuration["init_sample_size"])
            ),
            subjects=development_subjects,
        ).model
    return fit_gaussian_hmm_store(
        store,
        n_states=n_states,
        seed=seed,
        n_pca_components=int(configuration["n_pca_components"]),
        covariance_type=str(configuration["covariance_type"]),
        n_init=int(configuration["n_init"]),
        n_iter=int(configuration["n_iter"]),
        tol=float(configuration["tol"]),
        pca_batch_size=int(configuration["pca_batch_size"]),
        subjects=development_subjects,
        minimum_sequence_length=int(configuration["minimum_sequence_length"]),
    ).model


def _evaluate_selected_state_count(namespace: argparse.Namespace) -> dict[str, object]:
    output = namespace.output
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"state-count evaluation output already exists: {output}")
    selection = load_state_count_cross_validation(namespace.selection_workflow)
    store = FeatureStore.open(namespace.store)
    if (
        store.feature_contract_fingerprint()
        != selection.feature_contract_fingerprint
        or store.source_contract != selection.source_contract
        or not _sample_intervals_match(
            store.sample_interval_seconds,
            selection.sample_interval_seconds,
        )
    ):
        raise ValueError(
            "FeatureStore contract does not match the inner selection workflow"
        )
    missing_development = sorted(
        set(selection.development_subjects) - set(store.subjects)
    )
    if missing_development:
        raise ValueError(
            "inner development subjects are absent from the FeatureStore: "
            f"{missing_development}"
        )
    minimum_sequence_length = (
        1
        if selection.method == "kmeans"
        else int(selection.fit_configuration["minimum_sequence_length"])
    )
    observed_development_fingerprint = store.data_fingerprint(
        subjects=selection.development_subjects,
        minimum_sequence_length=minimum_sequence_length,
    )
    if observed_development_fingerprint != selection.development_data_fingerprint:
        raise ValueError(
            "FeatureStore development data changed after inner state-count selection"
        )
    for fold in selection.comparison.folds:
        observed_fit_fingerprint = store.data_fingerprint(
            subjects=fold.fit_subjects,
            minimum_sequence_length=minimum_sequence_length,
        )
        observed_evaluation_fingerprint = store.data_fingerprint(
            subjects=fold.evaluation_subjects,
            minimum_sequence_length=minimum_sequence_length,
        )
        if (
            observed_fit_fingerprint != fold.training_data_fingerprint
            or observed_evaluation_fingerprint != fold.evaluation_data_fingerprint
        ):
            raise ValueError(
                "FeatureStore development fold data changed after inner "
                "state-count selection"
            )
    test_subjects = _outer_test_subjects(
        store,
        namespace.test_subject,
        selection.development_subjects,
    )
    test_fingerprint = store.data_fingerprint(
        subjects=test_subjects,
        minimum_sequence_length=minimum_sequence_length,
    )
    selected_n_states = selection.selected_n_states(namespace.selection)
    configuration = dict(selection.fit_configuration)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    reports = []
    model_artifacts: list[str] = []
    score_artifacts: list[str] = []
    try:
        for seed in selection.model_seeds:
            stem = f"k-{selected_n_states}_seed-{seed}"
            model_relative = Path("models") / f"{stem}.model"
            score_relative = Path("scores") / f"{stem}.json"
            model = _fit_selected_state_count_model(
                store,
                method=selection.method,
                configuration=configuration,
                n_states=selected_n_states,
                seed=seed,
                development_subjects=selection.development_subjects,
            )
            if (
                model.fit_subjects != selection.development_subjects
                or model.training_data_fingerprint
                != selection.development_data_fingerprint
            ):
                raise RuntimeError(
                    "refitted model does not match the complete development cohort"
                )
            save_fitted_model(model, temporary / model_relative)
            (
                scores,
                observed_minimum,
                omitted_short_sequence_count,
                observed_test_fingerprint,
            ) = _cross_validation_scores(store, model, test_subjects)
            if (
                observed_minimum != minimum_sequence_length
                or observed_test_fingerprint != test_fingerprint
            ):
                raise RuntimeError("outer-test data changed during state-model evaluation")
            write_state_model_scores(
                scores,
                temporary / score_relative,
                model_kind=selection.model_kind,
                model_fingerprint=fitted_model_fingerprint(model),
                model_seed=model.seed,
                n_states=model.n_states,
                fit_subjects=model.fit_subjects,
                training_data_fingerprint=model.training_data_fingerprint,
                evaluation_data_fingerprint=observed_test_fingerprint,
                feature_contract_fingerprint=store.feature_contract_fingerprint(),
                source_contract=store.source_contract,
                sample_interval_seconds=store.sample_interval_seconds,
                minimum_sequence_length=observed_minimum,
                omitted_short_sequence_count=omitted_short_sequence_count,
                allow_fit_subjects=False,
                model_specification=state_model_specification(model),
            )
            reports.append(load_state_model_scores(temporary / score_relative))
            model_artifacts.append(model_relative.as_posix())
            score_artifacts.append(score_relative.as_posix())

        write_selected_state_count_evaluation(
            reports,
            temporary / "evaluation.json",
            model_artifacts=model_artifacts,
            score_artifacts=score_artifacts,
            method=selection.method,
            selection_policy=namespace.selection,
            selection_workflow_fingerprint=selection.workflow_fingerprint,
            selected_n_states=selected_n_states,
            development_data_fingerprint=selection.development_data_fingerprint,
            development_subjects=selection.development_subjects,
            test_data_fingerprint=test_fingerprint,
            test_subjects=test_subjects,
            fit_configuration=configuration,
        )
        evaluation = json.loads(
            (temporary / "evaluation.json").read_text(encoding="utf-8")
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"state-count evaluation output already exists: {output}"
            )
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "evaluation_artifact": str(output / "evaluation.json"),
        "model_kind": selection.model_kind,
        "selection_policy": namespace.selection,
        "selected_n_states": selected_n_states,
        "n_seeds": len(selection.model_seeds),
        "model_seeds": list(selection.model_seeds),
        "n_development_subjects": len(selection.development_subjects),
        "n_test_subjects": len(test_subjects),
        "test_subjects": list(test_subjects),
        "cohort_score": evaluation["cohort_score"],
    }


def _nested_checkpoint_payload(
    namespace: argparse.Namespace,
    store: FeatureStore,
    *,
    subjects: tuple[str, ...],
    candidates: tuple[int, ...],
    model_seeds: tuple[int, ...],
    minimum_sequence_length: int,
    configuration: dict[str, object],
) -> dict[str, object]:
    return {
        "format": "dfckit-nested-state-count-checkpoint",
        "format_version": 1,
        "method": namespace.method,
        "model_kind": (
            "kmeans-state" if namespace.method == "kmeans" else "gaussian-hmm-state"
        ),
        "source_contract": store.source_contract,
        "sample_interval_seconds": store.sample_interval_seconds,
        "feature_contract_fingerprint": store.feature_contract_fingerprint(),
        "cohort_data_fingerprint": store.data_fingerprint(
            subjects=subjects,
            minimum_sequence_length=minimum_sequence_length,
        ),
        "subjects": list(subjects),
        "candidate_n_states": list(candidates),
        "model_seeds": list(model_seeds),
        "selection_policy": namespace.selection,
        "fit_configuration": configuration,
        "outer_split": {
            "algorithm": "sha256-seed-subject-balanced-v1",
            "seed": namespace.outer_split_seed,
            "n_folds": namespace.outer_n_folds,
        },
        "inner_validation": {
            "split_seed": namespace.inner_split_seed,
            "n_folds": namespace.inner_n_folds,
        },
    }


def _remove_checkpoint_temporaries(parent: Path, output_name: str) -> None:
    if not parent.is_dir():
        return
    for path in parent.glob(f".{output_name}.tmp-*"):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _checkpoint_child(root: Path, relative: Path, name: str) -> Path:
    target = root / relative
    resolved_root = root.resolve()
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"checkpoint {name} path must not contain a symlink")
    try:
        target.resolve(strict=False).relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"checkpoint {name} path resolves outside the checkpoint") from error
    return target


def _require_symlink_free_tree(root: Path, name: str) -> None:
    for parent, directories, files in os.walk(root, followlinks=False):
        base = Path(parent)
        for entry in (*directories, *files):
            if (base / entry).is_symlink():
                raise ValueError(f"checkpoint {name} must not contain symlinks")


def _write_new_json(payload: dict[str, object], target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"JSON output already exists: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"JSON output already exists: {target}")
        os.rename(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_state_count_checkpoint_json(
    payload: dict[str, object],
    target: Path,
) -> None:
    """Atomically create a selection-checkpoint summary manifest."""
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"JSON output already exists: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"JSON output already exists: {target}")
        os.rename(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _nested_cross_validate_state_counts_unlocked(
    namespace: argparse.Namespace,
) -> dict[str, object]:
    output = namespace.output
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"nested cross-validation output already exists: {output}")
    candidates = _integer_grid(
        namespace.n_states,
        "--n-states",
        minimum=2,
        minimum_count=2,
    )
    model_seeds = _integer_grid(
        namespace.seed,
        "--seed",
        minimum=0,
        minimum_count=1,
    )
    if namespace.method == "hmm" and namespace.n_pca_components is None:
        raise ValueError("--n-pca-components is required when --method hmm")
    store = FeatureStore.open(namespace.store)
    subjects = _cross_validation_subjects(store, namespace.subject)
    minimum_sequence_length = (
        1 if namespace.method == "kmeans" else namespace.minimum_sequence_length
    )
    if namespace.method == "hmm":
        eligible_subjects = {
            identity[0]
            for identity, count in store.sequence_sample_counts
            if count >= minimum_sequence_length
        }
        missing = sorted(set(subjects) - eligible_subjects)
        if missing:
            raise ValueError(
                "nested cross-validation subjects have no HMM-eligible sequence: "
                f"{missing}"
            )
    outer_folds = make_subject_validation_folds(
        subjects,
        n_folds=namespace.outer_n_folds,
        seed=namespace.outer_split_seed,
    )
    if any(len(fold.fit_subjects) < namespace.inner_n_folds for fold in outer_folds):
        raise ValueError(
            "inner_n_folds cannot exceed the number of development subjects in "
            "an outer fold"
        )
    ordered_folds = sorted(
        (
            store.data_fingerprint(
                subjects=fold.evaluation_subjects,
                minimum_sequence_length=minimum_sequence_length,
            ),
            fold,
        )
        for fold in outer_folds
    )
    configuration = _cross_validation_configuration(namespace)

    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = _nested_checkpoint_payload(
        namespace,
        store,
        subjects=subjects,
        candidates=candidates,
        model_seeds=model_seeds,
        minimum_sequence_length=minimum_sequence_length,
        configuration=configuration,
    )
    preserve_on_failure = namespace.checkpoint is not None
    if namespace.checkpoint is None:
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
        )
        checkpoint_manifest = None
    else:
        temporary = namespace.checkpoint
        if (
            temporary.resolve(strict=False) == output.resolve(strict=False)
            or temporary.parent.resolve() != output.parent.resolve()
        ):
            raise ValueError("--checkpoint must be a sibling of the final output path")
        if temporary.is_symlink() or (temporary.exists() and not temporary.is_dir()):
            raise ValueError("--checkpoint must be a real directory, not a file or symlink")
        checkpoint_manifest = temporary / "checkpoint.json"
        completed_manifest = temporary / "nested_evaluation.json"
        if temporary.exists():
            if checkpoint_manifest.is_symlink() or completed_manifest.is_symlink():
                raise ValueError("checkpoint manifests must not be symlinks")
            if checkpoint_manifest.exists() and not checkpoint_manifest.is_file():
                raise ValueError("checkpoint.json must be a regular file")
            if checkpoint_manifest.is_file():
                observed_checkpoint = load_nested_state_count_checkpoint(
                    checkpoint_manifest
                )
                if observed_checkpoint != checkpoint_payload:
                    raise ValueError(
                        "checkpoint does not match the requested workflow or current data"
                    )
            elif not completed_manifest.is_file():
                raise ValueError(
                    "existing checkpoint has neither checkpoint.json nor a complete result"
                )
        else:
            temporary.mkdir()
            write_nested_state_count_checkpoint(
                checkpoint_payload,
                checkpoint_manifest,
            )
    fold_records: list[dict[str, object]] = []
    subject_score_map: dict[str, float] = {}
    selected_counts: list[int] = []
    reused_selections = 0
    reused_evaluations = 0
    reused_inner_models = 0
    reused_inner_scores = 0
    try:
        for fold_index, (test_fingerprint, fold) in enumerate(ordered_folds):
            fold_relative = Path("outer-folds") / f"fold-{fold_index + 1:03d}"
            selection_relative = fold_relative / "selection"
            evaluation_relative = fold_relative / "evaluation"
            selection_namespace = argparse.Namespace(**vars(namespace))
            selection_namespace.output = temporary / selection_relative
            selection_namespace.n_folds = namespace.inner_n_folds
            selection_namespace.split_seed = namespace.inner_split_seed
            selection_namespace.subject = list(fold.fit_subjects)
            selection_namespace._cell_checkpoint = (
                selection_namespace.output if preserve_on_failure else None
            )
            selection_path = (
                _checkpoint_child(
                    temporary,
                    selection_relative,
                    "selection",
                )
                if preserve_on_failure
                else temporary / selection_relative
            )
            if selection_path.is_symlink():
                raise ValueError("checkpoint selection path must not be a symlink")
            if preserve_on_failure:
                _remove_checkpoint_temporaries(
                    selection_path.parent,
                    selection_path.name,
                )
            selection_workflow = selection_path / "workflow.json"
            selection_checkpoint = selection_path / STATE_COUNT_CHECKPOINT_MANIFEST
            if selection_workflow.is_file():
                _require_symlink_free_tree(selection_path, "selection artifact")
                selection = load_state_count_cross_validation(selection_path)
                if selection_checkpoint.exists():
                    selection_plan = make_state_count_checkpoint_plan(
                        store,
                        method=namespace.method,
                        subjects=fold.fit_subjects,
                        candidate_n_states=candidates,
                        model_seeds=model_seeds,
                        n_folds=namespace.inner_n_folds,
                        split_seed=namespace.inner_split_seed,
                        configuration=configuration,
                        minimum_sequence_length=minimum_sequence_length,
                    )
                    validate_state_count_checkpoint_contract(
                        selection_path,
                        selection_plan,
                    )
                    validate_completed_state_count_checkpoint(
                        selection,
                        store,
                        selection_plan,
                    )
                    selection_checkpoint.unlink()
                reused_selections += 1
            else:
                if selection_path.exists() and not selection_checkpoint.is_file():
                    raise ValueError(
                        "checkpoint selection has neither workflow.json nor "
                        "its cell checkpoint manifest"
                    )
                selection_summary = _cross_validate_state_counts(
                    selection_namespace
                )
                reused_inner_models += int(selection_summary["reused_models"])
                reused_inner_scores += int(selection_summary["reused_scores"])
                selection = load_state_count_cross_validation(selection_path)
            validate_nested_selection(
                selection,
                store,
                fold,
                method=namespace.method,
                candidates=candidates,
                model_seeds=model_seeds,
                inner_n_folds=namespace.inner_n_folds,
                inner_split_seed=namespace.inner_split_seed,
                configuration=configuration,
                minimum_sequence_length=minimum_sequence_length,
            )

            evaluation_namespace = argparse.Namespace(
                output=temporary / evaluation_relative,
                selection_workflow=selection_path,
                store=namespace.store,
                selection=namespace.selection,
                test_subject=list(fold.evaluation_subjects),
            )
            evaluation_path = (
                _checkpoint_child(
                    temporary,
                    evaluation_relative,
                    "evaluation",
                )
                if preserve_on_failure
                else temporary / evaluation_relative
            )
            if evaluation_path.is_symlink():
                raise ValueError("checkpoint evaluation path must not be a symlink")
            if preserve_on_failure:
                _remove_checkpoint_temporaries(
                    evaluation_path.parent,
                    evaluation_path.name,
                )
            if evaluation_path.exists():
                _require_symlink_free_tree(evaluation_path, "evaluation artifact")
                evaluation = load_selected_state_count_evaluation(evaluation_path)
                reused_evaluations += 1
            else:
                _evaluate_selected_state_count(evaluation_namespace)
                evaluation = load_selected_state_count_evaluation(evaluation_path)
            validate_nested_evaluation(
                evaluation,
                selection,
                store,
                fold,
                selection_policy=namespace.selection,
                test_fingerprint=test_fingerprint,
                configuration=configuration,
            )
            if evaluation.test_data_fingerprint != test_fingerprint:
                raise RuntimeError(
                    "outer-test data changed during nested cross-validation"
                )
            selected_counts.append(evaluation.selected_n_states)
            for subject, score in zip(
                evaluation.test_subjects,
                evaluation.subject_scores,
                strict=True,
            ):
                if subject in subject_score_map:
                    raise RuntimeError(
                        "nested cross-validation scored an outer-test subject twice"
                    )
                subject_score_map[subject] = score
            fold_records.append(
                {
                    "fold_index": fold_index,
                    "assignment_index": fold.fold_index,
                    "development_subjects": list(fold.fit_subjects),
                    "test_subjects": list(fold.evaluation_subjects),
                    "development_data_fingerprint": (
                        selection.development_data_fingerprint
                    ),
                    "test_data_fingerprint": evaluation.test_data_fingerprint,
                    "selection_artifact": selection_relative.as_posix(),
                    "evaluation_artifact": evaluation_relative.as_posix(),
                    "selection_workflow_fingerprint": (
                        selection.workflow_fingerprint
                    ),
                    "evaluation_fingerprint": evaluation.artifact_fingerprint,
                    "selected_n_states": evaluation.selected_n_states,
                    "cohort_score": evaluation.cohort_score,
                    "subject_scores": [
                        {"subject": subject, "score": score}
                        for subject, score in zip(
                            evaluation.test_subjects,
                            evaluation.subject_scores,
                            strict=True,
                        )
                    ],
                }
            )

        if set(subject_score_map) != set(subjects):
            raise RuntimeError(
                "nested cross-validation did not score every subject exactly once"
            )
        subject_scores = [
            {"subject": subject, "score": subject_score_map[subject]}
            for subject in subjects
        ]
        cohort_score = sum(record["score"] for record in subject_scores) / len(
            subject_scores
        )
        manifest = {
            "format": "dfckit-nested-state-count-cross-validation",
            "format_version": 1,
            "method": namespace.method,
            "model_kind": (
                "kmeans-state"
                if namespace.method == "kmeans"
                else "gaussian-hmm-state"
            ),
            "source_contract": store.source_contract,
            "sample_interval_seconds": store.sample_interval_seconds,
            "feature_contract_fingerprint": store.feature_contract_fingerprint(),
            "subjects": list(subjects),
            "n_subjects": len(subjects),
            "candidate_n_states": list(candidates),
            "model_seeds": list(model_seeds),
            "selection_policy": namespace.selection,
            "selection_metric": (
                "subject-balanced mean squared distance"
                if namespace.method == "kmeans"
                else "subject-balanced log likelihood per sample"
            ),
            "selection_direction": (
                "lower is better" if namespace.method == "kmeans" else "higher is better"
            ),
            "score_weighting": (
                "runs sample-weighted within subject and seed; seeds equal within "
                "subject; subjects equal across outer folds"
            ),
            "fit_configuration": configuration,
            "outer_split": {
                "algorithm": "sha256-seed-subject-balanced-v1",
                "seed": namespace.outer_split_seed,
                "n_folds": namespace.outer_n_folds,
            },
            "inner_validation": {
                "split_seed": namespace.inner_split_seed,
                "n_folds": namespace.inner_n_folds,
            },
            "n_outer_folds": len(fold_records),
            "n_inner_fits": (
                len(fold_records)
                * namespace.inner_n_folds
                * len(candidates)
                * len(model_seeds)
            ),
            "n_outer_models": len(fold_records) * len(model_seeds),
            "folds": fold_records,
            "selected_state_counts": selected_counts,
            "subject_scores": subject_scores,
            "cohort_score": cohort_score,
        }
        nested_path = (
            _checkpoint_child(
                temporary,
                Path("nested_evaluation.json"),
                "final manifest",
            )
            if preserve_on_failure
            else temporary / "nested_evaluation.json"
        )
        if preserve_on_failure:
            _remove_checkpoint_temporaries(nested_path.parent, nested_path.name)
        if nested_path.exists() or nested_path.is_symlink():
            nested = load_nested_state_count_cross_validation(temporary)
            observed_manifest = json.loads(nested_path.read_text(encoding="utf-8"))
            if observed_manifest != manifest:
                raise ValueError(
                    "completed checkpoint does not match the reconstructed workflow"
                )
        else:
            _write_new_json(manifest, nested_path)
            nested = load_nested_state_count_cross_validation(temporary)
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"nested cross-validation output already exists: {output}"
            )
        _require_symlink_free_tree(temporary, "tree")
        if checkpoint_manifest is not None:
            checkpoint_manifest.unlink(missing_ok=True)
        os.rename(temporary, output)
    except BaseException:
        if not preserve_on_failure:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "nested_evaluation_artifact": str(output / "nested_evaluation.json"),
        "model_kind": nested.model_kind,
        "selection_policy": nested.selection_policy,
        "n_outer_folds": len(nested.folds),
        "n_inner_fits": (
            len(nested.folds)
            * namespace.inner_n_folds
            * len(nested.candidate_n_states)
            * len(nested.model_seeds)
        ),
        "n_outer_models": len(nested.folds) * len(nested.model_seeds),
        "n_subjects": len(nested.subjects),
        "candidate_n_states": list(nested.candidate_n_states),
        "model_seeds": list(nested.model_seeds),
        "selected_state_counts": list(nested.selected_state_counts),
        "cohort_score": nested.cohort_score,
        "workflow_fingerprint": nested.workflow_fingerprint,
        "reused_selections": reused_selections,
        "reused_evaluations": reused_evaluations,
        "reused_inner_models": reused_inner_models,
        "reused_inner_scores": reused_inner_scores,
    }


def _nested_cross_validate_state_counts(
    namespace: argparse.Namespace,
) -> dict[str, object]:
    if namespace.checkpoint is None:
        return _nested_cross_validate_state_counts_unlocked(namespace)
    output = namespace.output
    checkpoint = namespace.checkpoint
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"nested cross-validation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if (
        checkpoint.resolve(strict=False) == output.resolve(strict=False)
        or checkpoint.parent.resolve() != output.parent.resolve()
    ):
        raise ValueError("--checkpoint must be a sibling of the final output path")
    with acquire_nested_checkpoint_lock(checkpoint):
        return _nested_cross_validate_state_counts_unlocked(namespace)


def _inspect_nested_state_counts(namespace: argparse.Namespace) -> dict[str, object]:
    store = FeatureStore.open(namespace.store)
    progress = inspect_nested_state_count_progress(store, namespace.workflow)
    return nested_state_count_progress_payload(progress)


def _align_states(namespace: argparse.Namespace) -> dict[str, object]:
    reference = load_fitted_model(namespace.reference_model)
    candidate = load_fitted_model(namespace.candidate_model)
    if type(reference) is not type(candidate):
        raise TypeError("reference and candidate models must use the same state-model family")
    if isinstance(reference, KMeansStateModel):
        alignment = align_kmeans_centroids(reference, candidate)
        aligned_model = relabel_kmeans_model(candidate, alignment)
        model_kind = "kmeans-state"
    elif isinstance(reference, GaussianHMMStateModel):
        alignment = align_gaussian_hmm_emissions(reference, candidate)
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
        candidate_fingerprint = fitted_model_fingerprint(candidate)
        if predictions.model_fingerprint != candidate_fingerprint:
            raise ValueError("prediction artifact fingerprint does not match the candidate model")
        aligned_fingerprint = fitted_model_fingerprint(aligned_model)
        if model_kind == "kmeans-state":
            aligned_assignments = apply_state_alignment(predictions.assignments, alignment)
            aligned_predictions = StatePredictions(
                assignments=aligned_assignments,
                model_kind=model_kind,
                model_seed=candidate.seed,
                model_fingerprint=aligned_fingerprint,
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
                model_fingerprint=aligned_fingerprint,
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
        "matched_correlations": alignment.matched_correlations.tolist(),
        "output_model": str(model_path),
        "alignment_output": str(alignment_path),
        "predictions_output": None if prediction_path is None else str(prediction_path),
    }


def _repeated_fit_contract(model: KMeansStateModel | GaussianHMMStateModel) -> tuple[object, ...]:
    common = (
        type(model),
        model.feature_keys,
        model.source_contract,
        model.sample_interval_seconds,
        model.n_states,
        model.fit_subjects,
        model.fit_sample_count,
        model.training_data_fingerprint,
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
        raise ValueError(
            "out-of-core KMeans stability requires a recorded init_sample_size"
        )
    if (
        isinstance(model, GaussianHMMStateModel)
        and "IncrementalPCA" in model.implementation
        and model.pca_batch_size is None
    ):
        raise ValueError(
            "out-of-core HMM stability requires a recorded pca_batch_size"
        )
    if model.training_data_fingerprint is None:
        raise ValueError(
            "state stability requires a recorded training_data_fingerprint"
        )


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


def _summarize_stability(namespace: argparse.Namespace) -> dict[str, object]:
    if namespace.output.exists() or namespace.output.is_symlink():
        raise FileExistsError(
            f"state-stability output path already exists: {namespace.output}"
        )
    store = FeatureStore.open(namespace.store)
    subjects = _selected_store_subjects(store, namespace.subject)
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
    if any(
        candidate.fit_sample_count != reference.fit_sample_count
        for candidate in candidates
    ):
        raise ValueError("stability models must use the same fitted sample count")
    if any(
        candidate.training_data_fingerprint != reference.training_data_fingerprint
        for candidate in candidates
    ):
        raise ValueError("stability models must use the same exact training data")
    reference_contract = _repeated_fit_contract(reference)
    if any(_repeated_fit_contract(candidate) != reference_contract for candidate in candidates):
        raise ValueError(
            "stability models must share training subjects, fitted samples, and hyperparameters"
        )
    fingerprints = tuple(
        fitted_model_fingerprint(model) for model in (reference, *candidates)
    )
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("stability models must be distinct fitted artifacts")

    reference_assignments = _decode_assignments(
        reference,
        store,
        subjects=subjects,
        allow_fit_subjects=namespace.allow_fit_subjects,
    )
    aligned_assignments = [reference_assignments]
    alignments = []
    for candidate in candidates:
        candidate_assignments = _decode_assignments(
            candidate,
            store,
            subjects=subjects,
            allow_fit_subjects=namespace.allow_fit_subjects,
        )
        if isinstance(reference, KMeansStateModel):
            alignment = align_kmeans_centroids(reference, candidate)
        else:
            alignment = align_gaussian_hmm_emissions(reference, candidate)
        alignments.append(alignment)
        aligned_assignments.append(
            apply_state_alignment(candidate_assignments, alignment)
        )
    stability = summarize_state_stability(aligned_assignments)
    model_kind = (
        "kmeans-state"
        if isinstance(reference, KMeansStateModel)
        else "gaussian-hmm-state"
    )
    output = write_state_stability(
        stability,
        namespace.output,
        model_kind=model_kind,
        reference_model_fingerprint=fingerprints[0],
        reference_seed=reference.seed,
        candidate_model_fingerprints=fingerprints[1:],
        candidate_seeds=tuple(candidate.seed for candidate in candidates),
        alignments=tuple(alignments),
        training_data_fingerprint=reference.training_data_fingerprint,
        source_contract=store.source_contract,
        sample_interval_seconds=store.sample_interval_seconds,
        allow_fit_subjects=namespace.allow_fit_subjects,
    )
    return {
        "output": str(output),
        "model_kind": model_kind,
        "reference_model_fingerprint": fingerprints[0],
        "reference_seed": reference.seed,
        "candidate_model_fingerprints": list(fingerprints[1:]),
        "candidate_seeds": [candidate.seed for candidate in candidates],
        "subjects": list(dict.fromkeys(run.subject for run in stability)),
        "n_fits": len(aligned_assignments),
        "n_states": reference.n_states,
        "n_runs": len(stability),
        "n_sequences": sum(run.n_sequences for run in stability),
        "n_samples": sum(run.n_samples for run in stability),
        "allow_fit_subjects": bool(namespace.allow_fit_subjects),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dfc-kit",
        description="Censor-aware XCP-D to dynamic-connectivity workflows",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser(
        "inspect-xcpd",
        help="list matching XCP-D acquisitions without loading time series",
    )
    _add_xcpd_arguments(inspect)

    build = subparsers.add_parser(
        "build-store",
        help="load XCP-D acquisitions and stream a feature store",
    )
    _add_xcpd_arguments(build)
    _add_load_arguments(build)
    build.add_argument("output", type=Path, help="new FeatureStore directory")
    build.add_argument(
        "--method",
        choices=("window-fc", "cap", "ets", "leida", "mtd"),
        required=True,
    )
    build.add_argument("--chunk-size", type=int, default=128)
    build.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    build.add_argument("--window-length", type=int, default=56)
    build.add_argument("--window-step", type=int, default=8)
    build.add_argument("--taper", choices=("hamming", "uniform"), default="hamming")
    build.add_argument("--minimum-segment-length", type=int, default=20)

    information = subparsers.add_parser(
        "fixed-information",
        help="estimate fixed-length block MI/CMI from XCP-D acquisitions",
    )
    _add_xcpd_arguments(information)
    _add_load_arguments(information)
    information.add_argument(
        "output",
        type=Path,
        help="new fixed-information artifact directory",
    )
    information.add_argument(
        "--information-groups",
        type=Path,
        required=True,
        help="standalone JSON declaring named left, right, and conditioning ROI groups",
    )
    information.add_argument(
        "--length",
        action="append",
        type=int,
        required=True,
        help="fixed retained-frame length; repeat for a sensitivity set",
    )
    information.add_argument("--draws", type=int, required=True)
    information.add_argument("--sample-seed", type=int, required=True)
    information.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="validated execution parallelism (default: 1)",
    )
    information.add_argument(
        "--window-schedule",
        type=Path,
        help="optional strict TSV freezing acquisition, length, draw, and original frame bounds",
    )
    information.add_argument("--k", type=int, default=3)
    information.add_argument("--jitter", type=float, default=1e-10)
    information.add_argument("--jitter-seed", type=int, default=20260811)
    information.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="z-score each ROI within each fixed window (default: enabled)",
    )

    fit = subparsers.add_parser(
        "fit-states",
        help="fit a portable KMeans or Gaussian HMM model from a FeatureStore",
    )
    fit.add_argument("store", type=Path, help="existing FeatureStore directory")
    fit.add_argument("output", type=Path, help="new fitted-model artifact directory")
    fit.add_argument("--method", choices=("kmeans", "hmm"), required=True)
    fit.add_argument("--n-states", type=int, required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument(
        "--subject",
        action="append",
        help="fit only this subject; repeat for an explicit training cohort",
    )
    fit.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="initializations (default: 10 for KMeans, 1 for HMM)",
    )
    fit.add_argument("--standardize-features", action=argparse.BooleanOptionalAction, default=True)
    fit.add_argument("--fitting-mode", choices=("streaming", "materialized"), default="streaming")
    fit.add_argument("--algorithm", choices=("lloyd", "minibatch"), default="minibatch")
    fit.add_argument("--batch-size", type=int, default=4096)
    fit.add_argument("--max-iter", type=int, default=10)
    fit.add_argument("--reassignment-ratio", type=float, default=0.01)
    fit.add_argument("--init-sample-size", type=int, default=None)
    fit.add_argument("--n-pca-components", type=int, default=None)
    fit.add_argument("--covariance-type", choices=("diag", "full"), default="diag")
    fit.add_argument("--n-iter", type=int, default=200)
    fit.add_argument("--tol", type=float, default=1e-3)
    fit.add_argument("--pca-batch-size", type=int, default=4096)
    fit.add_argument("--minimum-sequence-length", type=int, default=2)

    predict = subparsers.add_parser(
        "predict-states",
        help="decode a FeatureStore with a saved KMeans or Gaussian HMM",
    )
    predict.add_argument("store", type=Path, help="FeatureStore to decode")
    predict.add_argument("model", type=Path, help="saved fitted-model artifact")
    predict.add_argument("output", type=Path, help="new state-prediction artifact directory")
    predict.add_argument(
        "--subject",
        action="append",
        help="decode only this subject; repeat for an explicit held-out cohort",
    )
    predict.add_argument(
        "--allow-fit-subjects",
        action="store_true",
        help="explicitly allow decoding subjects used during model fitting",
    )

    summarize = subparsers.add_parser(
        "summarize-states",
        help="write occupancy, dwell, switch, and transition metrics as JSON",
    )
    summarize.add_argument("predictions", type=Path, help="state-prediction artifact directory")
    summarize.add_argument("output", type=Path, help="new state-metrics JSON file")

    score = subparsers.add_parser(
        "score-states",
        help="score a saved KMeans or HMM on held-out FeatureStore acquisitions",
    )
    score.add_argument("store", type=Path, help="FeatureStore to score")
    score.add_argument("model", type=Path, help="saved fitted-model artifact")
    score.add_argument("output", type=Path, help="new held-out score JSON file")
    score.add_argument(
        "--subject",
        action="append",
        help="score only this subject; repeat for an explicit held-out cohort",
    )
    score.add_argument(
        "--allow-fit-subjects",
        action="store_true",
        help="explicitly allow scoring subjects used during model fitting",
    )

    compare = subparsers.add_parser(
        "compare-state-counts",
        help="compare complete candidate-state sets across subject-disjoint folds",
    )
    compare.add_argument("output", type=Path, help="new state-count comparison JSON file")
    compare.add_argument(
        "--score",
        action="append",
        type=Path,
        required=True,
        help=(
            "v2 score-states JSON artifact; repeat for every fold, state count, "
            "and complete-fit seed"
        ),
    )

    cross_validate = subparsers.add_parser(
        "cross-validate-state-counts",
        help="fit, score, and compare a complete subject-level state-count grid",
    )
    cross_validate.add_argument("store", type=Path, help="FeatureStore to partition")
    cross_validate.add_argument(
        "output",
        type=Path,
        help="new workflow directory containing models, scores, and comparison",
    )
    cross_validate.add_argument("--method", choices=("kmeans", "hmm"), required=True)
    cross_validate.add_argument(
        "--n-states",
        action="append",
        type=int,
        required=True,
        help="candidate state count; repeat for at least two values",
    )
    cross_validate.add_argument(
        "--seed",
        action="append",
        type=int,
        required=True,
        help="complete-fit model seed; repeat for every prespecified seed",
    )
    cross_validate.add_argument("--n-folds", type=int, default=5)
    cross_validate.add_argument("--split-seed", type=int, required=True)
    cross_validate.add_argument(
        "--subject",
        action="append",
        help="development subject; repeat to restrict an outer-training cohort",
    )
    cross_validate.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="initializations per fit (default: 10 for KMeans, 1 for HMM)",
    )
    cross_validate.add_argument(
        "--standardize-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    cross_validate.add_argument("--batch-size", type=int, default=4096)
    cross_validate.add_argument("--max-iter", type=int, default=10)
    cross_validate.add_argument("--reassignment-ratio", type=float, default=0.01)
    cross_validate.add_argument("--init-sample-size", type=int, default=None)
    cross_validate.add_argument("--n-pca-components", type=int, default=None)
    cross_validate.add_argument(
        "--covariance-type",
        choices=("diag", "full"),
        default="diag",
    )
    cross_validate.add_argument("--n-iter", type=int, default=200)
    cross_validate.add_argument("--tol", type=float, default=1e-3)
    cross_validate.add_argument("--pca-batch-size", type=int, default=4096)
    cross_validate.add_argument("--minimum-sequence-length", type=int, default=2)

    nested = subparsers.add_parser(
        "nested-cross-validate-state-counts",
        help="select state count inside every outer fold and aggregate outer-test scores",
    )
    nested.add_argument("store", type=Path, help="FeatureStore to partition")
    nested.add_argument(
        "output",
        type=Path,
        help="new nested workflow directory containing every inner and outer artifact",
    )
    nested.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "sibling working directory to preserve and strictly resume after failure; "
            "consumed into output on success"
        ),
    )
    nested.add_argument("--method", choices=("kmeans", "hmm"), required=True)
    nested.add_argument(
        "--n-states",
        action="append",
        type=int,
        required=True,
        help="candidate state count; repeat for at least two values",
    )
    nested.add_argument(
        "--seed",
        action="append",
        type=int,
        required=True,
        help="complete-fit model seed; repeat for every prespecified seed",
    )
    nested.add_argument("--outer-n-folds", type=int, default=5)
    nested.add_argument("--outer-split-seed", type=int, required=True)
    nested.add_argument("--inner-n-folds", type=int, default=5)
    nested.add_argument("--inner-split-seed", type=int, required=True)
    nested.add_argument(
        "--selection",
        choices=("one-standard-error", "best"),
        default="one-standard-error",
        help="inner decision refitted in each outer fold (default: one-standard-error)",
    )
    nested.add_argument(
        "--subject",
        action="append",
        help="cohort subject; repeat to restrict the complete nested evaluation",
    )
    nested.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="initializations per fit (default: 10 for KMeans, 1 for HMM)",
    )
    nested.add_argument(
        "--standardize-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    nested.add_argument("--batch-size", type=int, default=4096)
    nested.add_argument("--max-iter", type=int, default=10)
    nested.add_argument("--reassignment-ratio", type=float, default=0.01)
    nested.add_argument("--init-sample-size", type=int, default=None)
    nested.add_argument("--n-pca-components", type=int, default=None)
    nested.add_argument(
        "--covariance-type",
        choices=("diag", "full"),
        default="diag",
    )
    nested.add_argument("--n-iter", type=int, default=200)
    nested.add_argument("--tol", type=float, default=1e-3)
    nested.add_argument("--pca-batch-size", type=int, default=4096)
    nested.add_argument("--minimum-sequence-length", type=int, default=2)

    inspect_nested = subparsers.add_parser(
        "inspect-nested-state-counts",
        help="strictly audit progress in a nested checkpoint or completed workflow",
    )
    inspect_nested.add_argument(
        "store",
        type=Path,
        help="current FeatureStore used by the nested workflow",
    )
    inspect_nested.add_argument(
        "workflow",
        type=Path,
        help="checkpoint or completed nested workflow directory",
    )

    evaluate = subparsers.add_parser(
        "evaluate-selected-state-count",
        help="refit an inner-selected K on development subjects and score outer-test subjects",
    )
    evaluate.add_argument("store", type=Path, help="FeatureStore containing both cohorts")
    evaluate.add_argument(
        "selection_workflow",
        type=Path,
        help="completed cross-validate-state-counts workflow directory",
    )
    evaluate.add_argument(
        "output",
        type=Path,
        help="new directory containing full-development models and outer-test scores",
    )
    evaluate.add_argument(
        "--selection",
        choices=("one-standard-error", "best"),
        default="one-standard-error",
        help="inner state-count decision to refit (default: one-standard-error)",
    )
    evaluate.add_argument(
        "--test-subject",
        action="append",
        required=True,
        help="outer-test subject; repeat for an explicit untouched test cohort",
    )

    align = subparsers.add_parser(
        "align-states",
        help="align a candidate KMeans/HMM model and optional predictions to a reference",
    )
    align.add_argument("reference_model", type=Path, help="reference fitted-model artifact")
    align.add_argument("candidate_model", type=Path, help="candidate fitted-model artifact")
    align.add_argument("output_model", type=Path, help="new relabeled model artifact")
    align.add_argument("alignment_output", type=Path, help="new alignment artifact directory")
    align.add_argument(
        "--predictions",
        type=Path,
        help="candidate state-prediction artifact to relabel",
    )
    align.add_argument(
        "--predictions-output",
        type=Path,
        help="new aligned state-prediction artifact directory",
    )
    stability = subparsers.add_parser(
        "summarize-stability",
        help="decode, align, and summarize repeated-seed state models",
    )
    stability.add_argument("store", type=Path, help="FeatureStore to decode identically")
    stability.add_argument("reference_model", type=Path, help="reference fitted-model artifact")
    stability.add_argument("output", type=Path, help="new repeated-fit stability JSON file")
    stability.add_argument(
        "--candidate-model",
        action="append",
        type=Path,
        required=True,
        help="candidate fitted model; repeat for every additional seed",
    )
    stability.add_argument(
        "--subject",
        action="append",
        help="decode only this subject; repeat for an explicit held-out cohort",
    )
    stability.add_argument(
        "--allow-fit-subjects",
        action="store_true",
        help="explicitly allow decoding subjects used during model fitting",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and return a process-style status code."""
    parser = _parser()
    namespace = parser.parse_args(argv)
    try:
        if namespace.command == "inspect-xcpd":
            result = _inspect(namespace)
        elif namespace.command == "build-store":
            result = _build_store(namespace)
        elif namespace.command == "fixed-information":
            result = _fixed_information(namespace)
        elif namespace.command == "fit-states":
            result = _fit_states(namespace)
        elif namespace.command == "predict-states":
            result = _predict_states(namespace)
        elif namespace.command == "summarize-states":
            result = _summarize_states(namespace)
        elif namespace.command == "score-states":
            result = _score_states(namespace)
        elif namespace.command == "compare-state-counts":
            result = _compare_state_counts(namespace)
        elif namespace.command == "cross-validate-state-counts":
            result = _cross_validate_state_counts(namespace)
        elif namespace.command == "nested-cross-validate-state-counts":
            result = _nested_cross_validate_state_counts(namespace)
        elif namespace.command == "inspect-nested-state-counts":
            result = _inspect_nested_state_counts(namespace)
        elif namespace.command == "evaluate-selected-state-count":
            result = _evaluate_selected_state_count(namespace)
        elif namespace.command == "align-states":
            result = _align_states(namespace)
        else:
            result = _summarize_stability(namespace)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"dfc-kit: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
