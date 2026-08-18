"""Read-only progress audits for nested state-count workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..states.cross_validation import SubjectValidationFold, make_subject_validation_folds
from ..storage import FeatureStore
from .state_cross_validation import load_state_count_cross_validation
from .state_cross_validation_checkpoint import (
    MANIFEST_NAME as STATE_COUNT_CHECKPOINT_MANIFEST,
)
from .state_cross_validation_checkpoint import (
    inspect_state_count_checkpoint,
    make_state_count_checkpoint_plan,
    validate_completed_state_count_checkpoint,
    validate_state_count_checkpoint_contract,
)
from .state_evaluation import load_selected_state_count_evaluation
from .state_nested_checkpoint import load_nested_state_count_checkpoint
from .state_nested_contract import (
    sample_intervals_match,
    validate_nested_evaluation,
    validate_nested_selection,
)
from .state_nested_cross_validation import (
    NestedStateCountCrossValidationArtifact,
    load_nested_state_count_cross_validation,
)
from .state_nested_lock import (
    NestedCheckpointLockStatus,
    inspect_nested_checkpoint_lock,
)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    output = float(value)
    if not np.isfinite(output) or (positive and output <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return output


def _subjects(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("checkpoint subjects must be a non-empty array")
    output = tuple(value)
    if (
        any(not isinstance(subject, str) or not subject.strip() for subject in output)
        or len(set(output)) != len(output)
    ):
        raise ValueError("checkpoint subjects must contain unique non-empty labels")
    return output


def _integer_grid(
    value: object,
    name: str,
    *,
    minimum: int,
    minimum_count: int,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) < minimum_count:
        raise ValueError(f"{name} must contain at least {minimum_count} values")
    output = tuple(_integer(item, name, minimum=minimum) for item in value)
    if len(set(output)) != len(output) or tuple(sorted(output)) != output:
        raise ValueError(f"{name} must be unique and increasing")
    return output


def _fit_configuration(value: object, method: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("checkpoint fit_configuration must be an object")
    expected = (
        {
            "batch_size",
            "init_sample_size",
            "max_iter",
            "n_init",
            "reassignment_ratio",
            "standardize_features",
        }
        if method == "kmeans"
        else {
            "covariance_type",
            "minimum_sequence_length",
            "n_init",
            "n_iter",
            "n_pca_components",
            "pca_batch_size",
            "tol",
        }
    )
    if set(value) != expected:
        raise ValueError("checkpoint fit_configuration fields disagree with the method")
    output = dict(value)
    if method == "kmeans":
        for name in ("batch_size", "max_iter", "n_init"):
            output[name] = _integer(output[name], name, minimum=1)
        requested = output["init_sample_size"]
        if requested is not None:
            output["init_sample_size"] = _integer(
                requested,
                "init_sample_size",
                minimum=1,
            )
        if not isinstance(output["standardize_features"], bool):
            raise TypeError("standardize_features must be boolean")
        ratio = _finite_float(output["reassignment_ratio"], "reassignment_ratio")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("reassignment_ratio must be within [0, 1]")
        output["reassignment_ratio"] = ratio
    else:
        for name in (
            "minimum_sequence_length",
            "n_init",
            "n_iter",
            "n_pca_components",
            "pca_batch_size",
        ):
            output[name] = _integer(output[name], name, minimum=1)
        if output["minimum_sequence_length"] < 2:
            raise ValueError("minimum_sequence_length must be at least two for HMM")
        if output["covariance_type"] not in {"diag", "full"}:
            raise ValueError("checkpoint covariance_type is invalid")
        output["tol"] = _finite_float(output["tol"], "tol", positive=True)
    return output


def _require_symlink_free_tree(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("nested progress root must not be a symlink")
    for parent, directories, files in os.walk(root, followlinks=False):
        base = Path(parent)
        for entry in (*directories, *files):
            if (base / entry).is_symlink():
                raise ValueError("nested progress tree must not contain symlinks")


def _temporary_paths(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob(".*.tmp-*")
        )
    )


@dataclass(frozen=True)
class NestedStateCountFoldProgress:
    """Strictly audited completion state for one outer fold."""

    fold_index: int
    assignment_index: int
    status: str
    development_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    selection_complete: bool
    evaluation_complete: bool
    completed_inner_models: int
    completed_inner_scores: int
    total_inner_cells: int
    selected_n_states: int | None
    cohort_score: float | None


@dataclass(frozen=True)
class NestedStateCountProgress:
    """Read-only progress summary backed only by validated artifacts."""

    root: Path
    status: str
    method: str
    model_kind: str
    subjects: tuple[str, ...]
    candidate_n_states: tuple[int, ...]
    model_seeds: tuple[int, ...]
    selection_policy: str
    n_outer_folds: int
    n_inner_folds: int
    outer_split_seed: int
    inner_split_seed: int
    execution_status: str
    lock_path: Path
    lock_owner_pid: int | None
    lock_owner_hostname: str | None
    lock_owner_process_start_token: str | None
    lock_acquired_at_unix: float | None
    lock_released_at_unix: float | None
    completed_selections: int
    completed_evaluations: int
    completed_inner_fits: int
    total_inner_fits: int
    completed_inner_scores: int
    total_inner_scores: int
    completed_outer_models: int
    total_outer_models: int
    completed_fit_steps: int
    total_fit_steps: int
    fit_completion_fraction: float
    temporary_paths: tuple[str, ...]
    folds: tuple[NestedStateCountFoldProgress, ...]
    workflow_fingerprint: str | None
    cohort_score: float | None


@dataclass(frozen=True)
class _CheckpointPlan:
    method: str
    model_kind: str
    subjects: tuple[str, ...]
    candidates: tuple[int, ...]
    model_seeds: tuple[int, ...]
    selection_policy: str
    configuration: dict[str, object]
    minimum_sequence_length: int
    outer_folds: tuple[tuple[str, SubjectValidationFold], ...]
    outer_split_seed: int
    inner_n_folds: int
    inner_split_seed: int


def _checkpoint_plan(
    payload: dict[str, object],
    store: FeatureStore,
) -> _CheckpointPlan:
    method = payload["method"]
    if method not in {"kmeans", "hmm"}:
        raise ValueError("checkpoint method is invalid")
    model_kind = "kmeans-state" if method == "kmeans" else "gaussian-hmm-state"
    if payload["model_kind"] != model_kind:
        raise ValueError("checkpoint method and model kind disagree")
    subjects = _subjects(payload["subjects"])
    candidates = _integer_grid(
        payload["candidate_n_states"],
        "candidate_n_states",
        minimum=2,
        minimum_count=2,
    )
    model_seeds = _integer_grid(
        payload["model_seeds"],
        "model_seeds",
        minimum=0,
        minimum_count=1,
    )
    selection_policy = payload["selection_policy"]
    if selection_policy not in {"one-standard-error", "best"}:
        raise ValueError("checkpoint selection_policy is invalid")
    configuration = _fit_configuration(payload["fit_configuration"], method)
    minimum_sequence_length = (
        1 if method == "kmeans" else int(configuration["minimum_sequence_length"])
    )
    sample_interval = payload["sample_interval_seconds"]
    if sample_interval is not None:
        sample_interval = _finite_float(
            sample_interval,
            "sample_interval_seconds",
            positive=True,
        )
    if (
        payload["source_contract"] != store.source_contract
        or not sample_intervals_match(sample_interval, store.sample_interval_seconds)
        or payload["feature_contract_fingerprint"]
        != store.feature_contract_fingerprint()
        or payload["cohort_data_fingerprint"]
        != store.data_fingerprint(
            subjects=subjects,
            minimum_sequence_length=minimum_sequence_length,
        )
    ):
        raise ValueError("checkpoint does not match the current FeatureStore")

    outer = payload["outer_split"]
    if not isinstance(outer, dict) or set(outer) != {"algorithm", "n_folds", "seed"}:
        raise ValueError("checkpoint outer_split fields do not match the schema")
    if outer["algorithm"] != "sha256-seed-subject-balanced-v1":
        raise ValueError("checkpoint outer split algorithm is unsupported")
    outer_n_folds = _integer(outer["n_folds"], "outer n_folds", minimum=2)
    outer_seed = _integer(outer["seed"], "outer split seed")
    folds = make_subject_validation_folds(
        subjects,
        n_folds=outer_n_folds,
        seed=outer_seed,
    )

    inner = payload["inner_validation"]
    if not isinstance(inner, dict) or set(inner) != {"n_folds", "split_seed"}:
        raise ValueError("checkpoint inner_validation fields do not match the schema")
    inner_n_folds = _integer(inner["n_folds"], "inner n_folds", minimum=2)
    inner_seed = _integer(inner["split_seed"], "inner split seed")
    if any(len(fold.fit_subjects) < inner_n_folds for fold in folds):
        raise ValueError("checkpoint inner fold count exceeds an outer development cohort")
    eligible = {
        identity[0]
        for identity, count in store.sequence_sample_counts
        if count >= minimum_sequence_length
    }
    missing = sorted(set(subjects) - eligible)
    if missing:
        raise ValueError(f"checkpoint subjects lack eligible sequences: {missing}")
    ordered = tuple(
        sorted(
            (
                store.data_fingerprint(
                    subjects=fold.evaluation_subjects,
                    minimum_sequence_length=minimum_sequence_length,
                ),
                fold,
            )
            for fold in folds
        )
    )
    return _CheckpointPlan(
        method=method,
        model_kind=model_kind,
        subjects=subjects,
        candidates=candidates,
        model_seeds=model_seeds,
        selection_policy=selection_policy,
        configuration=configuration,
        minimum_sequence_length=minimum_sequence_length,
        outer_folds=ordered,
        outer_split_seed=outer_seed,
        inner_n_folds=inner_n_folds,
        inner_split_seed=inner_seed,
    )


def _progress(
    *,
    root: Path,
    status: str,
    method: str,
    model_kind: str,
    subjects: tuple[str, ...],
    candidates: tuple[int, ...],
    model_seeds: tuple[int, ...],
    selection_policy: str,
    inner_n_folds: int,
    outer_split_seed: int,
    inner_split_seed: int,
    lock_status: NestedCheckpointLockStatus,
    folds: tuple[NestedStateCountFoldProgress, ...],
    temporary_paths: tuple[str, ...],
    workflow_fingerprint: str | None,
    cohort_score: float | None,
) -> NestedStateCountProgress:
    completed_selections = sum(fold.selection_complete for fold in folds)
    completed_evaluations = sum(fold.evaluation_complete for fold in folds)
    fits_per_selection = inner_n_folds * len(candidates) * len(model_seeds)
    models_per_evaluation = len(model_seeds)
    completed_inner_fits = sum(fold.completed_inner_models for fold in folds)
    total_inner_fits = len(folds) * fits_per_selection
    completed_inner_scores = sum(fold.completed_inner_scores for fold in folds)
    total_inner_scores = total_inner_fits
    completed_outer_models = completed_evaluations * models_per_evaluation
    total_outer_models = len(folds) * models_per_evaluation
    completed_fit_steps = completed_inner_fits + completed_outer_models
    total_fit_steps = total_inner_fits + total_outer_models
    return NestedStateCountProgress(
        root=root,
        status=status,
        method=method,
        model_kind=model_kind,
        subjects=subjects,
        candidate_n_states=candidates,
        model_seeds=model_seeds,
        selection_policy=selection_policy,
        n_outer_folds=len(folds),
        n_inner_folds=inner_n_folds,
        outer_split_seed=outer_split_seed,
        inner_split_seed=inner_split_seed,
        execution_status=lock_status.status,
        lock_path=lock_status.path,
        lock_owner_pid=lock_status.pid,
        lock_owner_hostname=lock_status.hostname,
        lock_owner_process_start_token=lock_status.process_start_token,
        lock_acquired_at_unix=lock_status.acquired_at_unix,
        lock_released_at_unix=lock_status.released_at_unix,
        completed_selections=completed_selections,
        completed_evaluations=completed_evaluations,
        completed_inner_fits=completed_inner_fits,
        total_inner_fits=total_inner_fits,
        completed_inner_scores=completed_inner_scores,
        total_inner_scores=total_inner_scores,
        completed_outer_models=completed_outer_models,
        total_outer_models=total_outer_models,
        completed_fit_steps=completed_fit_steps,
        total_fit_steps=total_fit_steps,
        fit_completion_fraction=completed_fit_steps / total_fit_steps,
        temporary_paths=temporary_paths,
        folds=folds,
        workflow_fingerprint=workflow_fingerprint,
        cohort_score=cohort_score,
    )


def _completed_progress(
    root: Path,
    store: FeatureStore,
    nested: NestedStateCountCrossValidationArtifact,
    *,
    status: str,
    lock_status: NestedCheckpointLockStatus,
) -> NestedStateCountProgress:
    minimum_sequence_length = (
        1
        if nested.method == "kmeans"
        else _integer(
            nested.fit_configuration["minimum_sequence_length"],
            "minimum_sequence_length",
            minimum=2,
        )
    )
    if (
        nested.source_contract != store.source_contract
        or nested.feature_contract_fingerprint != store.feature_contract_fingerprint()
        or not sample_intervals_match(
            nested.sample_interval_seconds,
            store.sample_interval_seconds,
        )
    ):
        raise ValueError("completed nested workflow does not match the current FeatureStore")
    assignments = {
        fold.fold_index: fold
        for fold in make_subject_validation_folds(
            nested.subjects,
            n_folds=len(nested.folds),
            seed=nested.outer_split_seed,
        )
    }
    fold_progress: list[NestedStateCountFoldProgress] = []
    for nested_fold in nested.folds:
        fold = assignments[nested_fold.assignment_index]
        test_fingerprint = store.data_fingerprint(
            subjects=fold.evaluation_subjects,
            minimum_sequence_length=minimum_sequence_length,
        )
        validate_nested_selection(
            nested_fold.selection,
            store,
            fold,
            method=nested.method,
            candidates=nested.candidate_n_states,
            model_seeds=nested.model_seeds,
            inner_n_folds=nested_fold.selection.comparison.selection.n_folds,
            inner_split_seed=nested.inner_split_seed,
            configuration=dict(nested.fit_configuration),
            minimum_sequence_length=minimum_sequence_length,
        )
        validate_nested_evaluation(
            nested_fold.evaluation,
            nested_fold.selection,
            store,
            fold,
            selection_policy=nested.selection_policy,
            test_fingerprint=test_fingerprint,
            configuration=dict(nested.fit_configuration),
        )
        fold_progress.append(
            NestedStateCountFoldProgress(
                fold_index=nested_fold.fold_index,
                assignment_index=nested_fold.assignment_index,
                status="complete",
                development_subjects=fold.fit_subjects,
                test_subjects=fold.evaluation_subjects,
                selection_complete=True,
                evaluation_complete=True,
                completed_inner_models=(
                    nested_fold.selection.comparison.selection.n_folds
                    * len(nested.candidate_n_states)
                    * len(nested.model_seeds)
                ),
                completed_inner_scores=(
                    nested_fold.selection.comparison.selection.n_folds
                    * len(nested.candidate_n_states)
                    * len(nested.model_seeds)
                ),
                total_inner_cells=(
                    nested_fold.selection.comparison.selection.n_folds
                    * len(nested.candidate_n_states)
                    * len(nested.model_seeds)
                ),
                selected_n_states=nested_fold.evaluation.selected_n_states,
                cohort_score=nested_fold.evaluation.cohort_score,
            )
        )
    return _progress(
        root=root,
        status=status,
        method=nested.method,
        model_kind=nested.model_kind,
        subjects=nested.subjects,
        candidates=nested.candidate_n_states,
        model_seeds=nested.model_seeds,
        selection_policy=nested.selection_policy,
        inner_n_folds=nested.folds[0].selection.comparison.selection.n_folds,
        outer_split_seed=nested.outer_split_seed,
        inner_split_seed=nested.inner_split_seed,
        lock_status=lock_status,
        folds=tuple(fold_progress),
        temporary_paths=_temporary_paths(root),
        workflow_fingerprint=nested.workflow_fingerprint,
        cohort_score=nested.cohort_score,
    )


def inspect_nested_state_count_progress(
    store: FeatureStore,
    path: str | Path,
) -> NestedStateCountProgress:
    """Audit progress without trusting or modifying partial workflow files."""
    if not isinstance(store, FeatureStore):
        raise TypeError("store must be a FeatureStore")
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"nested progress directory does not exist: {root}")
    _require_symlink_free_tree(root)
    lock_status = inspect_nested_checkpoint_lock(root)
    checkpoint_path = root / "checkpoint.json"
    nested_path = root / "nested_evaluation.json"
    if checkpoint_path.is_file():
        payload = load_nested_state_count_checkpoint(checkpoint_path)
        plan = _checkpoint_plan(payload, store)
        if nested_path.is_file():
            nested = load_nested_state_count_cross_validation(root)
            progress = _completed_progress(
                root,
                store,
                nested,
                status="ready-for-promotion",
                lock_status=lock_status,
            )
            if (
                progress.method != plan.method
                or progress.model_kind != plan.model_kind
                or progress.subjects != plan.subjects
                or progress.candidate_n_states != plan.candidates
                or progress.model_seeds != plan.model_seeds
                or progress.selection_policy != plan.selection_policy
                or progress.n_outer_folds != len(plan.outer_folds)
                or progress.n_inner_folds != plan.inner_n_folds
                or progress.outer_split_seed != plan.outer_split_seed
                or progress.inner_split_seed != plan.inner_split_seed
                or dict(nested.fit_configuration) != plan.configuration
            ):
                raise ValueError("completed checkpoint disagrees with its checkpoint contract")
            return progress

        fold_progress: list[NestedStateCountFoldProgress] = []
        outer_root = root / "outer-folds"
        if outer_root.exists() and not outer_root.is_dir():
            raise ValueError("checkpoint outer-folds path must be a directory")
        for fold_index, (test_fingerprint, fold) in enumerate(plan.outer_folds):
            fold_root = outer_root / f"fold-{fold_index + 1:03d}"
            if fold_root.exists() and not fold_root.is_dir():
                raise ValueError("checkpoint outer-fold path must be a directory")
            selection_path = fold_root / "selection"
            evaluation_path = fold_root / "evaluation"
            selection = None
            evaluation = None
            partial_selection = None
            selection_workflow = selection_path / "workflow.json"
            selection_checkpoint = selection_path / STATE_COUNT_CHECKPOINT_MANIFEST
            if selection_workflow.is_file():
                selection = load_state_count_cross_validation(selection_path)
                if selection_checkpoint.exists():
                    selection_plan = make_state_count_checkpoint_plan(
                        store,
                        method=plan.method,
                        subjects=fold.fit_subjects,
                        candidate_n_states=plan.candidates,
                        model_seeds=plan.model_seeds,
                        n_folds=plan.inner_n_folds,
                        split_seed=plan.inner_split_seed,
                        configuration=plan.configuration,
                        minimum_sequence_length=plan.minimum_sequence_length,
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
                validate_nested_selection(
                    selection,
                    store,
                    fold,
                    method=plan.method,
                    candidates=plan.candidates,
                    model_seeds=plan.model_seeds,
                    inner_n_folds=plan.inner_n_folds,
                    inner_split_seed=plan.inner_split_seed,
                    configuration=plan.configuration,
                    minimum_sequence_length=plan.minimum_sequence_length,
                )
            elif selection_checkpoint.is_file():
                selection_plan = make_state_count_checkpoint_plan(
                    store,
                    method=plan.method,
                    subjects=fold.fit_subjects,
                    candidate_n_states=plan.candidates,
                    model_seeds=plan.model_seeds,
                    n_folds=plan.inner_n_folds,
                    split_seed=plan.inner_split_seed,
                    configuration=plan.configuration,
                    minimum_sequence_length=plan.minimum_sequence_length,
                )
                partial_selection = inspect_state_count_checkpoint(
                    store,
                    selection_plan,
                    selection_path,
                )
            elif selection_path.exists():
                raise ValueError(
                    "checkpoint selection has neither workflow.json nor its "
                    "cell checkpoint manifest"
                )
            if evaluation_path.exists():
                if selection is None:
                    raise ValueError("checkpoint evaluation exists without its selection")
                evaluation = load_selected_state_count_evaluation(evaluation_path)
                validate_nested_evaluation(
                    evaluation,
                    selection,
                    store,
                    fold,
                    selection_policy=plan.selection_policy,
                    test_fingerprint=test_fingerprint,
                    configuration=plan.configuration,
                )
            fold_progress.append(
                NestedStateCountFoldProgress(
                    fold_index=fold_index,
                    assignment_index=fold.fold_index,
                    status=(
                        "complete"
                        if evaluation is not None
                        else "pending-evaluation"
                        if selection is not None
                        else "selection-in-progress"
                        if partial_selection is not None
                        else "pending-selection"
                    ),
                    development_subjects=fold.fit_subjects,
                    test_subjects=fold.evaluation_subjects,
                    selection_complete=selection is not None,
                    evaluation_complete=evaluation is not None,
                    completed_inner_models=(
                        plan.inner_n_folds
                        * len(plan.candidates)
                        * len(plan.model_seeds)
                        if selection is not None
                        else 0
                        if partial_selection is None
                        else partial_selection.completed_models
                    ),
                    completed_inner_scores=(
                        plan.inner_n_folds
                        * len(plan.candidates)
                        * len(plan.model_seeds)
                        if selection is not None
                        else 0
                        if partial_selection is None
                        else partial_selection.completed_scores
                    ),
                    total_inner_cells=(
                        plan.inner_n_folds
                        * len(plan.candidates)
                        * len(plan.model_seeds)
                    ),
                    selected_n_states=(
                        None
                        if selection is None
                        else selection.selected_n_states(plan.selection_policy)
                    ),
                    cohort_score=None if evaluation is None else evaluation.cohort_score,
                )
            )
        return _progress(
            root=root,
            status="in-progress",
            method=plan.method,
            model_kind=plan.model_kind,
            subjects=plan.subjects,
            candidates=plan.candidates,
            model_seeds=plan.model_seeds,
            selection_policy=plan.selection_policy,
            inner_n_folds=plan.inner_n_folds,
            outer_split_seed=plan.outer_split_seed,
            inner_split_seed=plan.inner_split_seed,
            lock_status=lock_status,
            folds=tuple(fold_progress),
            temporary_paths=_temporary_paths(root),
            workflow_fingerprint=None,
            cohort_score=None,
        )
    if nested_path.is_file():
        nested = load_nested_state_count_cross_validation(root)
        return _completed_progress(
            root,
            store,
            nested,
            status="complete",
            lock_status=lock_status,
        )
    raise ValueError(
        "nested progress directory has neither checkpoint.json nor nested_evaluation.json"
    )


def nested_state_count_progress_payload(
    progress: NestedStateCountProgress,
) -> dict[str, object]:
    """Return a finite JSON payload for CLI and downstream progress reporting."""
    return {
        "workflow": str(progress.root),
        "status": progress.status,
        "method": progress.method,
        "model_kind": progress.model_kind,
        "subjects": list(progress.subjects),
        "n_subjects": len(progress.subjects),
        "candidate_n_states": list(progress.candidate_n_states),
        "model_seeds": list(progress.model_seeds),
        "selection_policy": progress.selection_policy,
        "n_outer_folds": progress.n_outer_folds,
        "n_inner_folds": progress.n_inner_folds,
        "outer_split_seed": progress.outer_split_seed,
        "inner_split_seed": progress.inner_split_seed,
        "execution_status": progress.execution_status,
        "lock_path": str(progress.lock_path),
        "lock_owner_pid": progress.lock_owner_pid,
        "lock_owner_hostname": progress.lock_owner_hostname,
        "lock_owner_process_start_token": progress.lock_owner_process_start_token,
        "lock_acquired_at_unix": progress.lock_acquired_at_unix,
        "lock_released_at_unix": progress.lock_released_at_unix,
        "completed_selections": progress.completed_selections,
        "completed_evaluations": progress.completed_evaluations,
        "completed_inner_fits": progress.completed_inner_fits,
        "total_inner_fits": progress.total_inner_fits,
        "completed_inner_scores": progress.completed_inner_scores,
        "total_inner_scores": progress.total_inner_scores,
        "completed_outer_models": progress.completed_outer_models,
        "total_outer_models": progress.total_outer_models,
        "completed_fit_steps": progress.completed_fit_steps,
        "total_fit_steps": progress.total_fit_steps,
        "fit_completion_fraction": progress.fit_completion_fraction,
        "temporary_paths": list(progress.temporary_paths),
        "folds": [
            {
                "fold_index": fold.fold_index,
                "assignment_index": fold.assignment_index,
                "status": fold.status,
                "development_subjects": list(fold.development_subjects),
                "test_subjects": list(fold.test_subjects),
                "selection_complete": fold.selection_complete,
                "evaluation_complete": fold.evaluation_complete,
                "completed_inner_models": fold.completed_inner_models,
                "completed_inner_scores": fold.completed_inner_scores,
                "total_inner_cells": fold.total_inner_cells,
                "selected_n_states": fold.selected_n_states,
                "cohort_score": fold.cohort_score,
            }
            for fold in progress.folds
        ],
        "workflow_fingerprint": progress.workflow_fingerprint,
        "cohort_score": progress.cohort_score,
    }


__all__ = [
    "NestedStateCountFoldProgress",
    "NestedStateCountProgress",
    "inspect_nested_state_count_progress",
    "nested_state_count_progress_payload",
]
