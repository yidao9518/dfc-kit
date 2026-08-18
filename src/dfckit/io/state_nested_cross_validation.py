"""Strict loading of complete nested state-count cross-validation workflows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np

from ..states.cross_validation import make_subject_validation_folds
from .state_cross_validation import (
    SPLIT_ALGORITHM,
    StateCountCrossValidationArtifact,
    load_state_count_cross_validation,
)
from .state_evaluation import (
    SelectedStateCountEvaluationArtifact,
    load_selected_state_count_evaluation,
)

FORMAT_NAME = "dfckit-nested-state-count-cross-validation"
FORMAT_VERSION = 1


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON field in nested cross-validation: {key}")
        output[key] = value
    return output


def _invalid_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant in nested cross-validation: {value}")


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"nested cross-validation manifest does not exist: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_invalid_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read nested cross-validation manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError("nested cross-validation manifest must be a JSON object")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    output = float(value)
    if not np.isfinite(output):
        raise ValueError(f"{name} must be finite")
    return output


def _fingerprint(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 value")
    return value


def _subjects(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    output = tuple(value)
    if (
        any(not isinstance(subject, str) or not subject.strip() for subject in output)
        or len(set(output)) != len(output)
    ):
        raise ValueError(f"{name} must contain unique non-empty subject labels")
    return output


def _integers(
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


def _relative_artifact(root: Path, value: object, name: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a non-empty POSIX relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{name} must stay inside the nested workflow directory")
    target = root.joinpath(*relative.parts)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} resolves outside the nested workflow directory") from error
    return relative.as_posix(), target


def _sample_interval_matches(left: float | None, right: float | None) -> bool:
    return (left is None and right is None) or (
        left is not None
        and right is not None
        and np.isclose(left, right, rtol=0.0, atol=1e-9)
    )


@dataclass(frozen=True)
class NestedStateCountFold:
    """One outer fold and its strictly loaded inner and outer evidence."""

    fold_index: int
    assignment_index: int
    selection_reference: str
    evaluation_reference: str
    selection: StateCountCrossValidationArtifact
    evaluation: SelectedStateCountEvaluationArtifact


@dataclass(frozen=True)
class NestedStateCountCrossValidationArtifact:
    """A complete participant-level nested state-count evaluation."""

    root: Path
    method: str
    model_kind: str
    source_contract: str
    sample_interval_seconds: float | None
    feature_contract_fingerprint: str
    subjects: tuple[str, ...]
    candidate_n_states: tuple[int, ...]
    model_seeds: tuple[int, ...]
    selection_policy: str
    outer_split_seed: int
    inner_split_seed: int
    fit_configuration: Mapping[str, object]
    folds: tuple[NestedStateCountFold, ...]
    selected_state_counts: tuple[int, ...]
    subject_scores: tuple[float, ...]
    cohort_score: float
    workflow_fingerprint: str


def load_nested_state_count_cross_validation(
    path: str | Path,
) -> NestedStateCountCrossValidationArtifact:
    """Load every outer fold and rebuild all nested-CV summary values."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"nested cross-validation directory does not exist: {root}")
    raw = _load_json(root / "nested_evaluation.json")
    expected_fields = {
        "candidate_n_states",
        "cohort_score",
        "feature_contract_fingerprint",
        "fit_configuration",
        "folds",
        "format",
        "format_version",
        "inner_validation",
        "method",
        "model_kind",
        "model_seeds",
        "n_inner_fits",
        "n_outer_models",
        "n_outer_folds",
        "n_subjects",
        "outer_split",
        "sample_interval_seconds",
        "score_weighting",
        "selected_state_counts",
        "selection_direction",
        "selection_metric",
        "selection_policy",
        "source_contract",
        "subject_scores",
        "subjects",
    }
    if set(raw) != expected_fields:
        raise ValueError("nested cross-validation fields do not match the schema")
    if raw["format"] != FORMAT_NAME or raw["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported nested cross-validation format or version")
    method = raw["method"]
    if method not in {"kmeans", "hmm"}:
        raise ValueError("nested cross-validation method is invalid")
    model_kind = "kmeans-state" if method == "kmeans" else "gaussian-hmm-state"
    if raw["model_kind"] != model_kind:
        raise ValueError("nested cross-validation method and model kind disagree")
    subjects = _subjects(raw["subjects"], "subjects")
    if _integer(raw["n_subjects"], "n_subjects", minimum=1) != len(subjects):
        raise ValueError("nested cross-validation subject count is inconsistent")
    candidates = _integers(
        raw["candidate_n_states"],
        "candidate_n_states",
        minimum=2,
        minimum_count=2,
    )
    model_seeds = _integers(
        raw["model_seeds"],
        "model_seeds",
        minimum=0,
        minimum_count=1,
    )
    selection_policy = raw["selection_policy"]
    if selection_policy not in {"one-standard-error", "best"}:
        raise ValueError("nested selection_policy is invalid")
    source_contract = raw["source_contract"]
    if not isinstance(source_contract, str) or not source_contract.strip():
        raise ValueError("nested source_contract must be non-empty")
    sample_interval = raw["sample_interval_seconds"]
    if sample_interval is not None:
        sample_interval = _finite_float(sample_interval, "sample_interval_seconds")
        if sample_interval <= 0.0:
            raise ValueError("sample_interval_seconds must be positive")
    feature_fingerprint = _fingerprint(
        raw["feature_contract_fingerprint"],
        "feature_contract_fingerprint",
    )
    configuration = raw["fit_configuration"]
    if not isinstance(configuration, dict) or not configuration:
        raise ValueError("nested fit_configuration must be a non-empty object")
    try:
        json.dumps(configuration, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("nested fit_configuration must contain finite JSON values") from error

    outer_split = raw["outer_split"]
    if not isinstance(outer_split, dict) or set(outer_split) != {
        "algorithm",
        "n_folds",
        "seed",
    }:
        raise ValueError("nested outer_split fields do not match the schema")
    if outer_split["algorithm"] != SPLIT_ALGORITHM:
        raise ValueError("nested outer split algorithm is unsupported")
    n_outer_folds = _integer(outer_split["n_folds"], "outer n_folds", minimum=2)
    if _integer(raw["n_outer_folds"], "n_outer_folds", minimum=2) != n_outer_folds:
        raise ValueError("nested outer fold count is inconsistent")
    outer_seed = _integer(outer_split["seed"], "outer split seed")
    expected_assignments = make_subject_validation_folds(
        subjects,
        n_folds=n_outer_folds,
        seed=outer_seed,
    )
    assignments = {fold.fold_index: fold for fold in expected_assignments}

    inner = raw["inner_validation"]
    if not isinstance(inner, dict) or set(inner) != {"n_folds", "split_seed"}:
        raise ValueError("nested inner_validation fields do not match the schema")
    inner_n_folds = _integer(inner["n_folds"], "inner n_folds", minimum=2)
    inner_seed = _integer(inner["split_seed"], "inner split seed")
    expected_inner_fits = n_outer_folds * inner_n_folds * len(candidates) * len(
        model_seeds
    )
    expected_outer_models = n_outer_folds * len(model_seeds)
    if (
        _integer(raw["n_inner_fits"], "n_inner_fits", minimum=1)
        != expected_inner_fits
        or _integer(raw["n_outer_models"], "n_outer_models", minimum=1)
        != expected_outer_models
    ):
        raise ValueError("nested model counts do not match the complete grid")

    records = raw["folds"]
    if not isinstance(records, list) or len(records) != n_outer_folds:
        raise ValueError("nested folds do not match the outer fold count")
    loaded_folds: list[NestedStateCountFold] = []
    selected_counts: list[int] = []
    observed_assignments: set[int] = set()
    observed_paths: set[str] = set()
    score_by_subject: dict[str, float] = {}
    for fold_index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "assignment_index",
            "cohort_score",
            "development_data_fingerprint",
            "development_subjects",
            "evaluation_artifact",
            "evaluation_fingerprint",
            "fold_index",
            "selected_n_states",
            "selection_artifact",
            "selection_workflow_fingerprint",
            "subject_scores",
            "test_data_fingerprint",
            "test_subjects",
        }:
            raise ValueError("nested fold fields do not match the schema")
        if _integer(record["fold_index"], "fold_index") != fold_index:
            raise ValueError("nested fold indices are misordered")
        assignment_index = _integer(record["assignment_index"], "assignment_index")
        if assignment_index in observed_assignments or assignment_index not in assignments:
            raise ValueError("nested assignment indices are invalid")
        observed_assignments.add(assignment_index)
        assignment = assignments[assignment_index]
        development_subjects = _subjects(
            record["development_subjects"],
            "development_subjects",
        )
        test_subjects = _subjects(record["test_subjects"], "test_subjects")
        if (
            development_subjects != assignment.fit_subjects
            or test_subjects != assignment.evaluation_subjects
        ):
            raise ValueError("nested fold plan disagrees with the outer split")
        selection_reference, selection_path = _relative_artifact(
            root,
            record["selection_artifact"],
            "selection_artifact",
        )
        evaluation_reference, evaluation_path = _relative_artifact(
            root,
            record["evaluation_artifact"],
            "evaluation_artifact",
        )
        if (
            selection_reference in observed_paths
            or evaluation_reference in observed_paths
            or selection_reference != f"outer-folds/fold-{fold_index + 1:03d}/selection"
            or evaluation_reference != f"outer-folds/fold-{fold_index + 1:03d}/evaluation"
        ):
            raise ValueError("nested child artifact paths are duplicated or misaligned")
        observed_paths.update((selection_reference, evaluation_reference))
        selection = load_state_count_cross_validation(selection_path)
        evaluation = load_selected_state_count_evaluation(evaluation_path)
        selected_n_states = selection.selected_n_states(selection_policy)
        record_subject_scores = record["subject_scores"]
        if not isinstance(record_subject_scores, list):
            raise TypeError("nested fold subject_scores must be an array")
        rebuilt_subject_scores = [
            {"subject": subject, "score": score}
            for subject, score in zip(
                evaluation.test_subjects,
                evaluation.subject_scores,
                strict=True,
            )
        ]
        if (
            selection.method != method
            or evaluation.method != method
            or selection.development_subjects != development_subjects
            or selection.candidate_n_states != candidates
            or selection.model_seeds != model_seeds
            or selection.split_seed != inner_seed
            or selection.comparison.selection.n_folds != inner_n_folds
            or dict(selection.fit_configuration) != configuration
            or evaluation.development_subjects != development_subjects
            or evaluation.test_subjects != test_subjects
            or evaluation.selection_policy != selection_policy
            or evaluation.selection_workflow_fingerprint
            != selection.workflow_fingerprint
            or evaluation.selected_n_states != selected_n_states
            or evaluation.model_seeds != model_seeds
            or dict(evaluation.fit_configuration) != configuration
            or selection.source_contract != source_contract
            or evaluation.source_contract != source_contract
            or selection.feature_contract_fingerprint != feature_fingerprint
            or evaluation.feature_contract_fingerprint != feature_fingerprint
            or not _sample_interval_matches(
                selection.sample_interval_seconds,
                sample_interval,
            )
            or not _sample_interval_matches(
                evaluation.sample_interval_seconds,
                sample_interval,
            )
            or selection.development_data_fingerprint
            != record["development_data_fingerprint"]
            or evaluation.development_data_fingerprint
            != record["development_data_fingerprint"]
            or evaluation.test_data_fingerprint != record["test_data_fingerprint"]
            or selection.workflow_fingerprint
            != record["selection_workflow_fingerprint"]
            or evaluation.artifact_fingerprint != record["evaluation_fingerprint"]
            or selected_n_states
            != _integer(record["selected_n_states"], "selected_n_states", minimum=2)
            or not np.isclose(
                evaluation.cohort_score,
                _finite_float(record["cohort_score"], "fold cohort_score"),
                rtol=1e-12,
                atol=1e-12,
            )
            or rebuilt_subject_scores != record_subject_scores
        ):
            raise ValueError("nested fold summary disagrees with its child evidence")
        for subject, score in zip(test_subjects, evaluation.subject_scores, strict=True):
            if subject in score_by_subject:
                raise ValueError("nested outer folds score a subject more than once")
            score_by_subject[subject] = score
        selected_counts.append(selected_n_states)
        loaded_folds.append(
            NestedStateCountFold(
                fold_index=fold_index,
                assignment_index=assignment_index,
                selection_reference=selection_reference,
                evaluation_reference=evaluation_reference,
                selection=selection,
                evaluation=evaluation,
            )
        )

    if set(score_by_subject) != set(subjects):
        raise ValueError("nested outer folds must score every subject exactly once")
    subject_scores = tuple(score_by_subject[subject] for subject in subjects)
    cohort_score = float(np.mean(subject_scores))
    rebuilt_subjects = [
        {"subject": subject, "score": score}
        for subject, score in zip(subjects, subject_scores, strict=True)
    ]
    if (
        raw["selected_state_counts"] != selected_counts
        or raw["subject_scores"] != rebuilt_subjects
        or not np.isclose(
            cohort_score,
            _finite_float(raw["cohort_score"], "cohort_score"),
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise ValueError("nested cross-validation summary cannot be rebuilt from outer folds")
    expected_metric = (
        "subject-balanced mean squared distance"
        if method == "kmeans"
        else "subject-balanced log likelihood per sample"
    )
    expected_direction = "lower is better" if method == "kmeans" else "higher is better"
    if (
        raw["selection_metric"] != expected_metric
        or raw["selection_direction"] != expected_direction
        or raw["score_weighting"]
        != "runs sample-weighted within subject and seed; seeds equal within subject; subjects equal across outer folds"
    ):
        raise ValueError("nested cross-validation score metadata is inconsistent")

    canonical = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return NestedStateCountCrossValidationArtifact(
        root=root,
        method=method,
        model_kind=model_kind,
        source_contract=source_contract,
        sample_interval_seconds=sample_interval,
        feature_contract_fingerprint=feature_fingerprint,
        subjects=subjects,
        candidate_n_states=candidates,
        model_seeds=model_seeds,
        selection_policy=selection_policy,
        outer_split_seed=outer_seed,
        inner_split_seed=inner_seed,
        fit_configuration=MappingProxyType(dict(configuration)),
        folds=tuple(loaded_folds),
        selected_state_counts=tuple(selected_counts),
        subject_scores=subject_scores,
        cohort_score=cohort_score,
        workflow_fingerprint=hashlib.sha256(canonical).hexdigest(),
    )


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "NestedStateCountCrossValidationArtifact",
    "NestedStateCountFold",
    "load_nested_state_count_cross_validation",
]
