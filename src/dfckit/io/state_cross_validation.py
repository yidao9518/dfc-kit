"""Validation and loading of automated state-count cross-validation workflows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise, product
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np

from ..states.cross_validation import make_subject_validation_folds
from ..states.hmm import GaussianHMMStateModel
from ..states.kmeans import KMeansStateModel
from .models import fitted_model_fingerprint, load_fitted_model
from .state_scoring import load_state_model_scores, state_model_specification
from .state_selection import (
    StateCountComparison,
    compare_state_model_scores,
    state_count_comparison_payload,
)

FORMAT_NAME = "dfckit-state-count-cross-validation"
FORMAT_VERSION = 1
SPLIT_ALGORITHM = "sha256-seed-subject-balanced-v1"


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON field in cross-validation artifact: {key}")
        output[key] = value
    return output


def _invalid_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant in cross-validation artifact: {value}")


def _load_json(path: Path, name: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_invalid_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


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


def _fingerprint(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 value")
    return value


def _string_array(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    output = tuple(value)
    if (
        any(not isinstance(item, str) or not item.strip() for item in output)
        or len(set(output)) != len(output)
    ):
        raise ValueError(f"{name} must contain unique non-empty strings")
    return output


def _integer_array(
    value: object,
    name: str,
    *,
    minimum: int,
    minimum_count: int,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) < minimum_count:
        raise ValueError(f"{name} must contain at least {minimum_count} values")
    output = tuple(_integer(item, name, minimum=minimum) for item in value)
    if len(set(output)) != len(output) or any(
        right <= left for left, right in pairwise(output)
    ):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return output


def _relative_artifact(root: Path, value: object, name: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a non-empty POSIX relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{name} must stay inside the workflow directory")
    target = root.joinpath(*relative.parts)
    resolved_root = root.resolve()
    try:
        target.resolve().relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{name} resolves outside the workflow directory") from error
    return relative.as_posix(), target


def _sample_interval_matches(left: object, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        observed = _finite_float(left, "sample_interval_seconds", positive=True)
    except (TypeError, ValueError):
        return False
    return bool(np.isclose(observed, right, rtol=0.0, atol=1e-9))


def _fit_configuration(raw: object, method: str) -> Mapping[str, object]:
    if not isinstance(raw, dict):
        raise TypeError("fit_configuration must be a JSON object")
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
    if set(raw) != expected:
        raise ValueError("fit_configuration fields do not match the model method")
    output = dict(raw)
    if method == "kmeans":
        for name in ("batch_size", "max_iter", "n_init"):
            output[name] = _integer(output[name], f"fit_configuration {name}", minimum=1)
        requested = output["init_sample_size"]
        if requested is not None:
            output["init_sample_size"] = _integer(
                requested,
                "fit_configuration init_sample_size",
                minimum=1,
            )
        if not isinstance(output["standardize_features"], bool):
            raise TypeError("fit_configuration standardize_features must be boolean")
        ratio = _finite_float(
            output["reassignment_ratio"],
            "fit_configuration reassignment_ratio",
        )
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("fit_configuration reassignment_ratio must be within [0, 1]")
        output["reassignment_ratio"] = ratio
    else:
        for name in (
            "minimum_sequence_length",
            "n_init",
            "n_iter",
            "n_pca_components",
            "pca_batch_size",
        ):
            output[name] = _integer(output[name], f"fit_configuration {name}", minimum=1)
        if output["minimum_sequence_length"] < 2:
            raise ValueError("HMM minimum_sequence_length must be at least two")
        if output["covariance_type"] not in {"diag", "full"}:
            raise ValueError("fit_configuration covariance_type is invalid")
        output["tol"] = _finite_float(
            output["tol"],
            "fit_configuration tol",
            positive=True,
        )
    return MappingProxyType(output)


def validate_state_count_model_configuration(
    model: KMeansStateModel | GaussianHMMStateModel,
    configuration: Mapping[str, object],
) -> None:
    """Validate one fitted model against a state-count workflow configuration."""
    if isinstance(model, KMeansStateModel):
        expected_init_sample_size = configuration["init_sample_size"]
        if expected_init_sample_size is None:
            expected_init_sample_size = max(1000, 20 * model.n_states)
        expected_init_sample_size = min(
            int(expected_init_sample_size),
            model.fit_sample_count,
        )
        observed = {
            "batch_size": model.batch_size,
            "init_sample_size": model.init_sample_size,
            "max_iter": model.max_iter,
            "n_init": model.n_init,
            "reassignment_ratio": model.reassignment_ratio,
            "standardize_features": model.standardize_features,
        }
        expected = {**dict(configuration), "init_sample_size": expected_init_sample_size}
    else:
        observed = {
            "covariance_type": model.covariance_type,
            "minimum_sequence_length": model.minimum_sequence_length,
            "n_init": model.n_init,
            "n_iter": model.n_iter,
            "n_pca_components": model.n_pca_components,
            "pca_batch_size": model.pca_batch_size,
            "tol": model.tol,
        }
        expected = dict(configuration)
    if observed != expected:
        raise ValueError("fitted model configuration disagrees with workflow.json")


@dataclass(frozen=True)
class StateCountCrossValidationArtifact:
    """Canonically validated state-count cross-validation workflow."""

    root: Path
    method: str
    model_kind: str
    source_contract: str
    sample_interval_seconds: float | None
    feature_contract_fingerprint: str
    development_data_fingerprint: str
    development_subjects: tuple[str, ...]
    candidate_n_states: tuple[int, ...]
    model_seeds: tuple[int, ...]
    split_seed: int
    fit_configuration: Mapping[str, object]
    comparison: StateCountComparison
    workflow_fingerprint: str

    def selected_n_states(self, policy: str = "one-standard-error") -> int:
        if policy == "one-standard-error":
            return self.comparison.selection.one_standard_error_n_states
        if policy == "best":
            return self.comparison.selection.best_n_states
        raise ValueError("selection policy must be 'one-standard-error' or 'best'")


def load_state_count_cross_validation(
    path: str | Path,
) -> StateCountCrossValidationArtifact:
    """Load and reconstruct all evidence in an automated selection workflow."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"cross-validation workflow directory does not exist: {root}")
    raw = _load_json(root / "workflow.json", "cross-validation workflow")
    expected_fields = {
        "best_n_states",
        "candidate_n_states",
        "comparison_artifact",
        "development_data_fingerprint",
        "development_subjects",
        "feature_contract_fingerprint",
        "fit_configuration",
        "fits",
        "format",
        "format_version",
        "method",
        "model_kind",
        "model_seeds",
        "n_fits",
        "one_standard_error_n_states",
        "sample_interval_seconds",
        "source_contract",
        "split",
    }
    if set(raw) != expected_fields:
        raise ValueError("cross-validation workflow fields do not match the schema")
    if raw["format"] != FORMAT_NAME or raw["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported cross-validation workflow format or version")
    method = raw["method"]
    if method not in {"kmeans", "hmm"}:
        raise ValueError("cross-validation workflow method is invalid")
    model_kind = "kmeans-state" if method == "kmeans" else "gaussian-hmm-state"
    if raw["model_kind"] != model_kind:
        raise ValueError("cross-validation workflow method and model kind disagree")
    source_contract = raw["source_contract"]
    if not isinstance(source_contract, str) or not source_contract.strip():
        raise ValueError("cross-validation source_contract must be non-empty")
    sample_interval = raw["sample_interval_seconds"]
    if sample_interval is not None:
        sample_interval = _finite_float(
            sample_interval,
            "sample_interval_seconds",
            positive=True,
        )
    feature_fingerprint = _fingerprint(
        raw["feature_contract_fingerprint"],
        "feature_contract_fingerprint",
    )
    development_fingerprint = _fingerprint(
        raw["development_data_fingerprint"],
        "development_data_fingerprint",
    )
    development_subjects = _string_array(
        raw["development_subjects"],
        "development_subjects",
    )
    candidates = _integer_array(
        raw["candidate_n_states"],
        "candidate_n_states",
        minimum=2,
        minimum_count=2,
    )
    model_seeds = _integer_array(
        raw["model_seeds"],
        "model_seeds",
        minimum=0,
        minimum_count=1,
    )
    configuration = _fit_configuration(raw["fit_configuration"], method)

    split = raw["split"]
    if not isinstance(split, dict) or set(split) != {
        "algorithm",
        "folds",
        "n_folds",
        "seed",
    }:
        raise ValueError("cross-validation split fields do not match the schema")
    if split["algorithm"] != SPLIT_ALGORITHM:
        raise ValueError("cross-validation split algorithm is unsupported")
    n_folds = _integer(split["n_folds"], "split n_folds", minimum=2)
    split_seed = _integer(split["seed"], "split seed", minimum=0)
    expected_assignment_folds = make_subject_validation_folds(
        development_subjects,
        n_folds=n_folds,
        seed=split_seed,
    )
    assignment_by_index = {
        fold.fold_index: fold for fold in expected_assignment_folds
    }

    comparison_reference, comparison_path = _relative_artifact(
        root,
        raw["comparison_artifact"],
        "comparison_artifact",
    )
    comparison_raw = _load_json(comparison_path, "state-count comparison")
    fits = raw["fits"]
    if not isinstance(fits, list):
        raise TypeError("cross-validation fits must be an array")
    expected_fit_count = n_folds * len(candidates) * len(model_seeds)
    if _integer(raw["n_fits"], "n_fits", minimum=1) != expected_fit_count or len(
        fits
    ) != expected_fit_count:
        raise ValueError("cross-validation fit count does not match the complete grid")

    expected_keys = tuple(product(range(n_folds), candidates, model_seeds))
    reports = []
    score_references: list[str] = []
    observed_model_paths: set[str] = set()
    observed_score_paths: set[str] = set()
    for expected_key, record in zip(expected_keys, fits, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "fold_index",
            "model_artifact",
            "model_fingerprint",
            "model_seed",
            "n_states",
            "score_artifact",
        }:
            raise ValueError("cross-validation fit fields do not match the schema")
        key = (
            _integer(record["fold_index"], "fit fold_index"),
            _integer(record["n_states"], "fit n_states", minimum=2),
            _integer(record["model_seed"], "fit model_seed"),
        )
        if key != expected_key:
            raise ValueError("cross-validation fit grid is incomplete or misordered")
        model_fingerprint = _fingerprint(
            record["model_fingerprint"],
            "fit model_fingerprint",
        )
        model_reference, model_path = _relative_artifact(
            root,
            record["model_artifact"],
            "model_artifact",
        )
        score_reference, score_path = _relative_artifact(
            root,
            record["score_artifact"],
            "score_artifact",
        )
        stem = f"fold-{key[0] + 1:03d}_k-{key[1]}_seed-{key[2]}"
        if (
            model_reference != f"models/{stem}.model"
            or score_reference != f"scores/{stem}.json"
        ):
            raise ValueError("cross-validation artifact path is misaligned with its fit")
        if model_reference in observed_model_paths or score_reference in observed_score_paths:
            raise ValueError("cross-validation artifact paths must be unique")
        observed_model_paths.add(model_reference)
        observed_score_paths.add(score_reference)
        model = load_fitted_model(model_path)
        if not isinstance(model, (KMeansStateModel, GaussianHMMStateModel)):
            raise TypeError("cross-validation model artifact has the wrong type")
        report = load_state_model_scores(score_path)
        if (
            fitted_model_fingerprint(model) != model_fingerprint
            or report.model_fingerprint != model_fingerprint
            or model.n_states != key[1]
            or report.n_states != key[1]
            or model.seed != key[2]
            or report.model_seed != key[2]
            or model.fit_subjects != report.fit_subjects
            or model.training_data_fingerprint != report.training_data_fingerprint
            or dict(report.model_specification) != state_model_specification(model)
        ):
            raise ValueError("cross-validation model, score, and manifest identities disagree")
        if (method == "kmeans") != isinstance(model, KMeansStateModel):
            raise TypeError("cross-validation model family disagrees with workflow")
        validate_state_count_model_configuration(model, configuration)
        reports.append(report)
        score_references.append(score_reference)

    comparison = compare_state_model_scores(
        reports,
        score_artifacts=score_references,
    )
    if comparison_raw != state_count_comparison_payload(comparison):
        raise ValueError("cross-validation comparison cannot be rebuilt from its scores")
    if (
        comparison_reference != "comparison.json"
        or comparison.model_kind != model_kind
        or comparison.source_contract != source_contract
        or not _sample_interval_matches(sample_interval, comparison.sample_interval_seconds)
        or comparison.feature_contract_fingerprint != feature_fingerprint
        or tuple(comparison.selection.candidate_n_states) != candidates
        or comparison.model_seeds != model_seeds
        or set(comparison.subjects) != set(development_subjects)
        or comparison.selection.best_n_states
        != _integer(raw["best_n_states"], "best_n_states", minimum=2)
        or comparison.selection.one_standard_error_n_states
        != _integer(
            raw["one_standard_error_n_states"],
            "one_standard_error_n_states",
            minimum=2,
        )
    ):
        raise ValueError("cross-validation workflow and comparison metadata disagree")

    fold_records = split["folds"]
    if not isinstance(fold_records, list) or len(fold_records) != n_folds:
        raise ValueError("cross-validation split folds do not match n_folds")
    observed_assignments: set[int] = set()
    for fold_index, (record, comparison_fold) in enumerate(
        zip(fold_records, comparison.folds, strict=True)
    ):
        if not isinstance(record, dict) or set(record) != {
            "assignment_index",
            "evaluation_data_fingerprint",
            "evaluation_subjects",
            "fit_subjects",
            "fold_index",
        }:
            raise ValueError("cross-validation fold fields do not match the schema")
        if _integer(record["fold_index"], "fold_index") != fold_index:
            raise ValueError("cross-validation comparison fold indices are misordered")
        assignment_index = _integer(record["assignment_index"], "assignment_index")
        if assignment_index in observed_assignments or assignment_index not in assignment_by_index:
            raise ValueError("cross-validation assignment indices are invalid")
        observed_assignments.add(assignment_index)
        assignment = assignment_by_index[assignment_index]
        fit_subjects = _string_array(record["fit_subjects"], "fold fit_subjects")
        evaluation_subjects = _string_array(
            record["evaluation_subjects"],
            "fold evaluation_subjects",
        )
        evaluation_fingerprint = _fingerprint(
            record["evaluation_data_fingerprint"],
            "fold evaluation_data_fingerprint",
        )
        if (
            fit_subjects != assignment.fit_subjects
            or evaluation_subjects != assignment.evaluation_subjects
            or fit_subjects != comparison_fold.fit_subjects
            or evaluation_subjects != comparison_fold.evaluation_subjects
            or evaluation_fingerprint != comparison_fold.evaluation_data_fingerprint
        ):
            raise ValueError("cross-validation fold plan disagrees with score evidence")

    canonical = json.dumps(
        {"workflow": raw, "comparison": comparison_raw},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    workflow_fingerprint = hashlib.sha256(canonical).hexdigest()
    return StateCountCrossValidationArtifact(
        root=root,
        method=method,
        model_kind=model_kind,
        source_contract=source_contract,
        sample_interval_seconds=sample_interval,
        feature_contract_fingerprint=feature_fingerprint,
        development_data_fingerprint=development_fingerprint,
        development_subjects=development_subjects,
        candidate_n_states=candidates,
        model_seeds=model_seeds,
        split_seed=split_seed,
        fit_configuration=configuration,
        comparison=comparison,
        workflow_fingerprint=workflow_fingerprint,
    )


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "SPLIT_ALGORITHM",
    "StateCountCrossValidationArtifact",
    "load_state_count_cross_validation",
    "validate_state_count_model_configuration",
]
