"""Strict persistence for selected state-count outer-test evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np

from ..states.hmm import GaussianHMMStateModel
from ..states.kmeans import KMeansStateModel
from ..states.scoring import RunKMeansScore
from .models import fitted_model_fingerprint, load_fitted_model
from .state_scoring import (
    StateModelScoreReport,
    load_state_model_scores,
    state_model_specification,
)

FORMAT_NAME = "dfckit-selected-state-count-evaluation"
FORMAT_VERSION = 1


def _fingerprint(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 value")
    return value


def _subjects(value: Sequence[str], name: str) -> tuple[str, ...]:
    output = tuple(value)
    if (
        not output
        or any(not isinstance(subject, str) or not subject.strip() for subject in output)
        or len(set(output)) != len(output)
    ):
        raise ValueError(f"{name} must contain unique non-empty subject labels")
    return output


def _relative_path(value: str | Path, name: str) -> str:
    raw = str(value)
    if not raw or "\\" in raw:
        raise ValueError(f"{name} must be a non-empty POSIX relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must stay inside the evaluation directory")
    return path.as_posix()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON field in state-count evaluation: {key}")
        output[key] = value
    return output


def _invalid_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant in state-count evaluation: {value}")


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"state-count evaluation does not exist: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_invalid_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read state-count evaluation {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError("state-count evaluation must be a JSON object")
    return value


def _relative_artifact(root: Path, value: object, name: str) -> tuple[str, Path]:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a POSIX relative path")
    reference = _relative_path(value, name)
    target = root.joinpath(*PurePosixPath(reference).parts)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} resolves outside the evaluation directory") from error
    return reference, target


def _sample_interval_matches(left: float | None, right: float | None) -> bool:
    return (left is None and right is None) or (
        left is not None
        and right is not None
        and np.isclose(left, right, rtol=0.0, atol=1e-9)
    )


def _run_boundaries(report: StateModelScoreReport) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            score.subject,
            score.session,
            score.acquisition_id,
            score.n_samples,
            score.n_sequences,
        )
        for score in report.scores
    )


def _subject_scores(report: StateModelScoreReport) -> tuple[float, ...]:
    totals: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    for score in report.scores:
        total = (
            score.total_squared_distance
            if isinstance(score, RunKMeansScore)
            else score.log_likelihood
        )
        totals[score.subject] = totals.get(score.subject, 0.0) + total
        sample_counts[score.subject] = sample_counts.get(score.subject, 0) + score.n_samples
    return tuple(
        totals[subject] / sample_counts[subject] for subject in report.subjects
    )


def _configuration_matches_report(
    configuration: Mapping[str, object],
    report: StateModelScoreReport,
) -> bool:
    specification = dict(report.model_specification or {})
    if report.model_kind == "kmeans-state":
        shared = {
            "batch_size",
            "max_iter",
            "n_init",
            "reassignment_ratio",
            "standardize_features",
        }
        if any(specification.get(name) != configuration.get(name) for name in shared):
            return False
        requested = configuration.get("init_sample_size")
        realized = specification.get("init_sample_size")
        return (
            isinstance(realized, int)
            and realized > 0
            and (requested is None or realized <= requested)
        )
    shared = {
        "covariance_type",
        "minimum_sequence_length",
        "n_init",
        "n_iter",
        "n_pca_components",
        "pca_batch_size",
        "tol",
    }
    return all(specification.get(name) == configuration.get(name) for name in shared)


def selected_state_count_evaluation_payload(
    reports: Sequence[StateModelScoreReport],
    *,
    model_artifacts: Sequence[str | Path],
    score_artifacts: Sequence[str | Path],
    method: str,
    selection_policy: str,
    selection_workflow_fingerprint: str,
    selected_n_states: int,
    development_data_fingerprint: str,
    development_subjects: Sequence[str],
    test_data_fingerprint: str,
    test_subjects: Sequence[str],
    fit_configuration: Mapping[str, object],
) -> dict[str, object]:
    """Build a finite, identity-checked outer-test evaluation payload."""
    reports = tuple(reports)
    models = tuple(_relative_path(path, "model_artifact") for path in model_artifacts)
    scores = tuple(_relative_path(path, "score_artifact") for path in score_artifacts)
    if not reports or len(reports) != len(models) or len(reports) != len(scores):
        raise ValueError("evaluation reports and artifact arrays must be equally non-empty")
    if len(set(models)) != len(models) or len(set(scores)) != len(scores):
        raise ValueError("evaluation artifact paths must be unique")
    if method not in {"kmeans", "hmm"}:
        raise ValueError("evaluation method must be 'kmeans' or 'hmm'")
    expected_kind = "kmeans-state" if method == "kmeans" else "gaussian-hmm-state"
    if selection_policy not in {"one-standard-error", "best"}:
        raise ValueError("selection_policy must be 'one-standard-error' or 'best'")
    if (
        isinstance(selected_n_states, (bool, np.bool_))
        or not isinstance(selected_n_states, (int, np.integer))
        or selected_n_states < 2
    ):
        raise ValueError("selected_n_states must be an integer of at least two")
    selected_n_states = int(selected_n_states)
    development_subjects = _subjects(development_subjects, "development_subjects")
    test_subjects = _subjects(test_subjects, "test_subjects")
    if set(development_subjects).intersection(test_subjects):
        raise ValueError("development and outer-test subjects must be disjoint")
    workflow_fingerprint = _fingerprint(
        selection_workflow_fingerprint,
        "selection_workflow_fingerprint",
    )
    development_fingerprint = _fingerprint(
        development_data_fingerprint,
        "development_data_fingerprint",
    )
    test_fingerprint = _fingerprint(test_data_fingerprint, "test_data_fingerprint")
    configuration = dict(fit_configuration)
    if not configuration:
        raise ValueError("fit_configuration must be non-empty")
    try:
        json.dumps(configuration, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("fit_configuration must contain finite JSON values") from error

    reference = reports[0]
    if reference.model_kind != expected_kind:
        raise ValueError("evaluation method and score model kind disagree")
    model_seeds = tuple(report.model_seed for report in reports)
    if (
        len(set(model_seeds)) != len(model_seeds)
        or tuple(sorted(model_seeds)) != model_seeds
    ):
        raise ValueError("evaluation model seeds must be unique and increasing")
    model_fingerprints = tuple(report.model_fingerprint for report in reports)
    if len(set(model_fingerprints)) != len(model_fingerprints):
        raise ValueError("evaluation fitted-model fingerprints must be distinct")
    boundaries = _run_boundaries(reference)
    for report in reports:
        if report.format_version < 2 or report.model_specification is None:
            raise ValueError("evaluation requires v2 score artifacts")
        if not _configuration_matches_report(configuration, report):
            raise ValueError(
                "evaluation fit_configuration disagrees with a fitted-model specification"
            )
        if (
            report.model_kind != expected_kind
            or report.n_states != selected_n_states
            or report.fit_subjects != development_subjects
            or report.training_data_fingerprint != development_fingerprint
            or report.evaluation_data_fingerprint != test_fingerprint
            or report.feature_contract_fingerprint
            != reference.feature_contract_fingerprint
            or report.source_contract != reference.source_contract
            or not _sample_interval_matches(
                report.sample_interval_seconds,
                reference.sample_interval_seconds,
            )
            or report.minimum_sequence_length != reference.minimum_sequence_length
            or report.omitted_short_sequence_count
            != reference.omitted_short_sequence_count
            or report.allow_fit_subjects
            or report.subjects != test_subjects
            or _run_boundaries(report) != boundaries
        ):
            raise ValueError("evaluation score artifacts do not share one outer-test contract")

    seed_subject_scores = tuple(_subject_scores(report) for report in reports)
    if not all(np.isfinite(values).all() for values in seed_subject_scores):
        raise ValueError("evaluation subject scores must be finite")
    subject_scores = tuple(
        float(value) for value in np.mean(seed_subject_scores, axis=0)
    )
    seed_scores = tuple(float(np.mean(values)) for values in seed_subject_scores)
    cohort_score = float(np.mean(subject_scores))
    if not np.isfinite(cohort_score):
        raise ValueError("evaluation cohort score must be finite")

    seed_payloads = []
    for index, report in enumerate(reports):
        expected_stem = f"k-{selected_n_states}_seed-{report.model_seed}"
        if (
            models[index] != f"models/{expected_stem}.model"
            or scores[index] != f"scores/{expected_stem}.json"
        ):
            raise ValueError("evaluation artifact path is misaligned with its model seed")
        seed_payloads.append(
            {
                "model_seed": report.model_seed,
                "model_fingerprint": report.model_fingerprint,
                "model_artifact": models[index],
                "score_artifact": scores[index],
                "cohort_score": seed_scores[index],
                "subject_scores": [
                    {"subject": subject, "score": score}
                    for subject, score in zip(
                        test_subjects,
                        seed_subject_scores[index],
                        strict=True,
                    )
                ],
            }
        )

    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model_kind": expected_kind,
        "method": method,
        "selection_policy": selection_policy,
        "selected_n_states": selected_n_states,
        "selection_workflow_fingerprint": workflow_fingerprint,
        "source_contract": reference.source_contract,
        "sample_interval_seconds": reference.sample_interval_seconds,
        "minimum_sequence_length": reference.minimum_sequence_length,
        "feature_contract_fingerprint": reference.feature_contract_fingerprint,
        "development_data_fingerprint": development_fingerprint,
        "development_subjects": list(development_subjects),
        "test_data_fingerprint": test_fingerprint,
        "test_subjects": list(test_subjects),
        "fit_configuration": configuration,
        "selection_metric": (
            "subject-balanced mean squared distance"
            if method == "kmeans"
            else "subject-balanced log likelihood per sample"
        ),
        "selection_direction": (
            "lower is better" if method == "kmeans" else "higher is better"
        ),
        "score_weighting": (
            "runs sample-weighted within subject and seed; seeds equal within "
            "subject; subjects equal across the outer test cohort"
        ),
        "n_models": len(reports),
        "n_seeds": len(reports),
        "model_seeds": list(model_seeds),
        "n_subjects": len(test_subjects),
        "n_runs": reference.n_runs,
        "n_sequences": reference.n_sequences,
        "n_samples": reference.n_samples,
        "omitted_short_sequence_count": reference.omitted_short_sequence_count,
        "cohort_score": cohort_score,
        "subject_scores": [
            {"subject": subject, "score": score}
            for subject, score in zip(test_subjects, subject_scores, strict=True)
        ],
        "seeds": seed_payloads,
    }


def write_selected_state_count_evaluation(
    reports: Sequence[StateModelScoreReport],
    path: str | Path,
    **metadata: object,
) -> Path:
    """Atomically write a selected-state-count evaluation JSON artifact."""
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"state-count evaluation output already exists: {target}")
    payload = selected_state_count_evaluation_payload(reports, **metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
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
            raise FileExistsError(
                f"state-count evaluation output already exists: {target}"
            )
        os.rename(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


@dataclass(frozen=True)
class SelectedStateCountEvaluationArtifact:
    """A selected-state-count evaluation rebuilt from its models and scores."""

    root: Path
    method: str
    model_kind: str
    selection_policy: str
    selected_n_states: int
    selection_workflow_fingerprint: str
    source_contract: str
    sample_interval_seconds: float | None
    minimum_sequence_length: int
    feature_contract_fingerprint: str
    development_data_fingerprint: str
    development_subjects: tuple[str, ...]
    test_data_fingerprint: str
    test_subjects: tuple[str, ...]
    fit_configuration: Mapping[str, object]
    model_seeds: tuple[int, ...]
    reports: tuple[StateModelScoreReport, ...]
    subject_scores: tuple[float, ...]
    cohort_score: float
    artifact_fingerprint: str


def load_selected_state_count_evaluation(
    path: str | Path,
) -> SelectedStateCountEvaluationArtifact:
    """Load an outer evaluation and reconstruct it from every model and score."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"state-count evaluation directory does not exist: {root}")
    raw = _load_json(root / "evaluation.json")
    expected_fields = {
        "cohort_score",
        "development_data_fingerprint",
        "development_subjects",
        "feature_contract_fingerprint",
        "fit_configuration",
        "format",
        "format_version",
        "method",
        "minimum_sequence_length",
        "model_kind",
        "model_seeds",
        "n_models",
        "n_runs",
        "n_samples",
        "n_seeds",
        "n_sequences",
        "n_subjects",
        "omitted_short_sequence_count",
        "sample_interval_seconds",
        "score_weighting",
        "seeds",
        "selected_n_states",
        "selection_direction",
        "selection_metric",
        "selection_policy",
        "selection_workflow_fingerprint",
        "source_contract",
        "subject_scores",
        "test_data_fingerprint",
        "test_subjects",
    }
    if set(raw) != expected_fields:
        raise ValueError("state-count evaluation fields do not match the schema")
    if raw["format"] != FORMAT_NAME or raw["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported state-count evaluation format or version")
    seeds = raw["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("state-count evaluation seeds must be a non-empty array")

    reports: list[StateModelScoreReport] = []
    model_references: list[str] = []
    score_references: list[str] = []
    for record in seeds:
        if not isinstance(record, dict) or set(record) != {
            "cohort_score",
            "model_artifact",
            "model_fingerprint",
            "model_seed",
            "score_artifact",
            "subject_scores",
        }:
            raise ValueError("state-count evaluation seed fields do not match the schema")
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
        model = load_fitted_model(model_path)
        report = load_state_model_scores(score_path)
        if not isinstance(model, (KMeansStateModel, GaussianHMMStateModel)):
            raise TypeError("state-count evaluation model artifact has the wrong type")
        fingerprint = fitted_model_fingerprint(model)
        if (
            fingerprint != record["model_fingerprint"]
            or fingerprint != report.model_fingerprint
            or model.seed != record["model_seed"]
            or model.seed != report.model_seed
            or model.n_states != raw["selected_n_states"]
            or model.n_states != report.n_states
            or model.fit_subjects != report.fit_subjects
            or model.training_data_fingerprint != report.training_data_fingerprint
            or dict(report.model_specification or {}) != state_model_specification(model)
        ):
            raise ValueError("state-count evaluation model, score, and summary identities disagree")
        if (raw["method"] == "kmeans") != isinstance(model, KMeansStateModel):
            raise TypeError("state-count evaluation model family disagrees with its method")
        model_references.append(model_reference)
        score_references.append(score_reference)
        reports.append(report)

    rebuilt = selected_state_count_evaluation_payload(
        reports,
        model_artifacts=model_references,
        score_artifacts=score_references,
        method=raw["method"],
        selection_policy=raw["selection_policy"],
        selection_workflow_fingerprint=raw["selection_workflow_fingerprint"],
        selected_n_states=raw["selected_n_states"],
        development_data_fingerprint=raw["development_data_fingerprint"],
        development_subjects=raw["development_subjects"],
        test_data_fingerprint=raw["test_data_fingerprint"],
        test_subjects=raw["test_subjects"],
        fit_configuration=raw["fit_configuration"],
    )
    if raw != rebuilt:
        raise ValueError("state-count evaluation cannot be rebuilt from its models and scores")
    canonical = json.dumps(
        rebuilt,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return SelectedStateCountEvaluationArtifact(
        root=root,
        method=str(rebuilt["method"]),
        model_kind=str(rebuilt["model_kind"]),
        selection_policy=str(rebuilt["selection_policy"]),
        selected_n_states=int(rebuilt["selected_n_states"]),
        selection_workflow_fingerprint=str(rebuilt["selection_workflow_fingerprint"]),
        source_contract=str(rebuilt["source_contract"]),
        sample_interval_seconds=(
            None
            if rebuilt["sample_interval_seconds"] is None
            else float(rebuilt["sample_interval_seconds"])
        ),
        minimum_sequence_length=int(rebuilt["minimum_sequence_length"]),
        feature_contract_fingerprint=str(rebuilt["feature_contract_fingerprint"]),
        development_data_fingerprint=str(rebuilt["development_data_fingerprint"]),
        development_subjects=tuple(rebuilt["development_subjects"]),
        test_data_fingerprint=str(rebuilt["test_data_fingerprint"]),
        test_subjects=tuple(rebuilt["test_subjects"]),
        fit_configuration=MappingProxyType(dict(rebuilt["fit_configuration"])),
        model_seeds=tuple(int(seed) for seed in rebuilt["model_seeds"]),
        reports=tuple(reports),
        subject_scores=tuple(
            float(record["score"]) for record in rebuilt["subject_scores"]
        ),
        cohort_score=float(rebuilt["cohort_score"]),
        artifact_fingerprint=hashlib.sha256(canonical).hexdigest(),
    )


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "SelectedStateCountEvaluationArtifact",
    "load_selected_state_count_evaluation",
    "selected_state_count_evaluation_payload",
    "write_selected_state_count_evaluation",
]
