"""Strict JSON persistence for held-out state-model scores."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .._validation import (
    validated_nonnegative_integer as _nonnegative_integer,
)
from .._validation import (
    validated_positive_integer as _positive_integer,
)
from ..states.hmm import GaussianHMMStateModel
from ..states.kmeans import KMeansStateModel
from ..states.scoring import RunGaussianHMMScore, RunKMeansScore, RunStateModelScore
from ._json import load_json_object, write_json_atomic

FORMAT_NAME = "dfckit-state-model-scores"
FORMAT_VERSION = 3


def state_model_specification(
    model: KMeansStateModel | GaussianHMMStateModel,
) -> dict[str, object]:
    """Return the complete score-comparison specification of a fitted model."""
    if isinstance(model, KMeansStateModel):
        return {
            "algorithm": model.algorithm,
            "batch_size": model.batch_size,
            "implementation": model.implementation,
            "init_sample_size": model.init_sample_size,
            "max_iter": model.max_iter,
            "n_init": model.n_init,
            "n_pca_components": model.n_pca_components,
            "pca_batch_size": model.pca_batch_size,
            "reassignment_ratio": model.reassignment_ratio,
            "standardize_features": model.standardize_features,
        }
    if isinstance(model, GaussianHMMStateModel):
        return {
            "covariance_type": model.covariance_type,
            "implementation": model.implementation,
            "minimum_sequence_length": model.minimum_sequence_length,
            "n_init": model.n_init,
            "n_iter": model.n_iter,
            "n_pca_components": model.n_pca_components,
            "pca_batch_size": model.pca_batch_size,
            "tol": model.tol,
        }
    raise TypeError("state-model specification requires KMeans or Gaussian HMM")


def run_boundaries(
    report: StateModelScoreReport,
) -> tuple[tuple[object, ...], ...]:
    """Return acquisition identities used to compare score collections."""
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

_KMEANS_SPECIFICATION_FIELDS = {
    "algorithm",
    "batch_size",
    "implementation",
    "init_sample_size",
    "max_iter",
    "n_init",
    "n_pca_components",
    "pca_batch_size",
    "reassignment_ratio",
    "standardize_features",
}
_HMM_SPECIFICATION_FIELDS = {
    "covariance_type",
    "implementation",
    "minimum_sequence_length",
    "n_init",
    "n_iter",
    "n_pca_components",
    "pca_batch_size",
    "tol",
}


def _sample_interval(value: float | None) -> float | None:
    if value is None:
        return None
    output = float(value)
    if not np.isfinite(output) or output <= 0.0:
        raise ValueError("sample_interval_seconds must be finite and positive")
    return output


def _model_specification(
    value: Mapping[str, object],
    model_kind: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("model_specification must be a mapping")
    output = dict(value)
    if model_kind == "kmeans-state":
        output.setdefault("n_pca_components", None)
        output.setdefault("pca_batch_size", None)
    expected = (
        _KMEANS_SPECIFICATION_FIELDS
        if model_kind == "kmeans-state"
        else _HMM_SPECIFICATION_FIELDS
    )
    if set(output) != expected:
        raise ValueError(
            "model_specification fields do not match model_kind; "
            f"missing={sorted(expected - set(output))}, "
            f"unexpected={sorted(set(output) - expected)}"
        )
    try:
        json.dumps(output, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("model_specification must contain finite JSON values") from error
    if not isinstance(output["implementation"], str) or not str(
        output["implementation"]
    ).strip():
        raise ValueError("model_specification implementation must be non-empty")
    if model_kind == "kmeans-state":
        if output["algorithm"] not in {"lloyd", "minibatch"}:
            raise ValueError("KMeans model_specification algorithm is invalid")
        if not isinstance(output["standardize_features"], bool):
            raise TypeError("KMeans model_specification standardize_features must be boolean")
        for name in ("n_init", "max_iter"):
            _positive_integer(output[name], f"model_specification {name}")
        for name in (
            "batch_size",
            "init_sample_size",
            "n_pca_components",
            "pca_batch_size",
        ):
            if output[name] is not None:
                _positive_integer(output[name], f"model_specification {name}")
        ratio = output["reassignment_ratio"]
        if ratio is not None and (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not np.isfinite(ratio)
            or not 0.0 <= ratio <= 1.0
        ):
            raise ValueError("KMeans model_specification reassignment_ratio is invalid")
    else:
        if output["covariance_type"] not in {"diag", "full"}:
            raise ValueError("HMM model_specification covariance_type is invalid")
        for name in ("n_init", "n_iter", "n_pca_components", "minimum_sequence_length"):
            _positive_integer(output[name], f"model_specification {name}")
        if output["pca_batch_size"] is not None:
            _positive_integer(
                output["pca_batch_size"],
                "model_specification pca_batch_size",
            )
        tolerance = output["tol"]
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("HMM model_specification tol must be finite and positive")
    return output


@dataclass(frozen=True)
class StateModelScoreReport:
    """Validated in-memory representation of one score artifact."""

    scores: tuple[RunStateModelScore, ...]
    format_version: int
    model_kind: str
    model_seed: int
    n_states: int
    fit_subjects: tuple[str, ...]
    source_contract: str
    sample_interval_seconds: float | None
    minimum_sequence_length: int
    omitted_short_sequence_count: int
    allow_fit_subjects: bool
    model_specification: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", tuple(self.scores))
        specification = _model_specification(self.model_specification, self.model_kind)
        object.__setattr__(
            self,
            "model_specification",
            MappingProxyType(specification),
        )
        canonical = state_model_scores_payload(
            self.scores,
            model_kind=self.model_kind,
            model_seed=self.model_seed,
            n_states=self.n_states,
            fit_subjects=self.fit_subjects,
            source_contract=self.source_contract,
            sample_interval_seconds=self.sample_interval_seconds,
            minimum_sequence_length=self.minimum_sequence_length,
            omitted_short_sequence_count=self.omitted_short_sequence_count,
            allow_fit_subjects=self.allow_fit_subjects,
            model_specification=self.model_specification,
        )
        if canonical["format_version"] != self.format_version:
            raise ValueError("state-model score format version and provenance disagree")

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(score.subject for score in self.scores))

    @property
    def n_runs(self) -> int:
        return len(self.scores)

    @property
    def n_sequences(self) -> int:
        return sum(score.n_sequences for score in self.scores)

    @property
    def n_samples(self) -> int:
        return sum(score.n_samples for score in self.scores)


def state_model_scores_payload(
    scores: Sequence[RunStateModelScore],
    *,
    model_kind: str,
    model_seed: int,
    n_states: int,
    fit_subjects: Sequence[str],
    source_contract: str,
    sample_interval_seconds: float | None,
    minimum_sequence_length: int,
    omitted_short_sequence_count: int,
    allow_fit_subjects: bool,
    model_specification: Mapping[str, object],
) -> dict[str, object]:
    """Build a finite held-out scoring payload."""
    scores = tuple(scores)
    if not scores:
        raise ValueError("state-model scoring requires at least one run")
    if model_kind not in {"kmeans-state", "gaussian-hmm-state"}:
        raise ValueError("state-model scoring model_kind is invalid")
    expected_type = RunKMeansScore if model_kind == "kmeans-state" else RunGaussianHMMScore
    if any(not isinstance(score, expected_type) for score in scores):
        raise TypeError("state-model score type does not match model_kind")
    if not isinstance(allow_fit_subjects, (bool, np.bool_)):
        raise TypeError("allow_fit_subjects must be boolean")
    if not isinstance(source_contract, str) or not source_contract.strip():
        raise ValueError("state-model scoring source_contract must be non-empty")
    fit_subjects = tuple(fit_subjects)
    if (
        not fit_subjects
        or any(not isinstance(subject, str) or not subject.strip() for subject in fit_subjects)
        or len(set(fit_subjects)) != len(fit_subjects)
    ):
        raise ValueError("fit_subjects must contain unique non-empty labels")
    identities = tuple(
        (score.subject, score.session, score.acquisition_id) for score in scores
    )
    if len(set(identities)) != len(identities):
        raise ValueError("state-model scores contain duplicate run identities")
    score_subjects = tuple(dict.fromkeys(score.subject for score in scores))
    overlap = sorted(set(fit_subjects).intersection(score_subjects))
    if overlap and not allow_fit_subjects:
        raise ValueError(
            "state-model score fit and evaluation subjects overlap without permission: "
            f"{overlap}"
        )
    specification = _model_specification(model_specification, model_kind)
    if (
        model_kind == "gaussian-hmm-state"
        and specification["minimum_sequence_length"] != minimum_sequence_length
    ):
        raise ValueError(
            "HMM model_specification minimum_sequence_length disagrees with scoring"
        )

    runs: list[dict[str, object]] = []
    if model_kind == "kmeans-state":
        total = sum(score.total_squared_distance for score in scores)
        n_samples = sum(score.n_samples for score in scores)
        metric_name = "mean_squared_distance"
        direction = "lower is better on an identical held-out cohort"
        summary = {
            "total_squared_distance": total,
            "mean_squared_distance": total / n_samples,
        }
        for score in scores:
            assert isinstance(score, RunKMeansScore)
            runs.append(
                {
                    "subject": score.subject,
                    "session": score.session,
                    "acquisition_id": score.acquisition_id,
                    "n_samples": score.n_samples,
                    "n_sequences": score.n_sequences,
                    "total_squared_distance": score.total_squared_distance,
                    "mean_squared_distance": score.mean_squared_distance,
                }
            )
    else:
        total = sum(score.log_likelihood for score in scores)
        n_samples = sum(score.n_samples for score in scores)
        metric_name = "log_likelihood_per_sample"
        direction = "higher is better on an identical held-out cohort"
        summary = {
            "log_likelihood": total,
            "log_likelihood_per_sample": total / n_samples,
        }
        for score in scores:
            assert isinstance(score, RunGaussianHMMScore)
            runs.append(
                {
                    "subject": score.subject,
                    "session": score.session,
                    "acquisition_id": score.acquisition_id,
                    "n_samples": score.n_samples,
                    "n_sequences": score.n_sequences,
                    "log_likelihood": score.log_likelihood,
                    "log_likelihood_per_sample": score.log_likelihood_per_sample,
                }
            )

    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model_kind": model_kind,
        "model_seed": _nonnegative_integer(model_seed, "model_seed"),
        "n_states": _positive_integer(n_states, "n_states"),
        "selection_metric": metric_name,
        "selection_direction": direction,
        "fit_subjects": list(fit_subjects),
        "source_contract": source_contract,
        "sample_interval_seconds": _sample_interval(sample_interval_seconds),
        "minimum_sequence_length": _positive_integer(
            minimum_sequence_length,
            "minimum_sequence_length",
        ),
        "omitted_short_sequence_count": _nonnegative_integer(
            omitted_short_sequence_count,
            "omitted_short_sequence_count",
        ),
        "allow_fit_subjects": bool(allow_fit_subjects),
        "n_runs": len(scores),
        "n_sequences": sum(score.n_sequences for score in scores),
        "n_samples": sum(score.n_samples for score in scores),
        "subjects": list(score_subjects),
        "summary": summary,
        "runs": runs,
        "model_specification": specification,
    }
    return payload


def load_state_model_scores(path: str | Path) -> StateModelScoreReport:
    """Load and canonically validate a state-model score artifact."""
    source = Path(path)
    raw = load_json_object(source, context="state-model score artifact")
    version = raw.get("format_version")
    if (
        raw.get("format") != FORMAT_NAME
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version != FORMAT_VERSION
    ):
        raise ValueError("unsupported state-model score artifact format or version")
    expected_fields = {
        "allow_fit_subjects",
        "fit_subjects",
        "format",
        "format_version",
        "minimum_sequence_length",
        "model_kind",
        "model_seed",
        "model_specification",
        "n_runs",
        "n_samples",
        "n_sequences",
        "n_states",
        "omitted_short_sequence_count",
        "runs",
        "sample_interval_seconds",
        "selection_direction",
        "selection_metric",
        "source_contract",
        "subjects",
        "summary",
    }
    if set(raw) != expected_fields:
        raise ValueError(
            "state-model score artifact fields do not match its version; "
            f"missing={sorted(expected_fields - set(raw))}, "
            f"unexpected={sorted(set(raw) - expected_fields)}"
        )
    model_kind = raw["model_kind"]
    if model_kind not in {"kmeans-state", "gaussian-hmm-state"}:
        raise ValueError("state-model score artifact model_kind is invalid")
    run_fields = {
        "acquisition_id",
        "mean_squared_distance",
        "n_samples",
        "n_sequences",
        "session",
        "subject",
        "total_squared_distance",
    } if model_kind == "kmeans-state" else {
        "acquisition_id",
        "log_likelihood",
        "log_likelihood_per_sample",
        "n_samples",
        "n_sequences",
        "session",
        "subject",
    }
    records = raw["runs"]
    if not isinstance(records, list) or not records:
        raise ValueError("state-model score artifact runs must be a non-empty array")
    scores: list[RunStateModelScore] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != run_fields:
            raise ValueError("state-model score run fields do not match model_kind")
        common = {
            "subject": record["subject"],
            "session": record["session"],
            "acquisition_id": record["acquisition_id"],
            "n_samples": record["n_samples"],
            "n_sequences": record["n_sequences"],
        }
        if model_kind == "kmeans-state":
            score = RunKMeansScore(
                **common,
                total_squared_distance=record["total_squared_distance"],
                mean_squared_distance=record["mean_squared_distance"],
            )
        else:
            score = RunGaussianHMMScore(
                **common,
                log_likelihood=record["log_likelihood"],
                log_likelihood_per_sample=record["log_likelihood_per_sample"],
            )
        scores.append(score)
    fit_subjects = raw["fit_subjects"]
    if not isinstance(fit_subjects, list):
        raise TypeError("state-model score fit_subjects must be an array")
    report = StateModelScoreReport(
        scores=tuple(scores),
        format_version=version,
        model_kind=model_kind,
        model_seed=raw["model_seed"],
        n_states=raw["n_states"],
        fit_subjects=tuple(fit_subjects),
        source_contract=raw["source_contract"],
        sample_interval_seconds=raw["sample_interval_seconds"],
        minimum_sequence_length=raw["minimum_sequence_length"],
        omitted_short_sequence_count=raw["omitted_short_sequence_count"],
        allow_fit_subjects=raw["allow_fit_subjects"],
        model_specification=raw["model_specification"],
    )
    canonical = state_model_scores_payload(
        report.scores,
        model_kind=report.model_kind,
        model_seed=report.model_seed,
        n_states=report.n_states,
        fit_subjects=report.fit_subjects,
        source_contract=report.source_contract,
        sample_interval_seconds=report.sample_interval_seconds,
        minimum_sequence_length=report.minimum_sequence_length,
        omitted_short_sequence_count=report.omitted_short_sequence_count,
        allow_fit_subjects=report.allow_fit_subjects,
        model_specification=report.model_specification,
    )
    if raw != canonical:
        raise ValueError("state-model score artifact summary or metadata is inconsistent")
    return report


def write_state_model_scores(
    scores: Sequence[RunStateModelScore],
    path: str | Path,
    **metadata: object,
) -> Path:
    """Atomically write held-out state-model scores without JSON NaN values."""
    target = Path(path)
    payload = state_model_scores_payload(scores, **metadata)
    return write_json_atomic(target, payload)


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "StateModelScoreReport",
    "load_state_model_scores",
    "run_boundaries",
    "state_model_scores_payload",
    "state_model_specification",
    "write_state_model_scores",
]
