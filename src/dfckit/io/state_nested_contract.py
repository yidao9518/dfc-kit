"""Shared current-data validation for nested state-count child artifacts."""

from __future__ import annotations

import numpy as np

from ..states.cross_validation import SubjectValidationFold
from ..states.hmm import GaussianHMMStateModel
from ..states.kmeans import KMeansStateModel
from ..storage import FeatureStore
from .models import fitted_model_fingerprint
from .state_cross_validation import (
    StateCountCrossValidationArtifact,
    validate_state_count_model_configuration,
)
from .state_evaluation import SelectedStateCountEvaluationArtifact
from .state_scoring import StateModelScoreReport, state_model_specification


def sample_intervals_match(left: float | None, right: float | None) -> bool:
    """Return whether optional sampling intervals agree at storage precision."""
    return (left is None and right is None) or (
        left is not None
        and right is not None
        and np.isclose(left, right, rtol=0.0, atol=1e-9)
    )


def validate_nested_selection(
    selection: StateCountCrossValidationArtifact,
    store: FeatureStore,
    fold: SubjectValidationFold,
    *,
    method: str,
    candidates: tuple[int, ...],
    model_seeds: tuple[int, ...],
    inner_n_folds: int,
    inner_split_seed: int,
    configuration: dict[str, object],
    minimum_sequence_length: int,
) -> None:
    """Validate a complete inner workflow against its plan and current data."""
    if (
        selection.method != method
        or selection.development_subjects != fold.fit_subjects
        or selection.candidate_n_states != candidates
        or selection.model_seeds != model_seeds
        or selection.split_seed != inner_split_seed
        or selection.comparison.selection.n_folds != inner_n_folds
        or dict(selection.fit_configuration) != configuration
        or selection.source_contract != store.source_contract
        or selection.feature_contract_fingerprint
        != store.feature_contract_fingerprint()
        or not sample_intervals_match(
            selection.sample_interval_seconds,
            store.sample_interval_seconds,
        )
        or selection.development_data_fingerprint
        != store.data_fingerprint(
            subjects=fold.fit_subjects,
            minimum_sequence_length=minimum_sequence_length,
        )
    ):
        raise ValueError(
            "nested inner selection does not match its plan or current data"
        )
    for inner_fold in selection.comparison.folds:
        if (
            inner_fold.training_data_fingerprint
            != store.data_fingerprint(
                subjects=inner_fold.fit_subjects,
                minimum_sequence_length=minimum_sequence_length,
            )
            or inner_fold.evaluation_data_fingerprint
            != store.data_fingerprint(
                subjects=inner_fold.evaluation_subjects,
                minimum_sequence_length=minimum_sequence_length,
            )
        ):
            raise ValueError("nested inner-fold data changed since selection")


def validate_state_count_cell_model(
    model: KMeansStateModel | GaussianHMMStateModel,
    store: FeatureStore,
    fold: SubjectValidationFold,
    *,
    method: str,
    n_states: int,
    seed: int,
    configuration: dict[str, object],
    minimum_sequence_length: int,
) -> None:
    """Validate one reusable inner-grid model against its exact current plan."""
    expected_type = KMeansStateModel if method == "kmeans" else GaussianHMMStateModel
    if not isinstance(model, expected_type):
        raise TypeError("checkpoint model family disagrees with the inner-grid plan")
    if (
        model.n_states != n_states
        or model.seed != seed
        or model.fit_subjects != fold.fit_subjects
        or model.feature_keys != store.feature_keys
        or model.source_contract != store.source_contract
        or not sample_intervals_match(
            model.sample_interval_seconds,
            store.sample_interval_seconds,
        )
        or model.training_data_fingerprint
        != store.data_fingerprint(
            subjects=fold.fit_subjects,
            minimum_sequence_length=minimum_sequence_length,
        )
    ):
        raise ValueError(
            "checkpoint model does not match its inner-grid cell or current data"
        )
    validate_state_count_model_configuration(model, configuration)


def validate_state_count_cell_score(
    report: StateModelScoreReport,
    model: KMeansStateModel | GaussianHMMStateModel,
    store: FeatureStore,
    fold: SubjectValidationFold,
    *,
    evaluation_data_fingerprint: str,
    minimum_sequence_length: int,
) -> None:
    """Validate one reusable inner-grid score against its model and held-out data."""
    selected = set(fold.evaluation_subjects)
    expected_omitted = sum(
        identity[0] in selected and count < minimum_sequence_length
        for identity, count in store.sequence_sample_counts
    )
    model_kind = (
        "kmeans-state"
        if isinstance(model, KMeansStateModel)
        else "gaussian-hmm-state"
    )
    model_fingerprint = fitted_model_fingerprint(model)
    if (
        report.model_kind != model_kind
        or report.model_fingerprint != model_fingerprint
        or report.model_seed != model.seed
        or report.n_states != model.n_states
        or report.fit_subjects != model.fit_subjects
        or report.training_data_fingerprint != model.training_data_fingerprint
        or report.evaluation_data_fingerprint != evaluation_data_fingerprint
        or report.feature_contract_fingerprint
        != store.feature_contract_fingerprint()
        or report.source_contract != store.source_contract
        or not sample_intervals_match(
            report.sample_interval_seconds,
            store.sample_interval_seconds,
        )
        or report.minimum_sequence_length != minimum_sequence_length
        or report.omitted_short_sequence_count != expected_omitted
        or report.allow_fit_subjects
        or dict(report.model_specification) != state_model_specification(model)
        or set(report.subjects) != selected
    ):
        raise ValueError(
            "checkpoint score does not match its model, inner-grid cell, or current data"
        )


def validate_nested_evaluation(
    evaluation: SelectedStateCountEvaluationArtifact,
    selection: StateCountCrossValidationArtifact,
    store: FeatureStore,
    fold: SubjectValidationFold,
    *,
    selection_policy: str,
    test_fingerprint: str,
    configuration: dict[str, object],
) -> None:
    """Validate a complete outer evaluation against its selection and data."""
    if (
        evaluation.method != selection.method
        or evaluation.development_subjects != fold.fit_subjects
        or evaluation.test_subjects != fold.evaluation_subjects
        or evaluation.selection_policy != selection_policy
        or evaluation.selection_workflow_fingerprint != selection.workflow_fingerprint
        or evaluation.selected_n_states
        != selection.selected_n_states(selection_policy)
        or evaluation.model_seeds != selection.model_seeds
        or dict(evaluation.fit_configuration) != configuration
        or evaluation.source_contract != store.source_contract
        or not sample_intervals_match(
            evaluation.sample_interval_seconds,
            store.sample_interval_seconds,
        )
        or evaluation.feature_contract_fingerprint
        != store.feature_contract_fingerprint()
        or evaluation.development_data_fingerprint
        != selection.development_data_fingerprint
        or evaluation.test_data_fingerprint != test_fingerprint
    ):
        raise ValueError(
            "nested outer evaluation does not match its selection or current data"
        )


__all__ = [
    "sample_intervals_match",
    "validate_nested_evaluation",
    "validate_nested_selection",
    "validate_state_count_cell_model",
    "validate_state_count_cell_score",
]
