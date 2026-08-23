"""Hungarian alignment of state patterns across fitted models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .._arrays import readonly_copy as _readonly
from .data import FeatureKey, StateAssignments, StateLabelSequence
from .hmm import GaussianHMMStateModel, GaussianHMMStateResult
from .kmeans import KMeansStateModel

AlignmentMetric = Literal["euclidean", "pearson"]


@dataclass(frozen=True)
class StateAlignment:
    candidate_to_reference: NDArray[np.int64]
    matched_costs: NDArray[np.float64]
    cost_matrix: NDArray[np.float64]
    reference_seed: int
    candidate_seed: int
    feature_keys: tuple[FeatureKey, ...]
    source_contract: str
    sample_interval_seconds: float | None
    metric: AlignmentMetric = "euclidean"

    def __post_init__(self) -> None:
        mapping = np.asarray(self.candidate_to_reference, dtype=np.int64)
        matched = np.asarray(self.matched_costs, dtype=float)
        costs = np.asarray(self.cost_matrix, dtype=float)
        if mapping.ndim != 1 or len(mapping) < 2:
            raise ValueError("state alignment mapping must contain at least two states")
        if set(mapping.tolist()) != set(range(len(mapping))):
            raise ValueError("state alignment mapping must be a complete permutation")
        if matched.shape != mapping.shape or costs.shape != (len(mapping), len(mapping)):
            raise ValueError("state alignment cost arrays have invalid shapes")
        if not np.isfinite(matched).all() or not np.isfinite(costs).all():
            raise ValueError("state alignment costs must be finite")
        if np.any(matched < -1e-12) or np.any(costs < -1e-12):
            raise ValueError("state alignment costs must be non-negative")
        if self.metric not in {"euclidean", "pearson"}:
            raise ValueError("state alignment metric must be 'euclidean' or 'pearson'")
        if self.metric == "pearson" and (
            np.any(matched > 2.0 + 1e-10) or np.any(costs > 2.0 + 1e-10)
        ):
            raise ValueError("Pearson alignment costs must lie within [0, 2]")
        for seed, name in (
            (self.reference_seed, "reference_seed"),
            (self.candidate_seed, "candidate_seed"),
        ):
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
                raise TypeError(f"state alignment {name} must be an integer")
        keys = tuple(tuple(str(part) for part in key) for key in self.feature_keys)
        if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("state alignment feature_keys are invalid")
        if not str(self.source_contract).strip():
            raise ValueError("state alignment source_contract must be non-empty")
        if self.sample_interval_seconds is not None and (
            not np.isfinite(self.sample_interval_seconds)
            or self.sample_interval_seconds <= 0.0
        ):
            raise ValueError("state alignment sample interval must be finite and positive")
        object.__setattr__(self, "candidate_to_reference", _readonly(mapping))
        object.__setattr__(self, "matched_costs", _readonly(matched))
        object.__setattr__(self, "cost_matrix", _readonly(costs))
        object.__setattr__(self, "reference_seed", int(self.reference_seed))
        object.__setattr__(self, "candidate_seed", int(self.candidate_seed))
        object.__setattr__(self, "feature_keys", keys)
        object.__setattr__(self, "source_contract", str(self.source_contract))
        object.__setattr__(self, "metric", str(self.metric))


def _intervals_match(left: float | None, right: float | None) -> bool:
    if (left is None) != (right is None):
        return False
    return left is None or right is None or np.isclose(left, right, rtol=0.0, atol=1e-9)


def _align_patterns(
    reference_patterns: NDArray[np.float64],
    candidate_patterns: NDArray[np.float64],
    *,
    reference_feature_scale: NDArray[np.float64],
    metric: AlignmentMetric,
    reference_seed: int,
    candidate_seed: int,
    feature_keys: tuple[FeatureKey, ...],
    source_contract: str,
    sample_interval_seconds: float | None,
) -> StateAlignment:
    reference_patterns = np.asarray(reference_patterns, dtype=float)
    candidate_patterns = np.asarray(candidate_patterns, dtype=float)
    if (
        reference_patterns.ndim != 2
        or candidate_patterns.ndim != 2
        or reference_patterns.shape != candidate_patterns.shape
        or reference_patterns.shape[0] < 2
    ):
        raise ValueError("state alignment patterns must have the same states-by-features shape")
    if not np.isfinite(reference_patterns).all() or not np.isfinite(candidate_patterns).all():
        raise ValueError("state alignment patterns must be finite")
    feature_scale = np.asarray(reference_feature_scale, dtype=float)
    if feature_scale.shape != (reference_patterns.shape[1],):
        raise ValueError("state alignment reference feature scale has an invalid shape")
    if not np.isfinite(feature_scale).all() or np.any(feature_scale <= 0.0):
        raise ValueError("state alignment reference feature scale must be finite and positive")
    if metric == "euclidean":
        differences = (
            reference_patterns[:, None, :] - candidate_patterns[None, :, :]
        ) / feature_scale[None, None, :]
        costs = np.linalg.norm(differences, axis=2)
    elif metric == "pearson":
        reference_scale = reference_patterns.std(axis=1, ddof=0)
        candidate_scale = candidate_patterns.std(axis=1, ddof=0)
        if np.any(reference_scale < 1e-12) or np.any(candidate_scale < 1e-12):
            raise ValueError("Pearson state alignment is undefined for a constant state pattern")
        reference_z = (
            reference_patterns - reference_patterns.mean(axis=1, keepdims=True)
        ) / reference_scale[:, None]
        candidate_z = (
            candidate_patterns - candidate_patterns.mean(axis=1, keepdims=True)
        ) / candidate_scale[:, None]
        correlations = reference_z @ candidate_z.T / reference_patterns.shape[1]
        costs = np.clip(1.0 - correlations, 0.0, 2.0)
    else:
        raise ValueError("metric must be 'euclidean' or 'pearson'")
    try:
        from scipy.optimize import linear_sum_assignment
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "state alignment requires the 'states' extra: pip install 'dfc-kit[states]'"
        ) from error
    reference_indices, candidate_indices = linear_sum_assignment(costs)
    mapping = np.full(reference_patterns.shape[0], -1, dtype=np.int64)
    matched = np.full(reference_patterns.shape[0], np.nan, dtype=float)
    for reference_index, candidate_index in zip(
        reference_indices, candidate_indices, strict=True
    ):
        mapping[candidate_index] = reference_index
        matched[reference_index] = costs[reference_index, candidate_index]
    if np.any(mapping < 0):
        raise RuntimeError("Hungarian state assignment was incomplete")
    return StateAlignment(
        candidate_to_reference=mapping,
        matched_costs=matched,
        cost_matrix=costs,
        reference_seed=reference_seed,
        candidate_seed=candidate_seed,
        feature_keys=feature_keys,
        source_contract=source_contract,
        sample_interval_seconds=sample_interval_seconds,
        metric=metric,
    )


def align_kmeans_centroids(
    reference: KMeansStateModel,
    candidate: KMeansStateModel,
    *,
    metric: AlignmentMetric = "euclidean",
) -> StateAlignment:
    """Find a one-to-one candidate-to-reference state mapping."""
    if reference.n_states != candidate.n_states:
        raise ValueError("state alignment requires the same n_states")
    if reference.feature_keys != candidate.feature_keys:
        raise ValueError("state alignment requires identical feature identities and order")
    if reference.source_contract != candidate.source_contract:
        raise ValueError("state alignment requires identical source contracts")
    if not _intervals_match(reference.sample_interval_seconds, candidate.sample_interval_seconds):
        raise ValueError("state alignment requires identical sample intervals")
    return _align_patterns(
        reference.centers,
        candidate.centers,
        reference_feature_scale=reference.feature_scale,
        metric=metric,
        reference_seed=reference.seed,
        candidate_seed=candidate.seed,
        feature_keys=reference.feature_keys,
        source_contract=reference.source_contract,
        sample_interval_seconds=reference.sample_interval_seconds,
    )


def align_cap_centroids(
    reference: KMeansStateModel,
    candidate: KMeansStateModel,
    *,
    metric: AlignmentMetric = "pearson",
) -> StateAlignment:
    """Align CAP state centres, using pattern-shape matching by default.

    CAP centres are relative ROI activation configurations because each input
    segment is ROI-wise z-scored. Pearson matching therefore compares the
    spatial configuration without treating a proportional amplitude change as
    a different CAP state. Pass ``metric="euclidean"`` when amplitude should
    also contribute to the CAP pairing.
    """
    if not reference.source_contract.startswith("cap:") or not candidate.source_contract.startswith(
        "cap:"
    ):
        raise ValueError("CAP alignment requires CAP source contracts")
    return align_kmeans_centroids(reference, candidate, metric=metric)


def align_gaussian_hmm_emissions(
    reference: GaussianHMMStateModel,
    candidate: GaussianHMMStateModel,
    *,
    metric: AlignmentMetric = "euclidean",
) -> StateAlignment:
    """Match candidate HMM states to reference original-feature emission means."""
    if reference.n_states != candidate.n_states:
        raise ValueError("HMM state alignment requires the same n_states")
    if reference.feature_keys != candidate.feature_keys:
        raise ValueError("HMM state alignment requires identical feature identities and order")
    if reference.source_contract != candidate.source_contract:
        raise ValueError("HMM state alignment requires identical source contracts")
    if not _intervals_match(reference.sample_interval_seconds, candidate.sample_interval_seconds):
        raise ValueError("HMM state alignment requires identical sample intervals")
    return _align_patterns(
        reference.emission_means,
        candidate.emission_means,
        reference_feature_scale=reference.feature_scale,
        metric=metric,
        reference_seed=reference.seed,
        candidate_seed=candidate.seed,
        feature_keys=reference.feature_keys,
        source_contract=reference.source_contract,
        sample_interval_seconds=reference.sample_interval_seconds,
    )


def apply_state_alignment(
    assignments: StateAssignments,
    alignment: StateAlignment,
) -> StateAssignments:
    """Relabel candidate assignments into the reference state numbering."""
    if len(alignment.candidate_to_reference) != assignments.n_states:
        raise ValueError("alignment mapping does not match assignment n_states")
    if assignments.source_contract != alignment.source_contract:
        raise ValueError("alignment and assignments use different source contracts")
    if not _intervals_match(
        assignments.sample_interval_seconds,
        alignment.sample_interval_seconds,
    ):
        raise ValueError("alignment and assignments use different sample intervals")
    sequences = tuple(
        StateLabelSequence(
            labels=alignment.candidate_to_reference[sequence.labels],
            sample_start_indices=sequence.sample_start_indices,
            sample_end_indices=sequence.sample_end_indices,
            subject=sequence.subject,
            session=sequence.session,
            acquisition_id=sequence.acquisition_id,
            segment_id=sequence.segment_id,
        )
        for sequence in assignments.sequences
    )
    return StateAssignments(
        sequences=sequences,
        n_states=assignments.n_states,
        source_contract=assignments.source_contract,
        sample_interval_seconds=assignments.sample_interval_seconds,
    )


def _reference_to_candidate(alignment: StateAlignment) -> NDArray[np.int64]:
    return np.argsort(alignment.candidate_to_reference)


def relabel_kmeans_model(
    model: KMeansStateModel,
    alignment: StateAlignment,
) -> KMeansStateModel:
    """Return a candidate KMeans model whose state axis uses reference numbering."""
    if model.n_states != len(alignment.candidate_to_reference):
        raise ValueError("alignment mapping does not match KMeans n_states")
    if model.seed != alignment.candidate_seed:
        raise ValueError("alignment candidate seed does not match the KMeans model")
    if model.feature_keys != alignment.feature_keys:
        raise ValueError("alignment and KMeans model use different feature identities")
    if model.source_contract != alignment.source_contract or not _intervals_match(
        model.sample_interval_seconds,
        alignment.sample_interval_seconds,
    ):
        raise ValueError("alignment and KMeans model use different source contracts")
    order = _reference_to_candidate(alignment)
    return replace(
        model,
        centers=_readonly(model.centers[order]),
        standardized_centers=_readonly(model.standardized_centers[order]),
        clustering_centers=_readonly(model.clustering_centers[order]),
    )


def relabel_gaussian_hmm_model(
    model: GaussianHMMStateModel,
    alignment: StateAlignment,
) -> GaussianHMMStateModel:
    """Return a candidate HMM with every state-indexed parameter in reference order."""
    if model.n_states != len(alignment.candidate_to_reference):
        raise ValueError("alignment mapping does not match Gaussian HMM n_states")
    if model.seed != alignment.candidate_seed:
        raise ValueError("alignment candidate seed does not match the Gaussian HMM model")
    if model.feature_keys != alignment.feature_keys:
        raise ValueError("alignment and Gaussian HMM use different feature identities")
    if model.source_contract != alignment.source_contract or not _intervals_match(
        model.sample_interval_seconds,
        alignment.sample_interval_seconds,
    ):
        raise ValueError("alignment and Gaussian HMM use different source contracts")
    order = _reference_to_candidate(alignment)
    emission_covariances = (
        None
        if model.emission_covariances is None
        else _readonly(model.emission_covariances[order])
    )
    return replace(
        model,
        start_probabilities=_readonly(model.start_probabilities[order]),
        transition_matrix=_readonly(model.transition_matrix[np.ix_(order, order)]),
        reduced_means=_readonly(model.reduced_means[order]),
        reduced_covariances=_readonly(model.reduced_covariances[order]),
        emission_means=_readonly(model.emission_means[order]),
        emission_covariances=emission_covariances,
    )


def apply_gaussian_hmm_alignment(
    result: GaussianHMMStateResult,
    alignment: StateAlignment,
) -> GaussianHMMStateResult:
    """Relabel HMM assignments and posterior columns into reference numbering."""
    assignments = apply_state_alignment(result.assignments, alignment)
    if len(result.posterior_probabilities) != len(result.assignments.sequences):
        raise ValueError("HMM posterior sequence count does not match assignments")
    mapping = alignment.candidate_to_reference
    aligned_posteriors = []
    for posterior, sequence in zip(
        result.posterior_probabilities,
        result.assignments.sequences,
        strict=True,
    ):
        values = np.asarray(posterior, dtype=float)
        if values.shape != (len(sequence.labels), result.assignments.n_states):
            raise ValueError("HMM posterior shape does not match its state-label sequence")
        if (
            not np.isfinite(values).all()
            or np.any(values < -1e-12)
            or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-8)
        ):
            raise ValueError("HMM posterior probabilities are invalid")
        aligned = np.empty_like(values)
        aligned[:, mapping] = values
        aligned_posteriors.append(_readonly(aligned))
    return GaussianHMMStateResult(
        assignments=assignments,
        posterior_probabilities=tuple(aligned_posteriors),
        log_likelihood=float(result.log_likelihood),
    )
