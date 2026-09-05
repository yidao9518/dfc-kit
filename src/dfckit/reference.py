"""Subject-balanced reference templates and leakage-safe scoring."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._arrays import readonly_copy as _readonly
from ._validation import validated_subject_labels
from .connectivity.lowrank import (
    mean_projector_basis,
    subspace_distance,
    subspace_similarity,
)

FeatureKey = tuple[str, ...]


def _validated_feature_values(values: ArrayLike) -> NDArray[np.float64]:
    output = np.asarray(values, dtype=float)
    if output.ndim != 2 or not len(output) or not output.shape[1]:
        raise ValueError("values must be a non-empty observations-by-features array")
    if not np.isfinite(output).all():
        raise ValueError("values contain non-finite samples")
    return output


def _validated_feature_keys(
    feature_keys: Iterable[Sequence[str]],
    *,
    n_features: int,
) -> tuple[FeatureKey, ...]:
    keys = tuple(tuple(str(part) for part in key) for key in feature_keys)
    if len(keys) != n_features or any(not key for key in keys):
        raise ValueError("feature_keys must identify every feature")
    if len(set(keys)) != len(keys):
        raise ValueError("feature_keys must be unique")
    return keys


def _validated_bases(
    bases: Iterable[ArrayLike],
) -> tuple[NDArray[np.float64], ...]:
    output = tuple(np.asarray(basis, dtype=float) for basis in bases)
    if not output:
        raise ValueError("at least one subspace basis is required")
    shape = output[0].shape
    if len(shape) != 2 or shape[0] < 2 or shape[1] < 1 or shape[1] > shape[0]:
        raise ValueError("each basis must be a valid ROI-by-rank matrix")
    for index, basis in enumerate(output):
        if basis.shape != shape:
            raise ValueError("all bases must have the same ROI and rank dimensions")
        if not np.isfinite(basis).all():
            raise ValueError(f"basis {index} contains non-finite values")
        if not np.allclose(basis.T @ basis, np.eye(shape[1]), rtol=1e-7, atol=1e-9):
            raise ValueError(f"basis {index} columns must be orthonormal")
    return output


def _subject_groups(
    subjects: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[NDArray[np.int64], ...]]:
    unique = tuple(dict.fromkeys(subjects))
    groups = tuple(
        np.flatnonzero(np.asarray(subjects, dtype=object) == subject).astype(np.int64)
        for subject in unique
    )
    return unique, groups


def _validate_scoring_identity(
    *,
    fit_subjects: tuple[str, ...],
    subjects: tuple[str, ...],
    allow_fit_subjects: bool,
) -> None:
    if not isinstance(allow_fit_subjects, (bool, np.bool_)):
        raise TypeError("allow_fit_subjects must be boolean")
    overlap = sorted(set(fit_subjects).intersection(subjects))
    if overlap and not allow_fit_subjects:
        raise ValueError(f"scoring subjects overlap reference-fit subjects: {overlap}")


@dataclass(frozen=True)
class FeatureReferenceModel:
    """A feature template with equal total weight for every fitted participant."""

    template: NDArray[np.float64]
    subject_means: NDArray[np.float64]
    fit_subjects: tuple[str, ...]
    observation_counts: NDArray[np.int64]
    feature_keys: tuple[FeatureKey, ...]

    def template_excluding(self, subject: str) -> NDArray[np.float64]:
        """Return a participant-balanced template that excludes one fit subject."""
        if subject not in self.fit_subjects:
            raise KeyError(f"subject {subject!r} was not used to fit this reference")
        if len(self.fit_subjects) < 2:
            raise ValueError("leave-one-subject-out template requires at least two fit subjects")
        position = self.fit_subjects.index(subject)
        retained = np.arange(len(self.fit_subjects)) != position
        return _readonly(self.subject_means[retained].mean(axis=0))


@dataclass(frozen=True)
class SubspaceReferenceModel:
    """A participant-balanced mean-projector subspace reference."""

    basis: NDArray[np.float64]
    subject_bases: tuple[NDArray[np.float64], ...]
    fit_subjects: tuple[str, ...]
    observation_counts: NDArray[np.int64]
    roi_names: tuple[str, ...]
    rank: int
    loo_distances: NDArray[np.float64]
    loo_distance_mean: float
    loo_distance_scale: float


@dataclass(frozen=True)
class SubspaceReferenceScores:
    """Distances to a fixed reference and their HC-LOO standardization."""

    distances: NDArray[np.float64]
    standardized_distances: NDArray[np.float64]
    subjects: tuple[str, ...]
    reference_fit_subjects: tuple[str, ...]
    loo_distance_mean: float
    loo_distance_scale: float


def fit_feature_reference(
    values: ArrayLike,
    subjects: Iterable[str],
    feature_keys: Iterable[Sequence[str]],
) -> FeatureReferenceModel:
    """Fit an equal-participant-weight feature template."""
    data = _validated_feature_values(values)
    identifiers = validated_subject_labels(subjects, n_observations=len(data))
    keys = _validated_feature_keys(feature_keys, n_features=data.shape[1])
    fit_subjects, groups = _subject_groups(identifiers)
    if len(fit_subjects) < 2:
        raise ValueError("feature reference requires at least two participants")
    subject_means = np.stack([data[group].mean(axis=0) for group in groups])
    counts = np.asarray([len(group) for group in groups], dtype=np.int64)
    return FeatureReferenceModel(
        template=_readonly(subject_means.mean(axis=0)),
        subject_means=_readonly(subject_means),
        fit_subjects=fit_subjects,
        observation_counts=_readonly(counts),
        feature_keys=keys,
    )


def pearson_pattern_similarity(
    values: ArrayLike,
    template: ArrayLike,
) -> NDArray[np.float64]:
    """Return row-wise Pearson spatial similarity to one feature template."""
    data = _validated_feature_values(values)
    target = np.asarray(template, dtype=float)
    if target.ndim != 1 or len(target) != data.shape[1]:
        raise ValueError("template must be one-dimensional and match the feature count")
    if not np.isfinite(target).all():
        raise ValueError("template contains non-finite values")
    centered = data - data.mean(axis=1, keepdims=True)
    centered_target = target - target.mean()
    denominator = np.linalg.norm(centered, axis=1) * np.linalg.norm(centered_target)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 1e-12):
        raise ValueError("Pearson pattern similarity is undefined for a constant pattern")
    return _readonly((centered @ centered_target) / denominator)


def score_feature_reference(
    model: FeatureReferenceModel,
    values: ArrayLike,
    subjects: Iterable[str],
    feature_keys: Iterable[Sequence[str]],
    *,
    allow_fit_subjects: bool = False,
) -> NDArray[np.float64]:
    """Score observations against a fixed subject-balanced feature template."""
    data = _validated_feature_values(values)
    identifiers = validated_subject_labels(subjects, n_observations=len(data))
    keys = _validated_feature_keys(feature_keys, n_features=data.shape[1])
    if keys != model.feature_keys:
        raise ValueError("reference model and values use different feature identities or order")
    _validate_scoring_identity(
        fit_subjects=model.fit_subjects,
        subjects=identifiers,
        allow_fit_subjects=allow_fit_subjects,
    )
    return pearson_pattern_similarity(data, model.template)


def leave_one_subject_out_feature_similarity(
    model: FeatureReferenceModel,
    values: ArrayLike,
    subjects: Iterable[str],
    feature_keys: Iterable[Sequence[str]],
) -> NDArray[np.float64]:
    """Score fitted-participant observations against templates excluding that participant."""
    data = _validated_feature_values(values)
    identifiers = validated_subject_labels(subjects, n_observations=len(data))
    keys = _validated_feature_keys(feature_keys, n_features=data.shape[1])
    if keys != model.feature_keys:
        raise ValueError("reference model and values use different feature identities or order")
    unknown = sorted(set(identifiers).difference(model.fit_subjects))
    if unknown:
        raise ValueError(f"LOO scoring contains subjects absent from the reference fit: {unknown}")
    output = np.empty(len(data), dtype=float)
    for subject in dict.fromkeys(identifiers):
        selected = np.asarray([identifier == subject for identifier in identifiers])
        output[selected] = pearson_pattern_similarity(
            data[selected],
            model.template_excluding(subject),
        )
    return _readonly(output)


def subject_balanced_quantiles(
    values: ArrayLike,
    subjects: Iterable[str],
    probabilities: Iterable[float],
) -> NDArray[np.float64]:
    """Return weighted quantiles giving every participant equal total weight."""
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("values must be a non-empty finite one-dimensional array")
    identifiers = validated_subject_labels(subjects, n_observations=len(samples))
    requested = np.asarray(tuple(probabilities), dtype=float)
    if requested.ndim != 1 or not len(requested):
        raise ValueError("probabilities must be a non-empty one-dimensional collection")
    if not np.isfinite(requested).all() or np.any((requested < 0.0) | (requested > 1.0)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    _, groups = _subject_groups(identifiers)
    weights = np.empty(len(samples), dtype=float)
    for group in groups:
        weights[group] = 1.0 / len(group)
    order = np.argsort(samples, kind="stable")
    sorted_values = samples[order]
    sorted_weights = weights[order]
    positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return _readonly(
        np.interp(
            requested,
            positions,
            sorted_values,
            left=sorted_values[0],
            right=sorted_values[-1],
        )
    )


def hierarchical_balanced_quantiles(
    values: ArrayLike,
    group_levels: Iterable[Iterable[Hashable]],
    probabilities: Iterable[float],
) -> NDArray[np.float64]:
    """Return inverse-ECDF quantiles with equal weight at each group level.

    For levels such as ``(subject, acquisition)``, every subject receives equal
    total weight, acquisitions receive equal weight within their subject, and
    observations receive equal weight within their acquisition.
    """
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("values must be a non-empty finite one-dimensional array")
    levels = tuple(tuple(level) for level in group_levels)
    if not levels:
        raise ValueError("group_levels must contain at least one grouping level")
    if any(len(level) != len(samples) for level in levels):
        raise ValueError("every grouping level must identify every observation")
    paths = tuple(tuple(level[index] for level in levels) for index in range(len(samples)))
    try:
        for path in paths:
            hash(path)
    except TypeError as error:
        raise TypeError("group levels must contain hashable labels") from error

    requested = np.asarray(tuple(probabilities), dtype=float)
    if requested.ndim != 1 or not len(requested):
        raise ValueError("probabilities must be a non-empty one-dimensional collection")
    if not np.isfinite(requested).all() or np.any((requested < 0.0) | (requested > 1.0)):
        raise ValueError("probabilities must be finite and within [0, 1]")

    weights = np.ones(len(samples), dtype=float)
    for depth in range(len(levels)):
        children: dict[tuple[Hashable, ...], set[Hashable]] = {}
        for path in paths:
            children.setdefault(path[:depth], set()).add(path[depth])
        for index, path in enumerate(paths):
            weights[index] /= len(children[path[:depth]])
    leaf_counts: dict[tuple[Hashable, ...], int] = {}
    for path in paths:
        leaf_counts[path] = leaf_counts.get(path, 0) + 1
    for index, path in enumerate(paths):
        weights[index] /= leaf_counts[path]

    order = np.argsort(samples, kind="stable")
    ordered_values = samples[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    indices = np.searchsorted(cumulative, requested, side="left")
    indices = np.minimum(indices, len(ordered_values) - 1)
    return _readonly(ordered_values[indices])


def fit_subspace_reference(
    bases: Iterable[ArrayLike],
    subjects: Iterable[str],
    roi_names: Iterable[str],
) -> SubspaceReferenceModel:
    """Fit a two-level participant-balanced mean-projector reference."""
    validated = _validated_bases(bases)
    identifiers = validated_subject_labels(subjects, n_observations=len(validated))
    names = tuple(str(name) for name in roi_names)
    if len(names) != validated[0].shape[0] or len(set(names)) != len(names):
        raise ValueError("roi_names must uniquely identify every basis row")
    fit_subjects, groups = _subject_groups(identifiers)
    if len(fit_subjects) < 3:
        raise ValueError("subspace reference requires at least three participants for LOO scoring")
    rank = validated[0].shape[1]
    subject_bases = tuple(
        mean_projector_basis([validated[index] for index in group], rank=rank)
        for group in groups
    )
    reference = mean_projector_basis(subject_bases, rank=rank)
    loo = np.empty(len(fit_subjects), dtype=float)
    for index, subject_basis in enumerate(subject_bases):
        retained = tuple(
            basis for other, basis in enumerate(subject_bases) if other != index
        )
        loo_reference = mean_projector_basis(retained, rank=rank)
        loo[index] = subspace_distance(subject_basis, loo_reference)
    return SubspaceReferenceModel(
        basis=reference,
        subject_bases=subject_bases,
        fit_subjects=fit_subjects,
        observation_counts=_readonly(
            np.asarray([len(group) for group in groups], dtype=np.int64)
        ),
        roi_names=names,
        rank=rank,
        loo_distances=_readonly(loo),
        loo_distance_mean=float(loo.mean()),
        loo_distance_scale=float(loo.std(ddof=1)),
    )


def score_subspace_reference(
    model: SubspaceReferenceModel,
    bases: Iterable[ArrayLike],
    subjects: Iterable[str],
    roi_names: Iterable[str],
    *,
    allow_fit_subjects: bool = False,
) -> SubspaceReferenceScores:
    """Score subspaces against a fixed reference and the HC-LOO benchmark."""
    validated = _validated_bases(bases)
    identifiers = validated_subject_labels(subjects, n_observations=len(validated))
    names = tuple(str(name) for name in roi_names)
    if names != model.roi_names:
        raise ValueError("reference model and bases use different ROI identities or order")
    if validated[0].shape != model.basis.shape:
        raise ValueError("reference model and bases use different ROI or rank dimensions")
    _validate_scoring_identity(
        fit_subjects=model.fit_subjects,
        subjects=identifiers,
        allow_fit_subjects=allow_fit_subjects,
    )
    if not np.isfinite(model.loo_distance_scale) or model.loo_distance_scale <= 1e-12:
        raise ValueError("LOO reference distance scale is undefined")
    distances = np.asarray(
        [subspace_distance(basis, model.basis) for basis in validated],
        dtype=float,
    )
    standardized = (distances - model.loo_distance_mean) / model.loo_distance_scale
    return SubspaceReferenceScores(
        distances=_readonly(distances),
        standardized_distances=_readonly(standardized),
        subjects=identifiers,
        reference_fit_subjects=model.fit_subjects,
        loo_distance_mean=model.loo_distance_mean,
        loo_distance_scale=model.loo_distance_scale,
    )


def reference_subspace_similarity(
    model: SubspaceReferenceModel,
    basis: ArrayLike,
    roi_names: Iterable[str],
) -> float:
    """Compare one basis with a reference after explicit ROI-order validation."""
    names = tuple(str(name) for name in roi_names)
    if names != model.roi_names:
        raise ValueError("reference model and basis use different ROI identities or order")
    validated = _validated_bases([basis])[0]
    if validated.shape != model.basis.shape:
        raise ValueError("reference model and basis use different ROI or rank dimensions")
    return subspace_similarity(validated, model.basis)
