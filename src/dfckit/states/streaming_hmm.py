"""Gaussian HMM fitting from chunked feature stores.

The original feature space stays on disk.  IncrementalPCA is fitted over
eligible censor-bounded sequences, then only the reduced observations are
materialized for hmmlearn, together with the explicit sequence-length vector.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._arrays import readonly_copy as _readonly
from ..storage import FeatureStore
from .data import _require_matching_feature_space
from .hmm import (
    GaussianHMMFitResult,
    GaussianHMMStateModel,
    GaussianHMMStateResult,
    _decode_reduced,
    _fit_reduced_hmm,
    _optional_dependencies,
    _reconstruct_estimator,
)
from .scoring import RunGaussianHMMScore
from .streaming import (
    StreamingPCAModel,
    _nonnegative_seed,
    _positive_int,
    _selected_sequence_indices,
    _subjects,
    fit_incremental_pca_store,
    iter_pca_store_chunks,
)


def _pca_view(model: GaussianHMMStateModel) -> StreamingPCAModel:
    """Represent frozen HMM preprocessing through the streaming PCA contract."""
    return StreamingPCAModel(
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        pca_mean=model.pca_mean,
        pca_components=model.pca_components,
        explained_variance_ratio=model.pca_explained_variance_ratio,
        feature_keys=model.feature_keys,
        source_contract=model.source_contract,
        sample_interval_seconds=model.sample_interval_seconds,
        n_components=model.n_pca_components,
        standardize_features=True,
        batch_size=1,
        fit_subjects=model.fit_subjects,
        fit_sample_count=model.fit_sample_count,
        fit_sequence_count=model.fit_sequence_count,
        implementation=model.implementation,
    )


@dataclass(frozen=True)
class _ReducedSequence:
    values: NDArray[np.float64]
    sample_start_indices: NDArray[np.int64]
    sample_end_indices: NDArray[np.int64]
    subject: str
    session: str | None
    acquisition_id: str | None
    segment_id: int
    sequence_index: int


def _collect_reduced_sequences(
    pca_model: StreamingPCAModel,
    store: FeatureStore,
    *,
    subjects: tuple[str, ...],
    sequence_indices: tuple[int, ...],
    allow_fit_subjects: bool,
) -> tuple[_ReducedSequence, ...]:
    values_by_sequence: dict[int, list[NDArray[np.float64]]] = {}
    starts_by_sequence: dict[int, list[NDArray[np.int64]]] = {}
    ends_by_sequence: dict[int, list[NDArray[np.int64]]] = {}
    allowed = set(sequence_indices)
    metadata = {
        index: identity
        for index, identity in enumerate(store.sequence_identities)
        if index in allowed
    }
    for chunk in iter_pca_store_chunks(
        pca_model,
        store,
        subjects=subjects,
        allow_fit_subjects=allow_fit_subjects,
        sequence_indices=sequence_indices,
    ):
        values_by_sequence.setdefault(chunk.sequence_index, []).append(
            np.asarray(chunk.values, dtype=np.float64)
        )
        starts_by_sequence.setdefault(chunk.sequence_index, []).append(
            np.asarray(chunk.sample_start_indices, dtype=np.int64)
        )
        ends_by_sequence.setdefault(chunk.sequence_index, []).append(
            np.asarray(chunk.sample_end_indices, dtype=np.int64)
        )
    output = []
    for sequence_index in sequence_indices:
        try:
            identity = metadata[sequence_index]
            values = np.concatenate(values_by_sequence[sequence_index], axis=0)
            starts = np.concatenate(starts_by_sequence[sequence_index])
            ends = np.concatenate(ends_by_sequence[sequence_index])
        except KeyError as error:
            raise RuntimeError(f"missing reduced observations for sequence {sequence_index}") from error
        output.append(
            _ReducedSequence(
                values=values,
                sample_start_indices=starts,
                sample_end_indices=ends,
                subject=identity[0],
                session=identity[1],
                acquisition_id=identity[2],
                segment_id=identity[3],
                sequence_index=sequence_index,
            )
        )
    return tuple(output)


def _decode_store(
    model: GaussianHMMStateModel,
    store: FeatureStore,
    *,
    subjects: tuple[str, ...],
    allow_fit_subjects: bool,
) -> GaussianHMMStateResult:
    pca_model = _pca_view(model)
    sequence_indices = _selected_sequence_indices(
        store,
        subjects,
        minimum_length=model.minimum_sequence_length,
    )
    sequences = _collect_reduced_sequences(
        pca_model,
        store,
        subjects=subjects,
        sequence_indices=sequence_indices,
        allow_fit_subjects=allow_fit_subjects,
    )
    return _decode_reduced(model, sequences, tuple(sequence.values for sequence in sequences))


def fit_gaussian_hmm_store(
    store: FeatureStore,
    *,
    n_states: int,
    seed: int,
    n_pca_components: int,
    covariance_type: str = "diag",
    n_init: int = 1,
    n_iter: int = 200,
    tol: float = 1e-3,
    pca_batch_size: int = 4096,
    subjects: Iterable[str] | None = None,
    minimum_sequence_length: int = 2,
) -> GaussianHMMFitResult:
    """Fit a Gaussian HMM while keeping the original feature store out of memory.

    PCA reduction is required explicitly.  Only the reduced observations are
    materialized for hmmlearn; sequence lengths remain separate so HMM
    likelihoods and transitions cannot cross censor gaps.
    """
    n_states = _positive_int(n_states, "n_states")
    if n_states < 2:
        raise ValueError("n_states must be at least two")
    seed = _nonnegative_seed(seed)
    n_pca_components = _positive_int(n_pca_components, "n_pca_components")
    if covariance_type not in {"diag", "full"}:
        raise ValueError("covariance_type must be 'diag' or 'full'")
    n_init = _positive_int(n_init, "n_init")
    n_iter = _positive_int(n_iter, "n_iter")
    pca_batch_size = _positive_int(pca_batch_size, "pca_batch_size")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    minimum_sequence_length = _positive_int(
        minimum_sequence_length, "minimum_sequence_length"
    )
    if minimum_sequence_length < 2:
        raise ValueError("minimum_sequence_length must be at least two")
    selected_subjects = _subjects(store, subjects)
    eligible_indices = _selected_sequence_indices(
        store,
        selected_subjects,
        minimum_length=minimum_sequence_length,
    )
    pca_model = fit_incremental_pca_store(
        store,
        n_components=n_pca_components,
        batch_size=pca_batch_size,
        subjects=selected_subjects,
        minimum_sequence_length=minimum_sequence_length,
    )
    sequences = _collect_reduced_sequences(
        pca_model,
        store,
        subjects=selected_subjects,
        sequence_indices=eligible_indices,
        allow_fit_subjects=True,
    )
    observations = np.concatenate([sequence.values for sequence in sequences], axis=0)
    lengths = [sequence.values.shape[0] for sequence in sequences]
    if len(observations) < n_states:
        raise ValueError("n_states cannot exceed the number of fitted samples")

    hmmlearn, sklearn, _, _ = _optional_dependencies()
    fitted = _fit_reduced_hmm(
        observations, lengths, n_states=n_states, covariance_type=covariance_type,
        n_iter=n_iter, tol=tol, seed=seed, n_init=n_init,
    )
    standardized_means = fitted["reduced_means"] @ pca_model.pca_components + pca_model.pca_mean
    emission_means = standardized_means * pca_model.feature_scale + pca_model.feature_mean
    selected_subjects_with_sequences = pca_model.fit_subjects
    selected_set = set(selected_subjects)
    all_selected_count = sum(
        identity[0] in selected_set for identity, _ in store.sequence_sample_counts
    )
    model = GaussianHMMStateModel(
        **fitted,
        emission_means=_readonly(emission_means),
        # The reduced covariance together with the frozen scaler and PCA
        # components is the compact original-space covariance representation.
        # Materializing a full states-by-features-by-features array would defeat
        # the FeatureStore memory contract for high-dimensional FC edges.
        emission_covariances=None,
        feature_mean=_readonly(pca_model.feature_mean),
        feature_scale=_readonly(pca_model.feature_scale),
        pca_mean=_readonly(pca_model.pca_mean),
        pca_components=_readonly(pca_model.pca_components),
        pca_explained_variance_ratio=_readonly(pca_model.explained_variance_ratio),
        feature_keys=store.feature_keys,
        source_contract=store.source_contract,
        sample_interval_seconds=store.sample_interval_seconds,
        n_pca_components=n_pca_components,
        minimum_sequence_length=minimum_sequence_length,
        pca_batch_size=pca_batch_size,
        fit_subjects=selected_subjects_with_sequences,
        fit_sample_count=len(observations),
        fit_sequence_count=len(sequences),
        omitted_short_sequence_count=all_selected_count - len(sequences),
        implementation=(
            f"hmmlearn {hmmlearn.__version__} GaussianHMM; "
            f"scikit-learn {sklearn.__version__} IncrementalPCA"
        ),
    )
    return GaussianHMMFitResult(
        model=model,
        states=_decode_store(
            model,
            store,
            subjects=selected_subjects,
            allow_fit_subjects=True,
        ),
    )


def predict_gaussian_hmm_store(
    model: GaussianHMMStateModel,
    store: FeatureStore,
    *,
    subjects: Iterable[str] | None = None,
    allow_fit_subjects: bool = False,
) -> GaussianHMMStateResult:
    """Decode store chunks with a frozen scaler, PCA, and Gaussian HMM."""
    _require_matching_feature_space(model, store, label="Gaussian HMM")
    if not isinstance(allow_fit_subjects, (bool, np.bool_)):
        raise TypeError("allow_fit_subjects must be boolean")
    selected_subjects = _subjects(store, subjects)
    overlap = sorted(set(model.fit_subjects).intersection(selected_subjects))
    if overlap and not allow_fit_subjects:
        raise ValueError(f"prediction subjects overlap Gaussian HMM fit subjects: {overlap}")
    return _decode_store(
        model,
        store,
        subjects=selected_subjects,
        allow_fit_subjects=allow_fit_subjects,
    )


def score_gaussian_hmm_store(
    model: GaussianHMMStateModel,
    store: FeatureStore,
    *,
    subjects: Iterable[str] | None = None,
    allow_fit_subjects: bool = False,
) -> tuple[RunGaussianHMMScore, ...]:
    """Score held-out acquisitions without transitions across censor gaps."""
    _require_matching_feature_space(model, store, label="Gaussian HMM")
    if not isinstance(allow_fit_subjects, (bool, np.bool_)):
        raise TypeError("allow_fit_subjects must be boolean")
    selected_subjects = _subjects(store, subjects)
    overlap = sorted(set(model.fit_subjects).intersection(selected_subjects))
    if overlap and not allow_fit_subjects:
        raise ValueError(f"scoring subjects overlap Gaussian HMM fit subjects: {overlap}")
    sequence_indices = _selected_sequence_indices(
        store,
        selected_subjects,
        minimum_length=model.minimum_sequence_length,
    )
    eligible_subjects = {
        store.sequence_identities[index][0] for index in sequence_indices
    }
    missing = sorted(set(selected_subjects) - eligible_subjects)
    if missing:
        raise ValueError(
            "scoring subjects have no sequence meeting the fitted HMM minimum length: "
            f"{missing}"
        )
    sequences = _collect_reduced_sequences(
        _pca_view(model),
        store,
        subjects=selected_subjects,
        sequence_indices=sequence_indices,
        allow_fit_subjects=allow_fit_subjects,
    )
    estimator = _reconstruct_estimator(model)
    totals: dict[tuple[str, str | None, str | None], float] = {}
    sample_counts: dict[tuple[str, str | None, str | None], int] = {}
    sequence_counts: dict[tuple[str, str | None, str | None], int] = {}
    for sequence in sequences:
        # Scoring each sequence separately restarts the fitted start
        # distribution at every censor boundary by construction.
        score = float(estimator.score(sequence.values, [len(sequence.values)]))
        if not np.isfinite(score):
            raise ValueError("Gaussian HMM produced a non-finite held-out log likelihood")
        identity = (sequence.subject, sequence.session, sequence.acquisition_id)
        totals[identity] = totals.get(identity, 0.0) + score
        sample_counts[identity] = sample_counts.get(identity, 0) + len(sequence.values)
        sequence_counts[identity] = sequence_counts.get(identity, 0) + 1

    return tuple(
        RunGaussianHMMScore(
            subject=subject,
            session=session,
            acquisition_id=acquisition_id,
            n_samples=sample_counts[identity],
            n_sequences=sequence_counts[identity],
            log_likelihood=total,
        )
        for identity, total in totals.items()
        for subject, session, acquisition_id in (identity,)
    )


__all__ = [
    "fit_gaussian_hmm_store",
    "predict_gaussian_hmm_store",
    "score_gaussian_hmm_store",
]
