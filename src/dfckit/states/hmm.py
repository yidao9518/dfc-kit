"""Subject-recorded Gaussian HMM state modeling for gap-bounded sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._arrays import readonly_copy as _readonly
from .data import (
    FeatureKey,
    FeatureSequence,
    FeatureSequenceDataset,
    StateAssignments,
    StateLabelSequence,
)


@dataclass(frozen=True)
class GaussianHMMStateModel:
    """Frozen preprocessing and emission parameters for a Gaussian HMM."""

    start_probabilities: NDArray[np.float64]
    transition_matrix: NDArray[np.float64]
    reduced_means: NDArray[np.float64]
    reduced_covariances: NDArray[np.float64]
    emission_means: NDArray[np.float64]
    emission_covariances: NDArray[np.float64] | None
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    pca_mean: NDArray[np.float64]
    pca_components: NDArray[np.float64]
    pca_explained_variance_ratio: NDArray[np.float64]
    feature_keys: tuple[FeatureKey, ...]
    source_contract: str
    sample_interval_seconds: float | None
    n_states: int
    n_pca_components: int
    covariance_type: str
    seed: int
    n_init: int
    n_iter: int
    tol: float
    minimum_sequence_length: int
    pca_batch_size: int | None
    selected_initialization: int
    initialization_seeds: tuple[int, ...]
    initialization_log_likelihoods: NDArray[np.float64]
    iterations: int
    converged: bool
    log_likelihood: float
    fit_subjects: tuple[str, ...]
    fit_sample_count: int
    fit_sequence_count: int
    omitted_short_sequence_count: int
    implementation: str


@dataclass(frozen=True)
class GaussianHMMStateResult:
    """Decoded labels and posterior probabilities for eligible sequences."""

    assignments: StateAssignments
    posterior_probabilities: tuple[NDArray[np.float64], ...]
    log_likelihood: float


@dataclass(frozen=True)
class GaussianHMMFitResult:
    model: GaussianHMMStateModel
    states: GaussianHMMStateResult


def reconstruct_emission_covariance(
    model: GaussianHMMStateModel,
    state: int,
) -> NDArray[np.float64]:
    """Return one original-feature covariance, reconstructing it when omitted."""
    if isinstance(state, (bool, np.bool_)) or not isinstance(state, (int, np.integer)):
        raise TypeError("state must be an integer")
    if state < 0 or state >= model.n_states:
        raise ValueError(f"state must be within [0, {model.n_states - 1}]")
    if model.emission_covariances is not None:
        return _readonly(model.emission_covariances[int(state)])
    standardized = (
        model.pca_components.T
        @ model.reduced_covariances[int(state)]
        @ model.pca_components
    )
    return _readonly(standardized * np.outer(model.feature_scale, model.feature_scale))


def _optional_dependencies():
    try:
        import hmmlearn
        import sklearn
        from hmmlearn.hmm import GaussianHMM
        from sklearn.decomposition import PCA
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Gaussian HMM fitting requires the 'hmm' extra: "
            "pip install 'dfc-kit[hmm]'"
        ) from error
    return hmmlearn, sklearn, GaussianHMM, PCA


def _eligible_sequences(
    dataset: FeatureSequenceDataset,
    minimum_length: int,
) -> tuple[FeatureSequence, ...]:
    return tuple(
        sequence for sequence in dataset.sequences if sequence.n_samples >= minimum_length
    )


def _validate_model_dataset(
    model: GaussianHMMStateModel,
    dataset: FeatureSequenceDataset,
) -> None:
    if dataset.feature_keys != model.feature_keys:
        raise ValueError("Gaussian HMM and dataset use different feature identities or order")
    if dataset.source_contract != model.source_contract:
        raise ValueError("Gaussian HMM and dataset use different source contracts")
    intervals_differ = (
        (dataset.sample_interval_seconds is None) != (model.sample_interval_seconds is None)
        or (
            dataset.sample_interval_seconds is not None
            and model.sample_interval_seconds is not None
            and not np.isclose(
                dataset.sample_interval_seconds,
                model.sample_interval_seconds,
                rtol=0.0,
                atol=1e-9,
            )
        )
    )
    if intervals_differ:
        raise ValueError("Gaussian HMM and dataset use different sample intervals")


def _transform_sequences(
    model: GaussianHMMStateModel,
    sequences: tuple[FeatureSequence, ...],
) -> tuple[NDArray[np.float64], ...]:
    output = []
    for sequence in sequences:
        standardized = (sequence.values - model.feature_mean) / model.feature_scale
        output.append((standardized - model.pca_mean) @ model.pca_components.T)
    return tuple(output)


def _reconstruct_estimator(model: GaussianHMMStateModel):
    _, _, GaussianHMM, _ = _optional_dependencies()
    estimator = GaussianHMM(
        n_components=model.n_states,
        covariance_type=model.covariance_type,
        n_iter=model.n_iter,
        tol=model.tol,
        random_state=model.seed,
        verbose=False,
        init_params="",
        params="",
    )
    estimator.startprob_ = np.asarray(model.start_probabilities, dtype=float)
    estimator.transmat_ = np.asarray(model.transition_matrix, dtype=float)
    estimator.means_ = np.asarray(model.reduced_means, dtype=float)
    if model.covariance_type == "diag":
        estimator.covars_ = np.diagonal(
            model.reduced_covariances,
            axis1=1,
            axis2=2,
        ).copy()
    else:
        estimator.covars_ = np.asarray(model.reduced_covariances, dtype=float)
    return estimator


def _decode(
    model: GaussianHMMStateModel,
    dataset: FeatureSequenceDataset,
) -> GaussianHMMStateResult:
    _validate_model_dataset(model, dataset)
    sequences = _eligible_sequences(dataset, model.minimum_sequence_length)
    if not sequences:
        raise ValueError("no feature sequence meets the HMM minimum sequence length")
    reduced = _transform_sequences(model, sequences)
    lengths = [len(values) for values in reduced]
    observations = np.concatenate(reduced, axis=0)
    estimator = _reconstruct_estimator(model)
    labels = np.asarray(estimator.predict(observations, lengths), dtype=np.int64)
    log_likelihood, posterior = estimator.score_samples(observations, lengths)

    label_sequences: list[StateLabelSequence] = []
    posterior_sequences: list[NDArray[np.float64]] = []
    offset = 0
    for sequence, length in zip(sequences, lengths, strict=True):
        selected = slice(offset, offset + length)
        label_sequences.append(
            StateLabelSequence(
                labels=labels[selected],
                sample_start_indices=sequence.sample_start_indices,
                sample_end_indices=sequence.sample_end_indices,
                subject=sequence.subject,
                session=sequence.session,
                acquisition_id=sequence.acquisition_id,
                segment_id=sequence.segment_id,
            )
        )
        posterior_sequences.append(_readonly(posterior[selected]))
        offset += length
    return GaussianHMMStateResult(
        assignments=StateAssignments(
            sequences=tuple(label_sequences),
            n_states=model.n_states,
            source_contract=model.source_contract,
            sample_interval_seconds=model.sample_interval_seconds,
        ),
        posterior_probabilities=tuple(posterior_sequences),
        log_likelihood=float(log_likelihood),
    )


def fit_gaussian_hmm_states(
    dataset: FeatureSequenceDataset,
    *,
    n_states: int,
    seed: int,
    n_pca_components: int | None = None,
    covariance_type: str = "diag",
    n_init: int = 1,
    n_iter: int = 200,
    tol: float = 1e-3,
    minimum_sequence_length: int = 2,
) -> GaussianHMMFitResult:
    """Fit a reproducible Gaussian HMM without bridging sequence boundaries."""
    if isinstance(n_states, (bool, np.bool_)) or not isinstance(n_states, (int, np.integer)):
        raise TypeError("n_states must be an integer")
    if n_states < 2:
        raise ValueError("n_states must be at least two")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if covariance_type not in {"diag", "full"}:
        raise ValueError("covariance_type must be 'diag' or 'full'")
    for value, label in ((n_init, "n_init"), (n_iter, "n_iter")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{label} must be an integer")
        if value < 1:
            raise ValueError(f"{label} must be at least one")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if isinstance(minimum_sequence_length, (bool, np.bool_)) or not isinstance(
        minimum_sequence_length, (int, np.integer)
    ):
        raise TypeError("minimum_sequence_length must be an integer")
    if minimum_sequence_length < 2:
        raise ValueError("minimum_sequence_length must be at least two")

    hmmlearn, sklearn, GaussianHMM, PCA = _optional_dependencies()
    sequences = _eligible_sequences(dataset, int(minimum_sequence_length))
    if not sequences:
        raise ValueError("no feature sequence meets the HMM minimum sequence length")
    observations = np.concatenate([sequence.values for sequence in sequences], axis=0)
    if len(observations) < n_states:
        raise ValueError("n_states cannot exceed the number of fitted samples")
    feature_mean = observations.mean(axis=0)
    feature_scale = observations.std(axis=0, ddof=0)
    invalid = np.flatnonzero((~np.isfinite(feature_scale)) | (feature_scale <= 1e-12))
    if len(invalid):
        raise ValueError(f"HMM feature scale is undefined for indices {invalid.tolist()}")
    standardized = (observations - feature_mean) / feature_scale
    maximum_components = min(len(standardized) - 1, standardized.shape[1])
    if n_pca_components is None:
        component_count = standardized.shape[1]
        pca_mean = np.zeros(component_count, dtype=float)
        pca_components = np.eye(component_count, dtype=float)
        explained = np.var(standardized, axis=0, ddof=1)
        explained /= explained.sum()
        reduced_observations = standardized
    else:
        if isinstance(n_pca_components, (bool, np.bool_)) or not isinstance(
            n_pca_components, (int, np.integer)
        ):
            raise TypeError("n_pca_components must be an integer or None")
        if n_pca_components < 1 or n_pca_components > maximum_components:
            raise ValueError(
                f"n_pca_components must be between 1 and {maximum_components}"
            )
        component_count = int(n_pca_components)
        pca = PCA(n_components=component_count, random_state=int(seed)).fit(standardized)
        pca_mean = np.asarray(pca.mean_, dtype=float)
        pca_components = np.asarray(pca.components_, dtype=float)
        explained = np.asarray(pca.explained_variance_ratio_, dtype=float)
        reduced_observations = pca.transform(standardized)

    lengths = [sequence.n_samples for sequence in sequences]
    initialization_seeds = tuple(int(seed) + index for index in range(int(n_init)))
    estimators = []
    likelihoods = np.empty(int(n_init), dtype=float)
    for index, initialization_seed in enumerate(initialization_seeds):
        estimator = GaussianHMM(
            n_components=int(n_states),
            covariance_type=covariance_type,
            n_iter=int(n_iter),
            tol=float(tol),
            random_state=initialization_seed,
            verbose=False,
        ).fit(reduced_observations, lengths)
        likelihoods[index] = estimator.score(reduced_observations, lengths)
        estimators.append(estimator)
    selected = int(np.argmax(likelihoods))
    best = estimators[selected]
    reduced_covariances = np.asarray(best.covars_, dtype=float)
    if reduced_covariances.shape != (n_states, component_count, component_count):
        raise RuntimeError(
            "hmmlearn returned an unexpected covariance shape: "
            f"{reduced_covariances.shape}"
        )
    standardized_means = best.means_ @ pca_components + pca_mean
    emission_means = standardized_means * feature_scale + feature_mean
    emission_covariances = np.empty(
        (n_states, observations.shape[1], observations.shape[1]),
        dtype=float,
    )
    scale_outer = np.outer(feature_scale, feature_scale)
    for state in range(n_states):
        standardized_covariance = (
            pca_components.T @ reduced_covariances[state] @ pca_components
        )
        emission_covariances[state] = standardized_covariance * scale_outer

    model = GaussianHMMStateModel(
        start_probabilities=_readonly(best.startprob_),
        transition_matrix=_readonly(best.transmat_),
        reduced_means=_readonly(best.means_),
        reduced_covariances=_readonly(reduced_covariances),
        emission_means=_readonly(emission_means),
        emission_covariances=_readonly(emission_covariances),
        feature_mean=_readonly(feature_mean),
        feature_scale=_readonly(feature_scale),
        pca_mean=_readonly(pca_mean),
        pca_components=_readonly(pca_components),
        pca_explained_variance_ratio=_readonly(explained),
        feature_keys=dataset.feature_keys,
        source_contract=dataset.source_contract,
        sample_interval_seconds=dataset.sample_interval_seconds,
        n_states=int(n_states),
        n_pca_components=component_count,
        covariance_type=covariance_type,
        seed=int(seed),
        n_init=int(n_init),
        n_iter=int(n_iter),
        tol=float(tol),
        minimum_sequence_length=int(minimum_sequence_length),
        pca_batch_size=None,
        selected_initialization=selected,
        initialization_seeds=initialization_seeds,
        initialization_log_likelihoods=_readonly(likelihoods),
        iterations=int(best.monitor_.iter),
        converged=bool(best.monitor_.converged),
        log_likelihood=float(likelihoods[selected]),
        fit_subjects=tuple(dict.fromkeys(sequence.subject for sequence in sequences)),
        fit_sample_count=len(reduced_observations),
        fit_sequence_count=len(sequences),
        omitted_short_sequence_count=len(dataset.sequences) - len(sequences),
        implementation=(
            f"hmmlearn {hmmlearn.__version__} GaussianHMM; "
            f"scikit-learn {sklearn.__version__} PCA"
        ),
    )
    return GaussianHMMFitResult(model=model, states=_decode(model, dataset))


def predict_gaussian_hmm_states(
    model: GaussianHMMStateModel,
    dataset: FeatureSequenceDataset,
    *,
    allow_fit_subjects: bool = False,
) -> GaussianHMMStateResult:
    """Decode held-out sequences, rejecting fit-subject overlap by default."""
    _validate_model_dataset(model, dataset)
    if not isinstance(allow_fit_subjects, (bool, np.bool_)):
        raise TypeError("allow_fit_subjects must be boolean")
    overlap = sorted(set(model.fit_subjects).intersection(dataset.subjects))
    if overlap and not allow_fit_subjects:
        raise ValueError(f"prediction subjects overlap Gaussian HMM fit subjects: {overlap}")
    return _decode(model, dataset)
