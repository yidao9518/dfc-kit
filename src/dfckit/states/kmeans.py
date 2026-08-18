"""Subject-recorded KMeans state estimation for feature sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import (
    FeatureKey,
    FeatureSequenceDataset,
    StateAssignments,
    StateLabelSequence,
    _readonly,
)


@dataclass(frozen=True)
class KMeansStateModel:
    centers: NDArray[np.float64]
    standardized_centers: NDArray[np.float64]
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    feature_keys: tuple[FeatureKey, ...]
    source_contract: str
    sample_interval_seconds: float | None
    n_states: int
    seed: int
    n_init: int
    max_iter: int
    algorithm: str
    standardize_features: bool
    batch_size: int | None
    reassignment_ratio: float | None
    init_sample_size: int | None
    iterations: int
    inertia: float
    fit_subjects: tuple[str, ...]
    fit_sample_count: int
    training_data_fingerprint: str | None
    implementation: str


@dataclass(frozen=True)
class KMeansFitResult:
    model: KMeansStateModel
    assignments: StateAssignments


def _assign(
    model: KMeansStateModel,
    dataset: FeatureSequenceDataset,
) -> StateAssignments:
    label_sequences: list[StateLabelSequence] = []
    for sequence in dataset.sequences:
        standardized = (sequence.values - model.feature_mean) / model.feature_scale
        squared_distance = np.square(
            standardized[:, None, :] - model.standardized_centers[None, :, :]
        ).sum(axis=2)
        labels = np.argmin(squared_distance, axis=1)
        label_sequences.append(
            StateLabelSequence(
                labels=labels,
                sample_start_indices=sequence.sample_start_indices,
                sample_end_indices=sequence.sample_end_indices,
                subject=sequence.subject,
                session=sequence.session,
                acquisition_id=sequence.acquisition_id,
                segment_id=sequence.segment_id,
            )
        )
    return StateAssignments(
        sequences=tuple(label_sequences),
        n_states=model.n_states,
        source_contract=model.source_contract,
        sample_interval_seconds=model.sample_interval_seconds,
    )


def _validate_model_dataset(model: KMeansStateModel, dataset: FeatureSequenceDataset) -> None:
    if dataset.feature_keys != model.feature_keys:
        raise ValueError("KMeans model and dataset use different feature identities or order")
    if dataset.source_contract != model.source_contract:
        raise ValueError("KMeans model and dataset use different source contracts")
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
        raise ValueError("KMeans model and dataset use different sample intervals")


def fit_kmeans_states(
    dataset: FeatureSequenceDataset,
    *,
    n_states: int,
    seed: int,
    n_init: int = 50,
    max_iter: int = 300,
    algorithm: str = "lloyd",
    standardize_features: bool = True,
    batch_size: int = 4096,
    reassignment_ratio: float = 0.01,
) -> KMeansFitResult:
    """Fit reproducible Lloyd or mini-batch KMeans states."""
    if not isinstance(n_states, (int, np.integer)) or n_states < 2:
        raise ValueError("n_states must be at least two")
    if dataset.n_samples < n_states:
        raise ValueError("n_states cannot exceed the number of fitted samples")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if not isinstance(n_init, (int, np.integer)) or n_init < 1:
        raise ValueError("n_init must be a positive integer")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if algorithm not in {"lloyd", "minibatch"}:
        raise ValueError("algorithm must be 'lloyd' or 'minibatch'")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(reassignment_ratio) or not 0.0 <= reassignment_ratio <= 1.0:
        raise ValueError("reassignment_ratio must be within [0, 1]")
    try:
        import sklearn
        from sklearn.cluster import KMeans, MiniBatchKMeans
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "KMeans state fitting requires the 'states' extra: pip install 'dfc-kit[states]'"
        ) from error

    pooled = np.concatenate([sequence.values for sequence in dataset.sequences], axis=0)
    if standardize_features:
        mean = pooled.mean(axis=0)
        scale = pooled.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
    else:
        mean = np.zeros(pooled.shape[1], dtype=float)
        scale = np.ones(pooled.shape[1], dtype=float)
    standardized = (pooled - mean) / scale
    if algorithm == "lloyd":
        estimator = KMeans(
            n_clusters=int(n_states),
            n_init=int(n_init),
            max_iter=int(max_iter),
            random_state=int(seed),
            algorithm="lloyd",
        ).fit(standardized)
    else:
        estimator = MiniBatchKMeans(
            n_clusters=int(n_states),
            n_init=int(n_init),
            max_iter=int(max_iter),
            random_state=int(seed),
            batch_size=int(batch_size),
            reassignment_ratio=float(reassignment_ratio),
        ).fit(standardized)
    fit_labels = np.asarray(estimator.labels_, dtype=np.int64)
    fit_counts = np.bincount(fit_labels, minlength=n_states)
    if np.any(fit_counts == 0):
        raise RuntimeError(
            f"KMeans returned an empty state; fitted counts={fit_counts.tolist()}"
        )
    standardized_centers = np.asarray(estimator.cluster_centers_, dtype=float)
    model = KMeansStateModel(
        centers=_readonly(standardized_centers * scale + mean),
        standardized_centers=_readonly(standardized_centers),
        feature_mean=_readonly(mean),
        feature_scale=_readonly(scale),
        feature_keys=dataset.feature_keys,
        source_contract=dataset.source_contract,
        sample_interval_seconds=dataset.sample_interval_seconds,
        n_states=int(n_states),
        seed=int(seed),
        n_init=int(n_init),
        max_iter=int(max_iter),
        algorithm=algorithm,
        standardize_features=bool(standardize_features),
        batch_size=None if algorithm == "lloyd" else int(batch_size),
        reassignment_ratio=(
            None if algorithm == "lloyd" else float(reassignment_ratio)
        ),
        init_sample_size=None,
        iterations=int(estimator.n_iter_),
        inertia=float(estimator.inertia_),
        fit_subjects=dataset.subjects,
        fit_sample_count=dataset.n_samples,
        training_data_fingerprint=None,
        implementation=(
            f"scikit-learn {sklearn.__version__} "
            f"{'KMeans algorithm=lloyd' if algorithm == 'lloyd' else 'MiniBatchKMeans'}"
        ),
    )
    assignments = _assign(model, dataset)
    return KMeansFitResult(model=model, assignments=assignments)


def predict_kmeans_states(
    model: KMeansStateModel,
    dataset: FeatureSequenceDataset,
    *,
    allow_fit_subjects: bool = False,
) -> StateAssignments:
    """Assign samples, rejecting train/test subject overlap by default."""
    _validate_model_dataset(model, dataset)
    if not isinstance(allow_fit_subjects, (bool, np.bool_)):
        raise TypeError("allow_fit_subjects must be boolean")
    overlap = sorted(set(model.fit_subjects).intersection(dataset.subjects))
    if overlap and not allow_fit_subjects:
        raise ValueError(f"prediction subjects overlap KMeans fit subjects: {overlap}")
    return _assign(model, dataset)
