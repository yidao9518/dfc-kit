"""Subject-recorded KMeans state estimation for feature sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._arrays import readonly_copy as _readonly
from .data import (
    FeatureKey,
    FeatureSequenceDataset,
    StateAssignments,
    StateLabelSequence,
    _require_matching_feature_space,
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
    implementation: str
    clustering_centers: NDArray[np.float64] | None = None
    pca_mean: NDArray[np.float64] | None = None
    pca_components: NDArray[np.float64] | None = None
    pca_explained_variance_ratio: NDArray[np.float64] | None = None
    n_pca_components: int | None = None
    pca_batch_size: int | None = None

    def __post_init__(self) -> None:
        feature_count = len(self.feature_keys)
        centers = np.asarray(self.centers, dtype=float)
        standardized_centers = np.asarray(self.standardized_centers, dtype=float)
        clustering = (
            self.standardized_centers
            if self.clustering_centers is None
            else self.clustering_centers
        )
        pca_mean = (
            np.zeros(feature_count, dtype=float)
            if self.pca_mean is None
            else self.pca_mean
        )
        pca_components = (
            np.empty((0, feature_count), dtype=float)
            if self.pca_components is None
            else self.pca_components
        )
        explained = (
            np.empty(0, dtype=float)
            if self.pca_explained_variance_ratio is None
            else self.pca_explained_variance_ratio
        )
        arrays = (centers, standardized_centers, clustering, pca_mean, pca_components, explained)
        if any(not np.isfinite(np.asarray(values, dtype=float)).all() for values in arrays):
            raise ValueError("KMeans model arrays must be finite")
        if centers.shape != (self.n_states, feature_count) or standardized_centers.shape != (
            self.n_states,
            feature_count,
        ):
            raise ValueError("KMeans centers have an invalid shape")
        if np.asarray(pca_mean).shape != (feature_count,):
            raise ValueError("KMeans PCA mean has an invalid shape")
        if self.n_pca_components is None:
            if len(pca_components) or len(explained) or self.pca_batch_size is not None:
                raise ValueError("non-PCA KMeans models cannot contain PCA parameters")
            if np.asarray(clustering).shape != (self.n_states, feature_count):
                raise ValueError("KMeans clustering centers have an invalid shape")
        else:
            if self.n_pca_components < 1:
                raise ValueError("n_pca_components must be positive")
            if pca_components.shape != (self.n_pca_components, feature_count):
                raise ValueError("KMeans PCA components have an invalid shape")
            if explained.shape != (self.n_pca_components,):
                raise ValueError("KMeans PCA explained variance has an invalid shape")
            if np.asarray(clustering).shape != (
                self.n_states,
                self.n_pca_components,
            ):
                raise ValueError("KMeans clustering centers have an invalid shape")
            if np.any(np.asarray(explained) < 0.0) or np.asarray(explained).sum() > 1.0 + 1e-7:
                raise ValueError("KMeans PCA explained variance ratios are invalid")
            if self.pca_batch_size is not None and self.pca_batch_size < 1:
                raise ValueError("pca_batch_size must be positive")
        object.__setattr__(self, "centers", _readonly(centers))
        object.__setattr__(self, "standardized_centers", _readonly(standardized_centers))
        object.__setattr__(self, "clustering_centers", _readonly(clustering))
        object.__setattr__(self, "pca_mean", _readonly(pca_mean))
        object.__setattr__(self, "pca_components", _readonly(pca_components))
        object.__setattr__(self, "pca_explained_variance_ratio", _readonly(explained))


@dataclass(frozen=True)
class KMeansFitResult:
    model: KMeansStateModel
    assignments: StateAssignments
    converged: bool | None = None
    passes_completed: int | None = None
    initialization_passes: tuple[int, ...] = ()


def _transform_features(
    model: KMeansStateModel,
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    standardized = (values - model.feature_mean) / model.feature_scale
    if model.n_pca_components is None:
        return standardized
    return (standardized - model.pca_mean) @ model.pca_components.T


def _assign(
    model: KMeansStateModel,
    dataset: FeatureSequenceDataset,
) -> StateAssignments:
    label_sequences: list[StateLabelSequence] = []
    for sequence in dataset.sequences:
        transformed = _transform_features(model, sequence.values)
        squared_distance = np.square(
            transformed[:, None, :] - model.clustering_centers[None, :, :]
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


def fit_kmeans_states(
    dataset: FeatureSequenceDataset,
    *,
    n_states: int,
    seed: int,
    n_init: int = 50,
    max_iter: int = 300,
    algorithm: str = "lloyd",
    standardize_features: bool = True,
    n_pca_components: int | None = None,
    batch_size: int = 4096,
    reassignment_ratio: float = 0.01,
) -> KMeansFitResult:
    """Fit reproducible KMeans states in raw or training-fitted PCA space.

    ``n_pca_components=None`` keeps the original standardized feature space.
    Otherwise PCA is fitted on the pooled input rows before KMeans; the frozen
    transform is applied again by prediction and the resulting centres are
    reconstructed in the original feature space for interpretation.
    """
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
        from sklearn.decomposition import PCA
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
    if n_pca_components is None:
        pca_mean = np.zeros(standardized.shape[1], dtype=float)
        pca_components = np.empty((0, standardized.shape[1]), dtype=float)
        pca_explained = np.empty(0, dtype=float)
        clustering_values = standardized
    else:
        if isinstance(n_pca_components, (bool, np.bool_)) or not isinstance(
            n_pca_components, (int, np.integer)
        ):
            raise TypeError("n_pca_components must be an integer or None")
        maximum_components = min(standardized.shape)
        if n_pca_components < 1 or n_pca_components > maximum_components:
            raise ValueError(
                f"n_pca_components must be between 1 and {maximum_components}"
            )
        reducer = PCA(
            n_components=int(n_pca_components),
            random_state=int(seed),
        ).fit(standardized)
        pca_mean = np.asarray(reducer.mean_, dtype=float)
        pca_components = np.asarray(reducer.components_, dtype=float)
        pca_explained = np.asarray(reducer.explained_variance_ratio_, dtype=float)
        clustering_values = reducer.transform(standardized)
    if algorithm == "lloyd":
        estimator = KMeans(
            n_clusters=int(n_states),
            n_init=int(n_init),
            max_iter=int(max_iter),
            random_state=int(seed),
            algorithm="lloyd",
        ).fit(clustering_values)
    else:
        estimator = MiniBatchKMeans(
            n_clusters=int(n_states),
            n_init=int(n_init),
            max_iter=int(max_iter),
            random_state=int(seed),
            batch_size=int(batch_size),
            reassignment_ratio=float(reassignment_ratio),
        ).fit(clustering_values)
    fit_labels = np.asarray(estimator.labels_, dtype=np.int64)
    fit_counts = np.bincount(fit_labels, minlength=n_states)
    if np.any(fit_counts == 0):
        raise RuntimeError(
            f"KMeans returned an empty state; fitted counts={fit_counts.tolist()}"
        )
    clustering_centers = np.asarray(estimator.cluster_centers_, dtype=float)
    standardized_centers = (
        clustering_centers
        if n_pca_components is None
        else clustering_centers @ pca_components + pca_mean
    )
    model = KMeansStateModel(
        centers=_readonly(standardized_centers * scale + mean),
        standardized_centers=_readonly(standardized_centers),
        clustering_centers=_readonly(clustering_centers),
        feature_mean=_readonly(mean),
        feature_scale=_readonly(scale),
        pca_mean=_readonly(pca_mean),
        pca_components=_readonly(pca_components),
        pca_explained_variance_ratio=_readonly(pca_explained),
        feature_keys=dataset.feature_keys,
        source_contract=dataset.source_contract,
        sample_interval_seconds=dataset.sample_interval_seconds,
        n_states=int(n_states),
        seed=int(seed),
        n_init=int(n_init),
        max_iter=int(max_iter),
        algorithm=algorithm,
        standardize_features=bool(standardize_features),
        n_pca_components=(
            None if n_pca_components is None else int(n_pca_components)
        ),
        pca_batch_size=None,
        batch_size=None if algorithm == "lloyd" else int(batch_size),
        reassignment_ratio=(
            None if algorithm == "lloyd" else float(reassignment_ratio)
        ),
        init_sample_size=None,
        iterations=int(estimator.n_iter_),
        inertia=float(estimator.inertia_),
        fit_subjects=dataset.subjects,
        fit_sample_count=dataset.n_samples,
        implementation=(
            f"scikit-learn {sklearn.__version__} "
            f"{'KMeans algorithm=lloyd' if algorithm == 'lloyd' else 'MiniBatchKMeans'}; "
            f"{'original standardized feature space' if n_pca_components is None else f'PCA n_components={int(n_pca_components)}'}"
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
    _require_matching_feature_space(model, dataset, label="KMeans")
    if not isinstance(allow_fit_subjects, (bool, np.bool_)):
        raise TypeError("allow_fit_subjects must be boolean")
    overlap = sorted(set(model.fit_subjects).intersection(dataset.subjects))
    if overlap and not allow_fit_subjects:
        raise ValueError(f"prediction subjects overlap KMeans fit subjects: {overlap}")
    return _assign(model, dataset)
