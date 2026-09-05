"""Bounded-memory state-model fitting from :class:`FeatureStore` objects.

The functions in this module deliberately keep the feature rows on disk.  A
feature store is traversed in memory-mapped chunks to estimate the global
standardization parameters, initialize MiniBatchKMeans, and score the fitted
centres.  Only state labels and their original sample indices are materialized
for the returned assignments.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .._arrays import readonly_copy as _readonly
from .._validation import validated_positive_integer as _positive_int
from ..storage import FeatureStore
from ..storage._statistics import StreamingFeatureMoments
from ..storage.store import StoredFeatureChunk
from ..storage.store import StoredFeatureChunk as TransformedFeatureChunk
from .data import FeatureKey, StateAssignments, StateLabelSequence, _require_matching_feature_space
from .kmeans import KMeansFitResult, KMeansStateModel, fit_kmeans_states
from .scoring import RunKMeansScore


@dataclass(frozen=True)
class StreamingPCAModel:
    """Frozen scaler and IncrementalPCA parameters learned from a feature store."""

    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    pca_mean: NDArray[np.float64]
    pca_components: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    feature_keys: tuple[FeatureKey, ...]
    source_contract: str
    sample_interval_seconds: float | None
    n_components: int
    standardize_features: bool
    batch_size: int
    fit_subjects: tuple[str, ...]
    fit_sample_count: int
    fit_sequence_count: int
    implementation: str

    def __post_init__(self) -> None:
        arrays = {
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "pca_mean": self.pca_mean,
            "pca_components": self.pca_components,
            "explained_variance_ratio": self.explained_variance_ratio,
        }
        for name, value in arrays.items():
            array = np.asarray(value, dtype=float)
            if not np.isfinite(array).all():
                raise ValueError(f"Streaming PCA {name} contains non-finite values")
            object.__setattr__(self, name, _readonly(array))
        if self.pca_components.ndim != 2:
            raise ValueError("Streaming PCA components must be two-dimensional")
        if self.pca_components.shape[0] != self.n_components:
            raise ValueError("Streaming PCA component count does not match n_components")
        if self.pca_components.shape[1] != len(self.feature_keys):
            raise ValueError("Streaming PCA component width does not match feature keys")
        expected_features = (len(self.feature_keys),)
        if self.feature_mean.shape != expected_features or self.feature_scale.shape != expected_features:
            raise ValueError("Streaming PCA feature scaler has an invalid shape")
        if self.pca_mean.shape != expected_features:
            raise ValueError("Streaming PCA mean has an invalid shape")
        if self.explained_variance_ratio.shape != (self.n_components,):
            raise ValueError("Streaming PCA explained variance has an invalid shape")
        if self.n_components < 1 or self.fit_sample_count < self.n_components:
            raise ValueError("Streaming PCA fit metadata is invalid")
        if self.batch_size < 1 or not str(self.source_contract).strip():
            raise ValueError("Streaming PCA fit metadata is invalid")




def _nonnegative_seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if value < 0:
        raise ValueError("seed must be non-negative")
    return int(value)


def _subjects(store: FeatureStore, requested: Iterable[str] | None) -> tuple[str, ...]:
    selected = None if requested is None else {str(subject) for subject in requested}
    output = tuple(subject for subject in store.subjects if selected is None or subject in selected)
    if not output:
        raise ValueError("no stored feature sequences match the requested subjects")
    return output


def _iter_chunks(
    store: FeatureStore,
    subjects: tuple[str, ...],
    sequence_indices: Iterable[int] | None = None,
) -> Iterator[StoredFeatureChunk]:
    allowed = None if sequence_indices is None else {int(index) for index in sequence_indices}
    for chunk in store.iter_chunks(subjects=subjects, mmap=True):
        if allowed is None or chunk.sequence_index in allowed:
            yield chunk


def _feature_statistics(
    store: FeatureStore,
    subjects: tuple[str, ...],
    sequence_indices: Iterable[int] | None = None,
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    """Return count, mean, and population M2 using batch-combined Welford sums."""
    moments = StreamingFeatureMoments.empty(store.n_features)
    for chunk in _iter_chunks(store, subjects, sequence_indices):
        moments.update(chunk.values)
    if moments.count == 0:
        raise ValueError("cannot fit a state model to an empty feature store selection")
    return moments.count, moments.mean, moments.m2


def _normalization(
    store: FeatureStore,
    subjects: tuple[str, ...],
    *,
    standardize_features: bool,
    sequence_indices: Iterable[int] | None = None,
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    count, mean, m2 = _feature_statistics(store, subjects, sequence_indices)
    if not standardize_features:
        return count, np.zeros(store.n_features, dtype=np.float64), np.ones(
            store.n_features, dtype=np.float64
        )
    scale = np.sqrt(m2 / count)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return count, mean, scale


def _iter_batches(
    store: FeatureStore,
    subjects: tuple[str, ...],
    *,
    total_samples: int,
    minimum_rows: int,
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
    batch_size: int,
    sequence_indices: Iterable[int] | None = None,
    pca_mean: NDArray[np.float64] | None = None,
    pca_components: NDArray[np.float64] | None = None,
) -> Iterator[NDArray[np.float64]]:
    # sklearn requires every ``partial_fit`` call to contain at least
    # ``n_clusters`` rows.  The caller chooses a balanced partition below so
    # that a short final chunk is redistributed rather than dropped or padded.
    n_batches = min(
        max(1, (total_samples + batch_size - 1) // batch_size),
        total_samples // minimum_rows,
    )
    if n_batches < 1:
        raise ValueError("the feature store has fewer rows than the requested state count")
    base, remainder = divmod(total_samples, n_batches)
    target_sizes = [base + int(index < remainder) for index in range(n_batches)]
    if min(target_sizes) < minimum_rows:
        raise RuntimeError("internal batch partition produced an undersized KMeans batch")

    pending: list[NDArray[np.float64]] = []
    pending_count = 0
    target_index = 0
    target_remaining = target_sizes[0]
    for chunk in _iter_chunks(store, subjects, sequence_indices):
        values = np.asarray(chunk.values, dtype=np.float64)
        start = 0
        while start < len(values):
            take = min(target_remaining - pending_count, len(values) - start)
            pending.append((values[start : start + take] - mean) / scale)
            pending_count += take
            start += take
            if pending_count == target_remaining:
                batch = np.concatenate(pending, axis=0)
                if pca_components is not None:
                    assert pca_mean is not None
                    batch = (batch - pca_mean) @ pca_components.T
                yield batch
                pending = []
                pending_count = 0
                target_index += 1
                if target_index < len(target_sizes):
                    target_remaining = target_sizes[target_index]


def _uniform_sample(
    store: FeatureStore,
    subjects: tuple[str, ...],
    *,
    total_samples: int,
    sample_size: int,
    seed: int,
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
    pca_mean: NDArray[np.float64] | None = None,
    pca_components: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Draw deterministic, without-replacement rows from the global row axis."""
    size = min(_positive_int(sample_size, "init_sample_size"), total_samples)
    chosen = np.sort(
        np.random.default_rng(seed).choice(total_samples, size=size, replace=False)
    )
    pieces: list[NDArray[np.float64]] = []
    offset = 0
    for chunk in _iter_chunks(store, subjects):
        stop = offset + len(chunk.values)
        left = int(np.searchsorted(chosen, offset, side="left"))
        right = int(np.searchsorted(chosen, stop, side="left"))
        if right > left:
            local = chosen[left:right] - offset
            values = np.asarray(chunk.values[local], dtype=np.float64)
            pieces.append((values - mean) / scale)
        offset = stop
    if offset != total_samples or not pieces:
        raise RuntimeError("feature-store global row index could not be reconstructed")
    sample = np.concatenate(pieces, axis=0)
    if pca_components is not None:
        assert pca_mean is not None
        sample = (sample - pca_mean) @ pca_components.T
    return sample


def _labels_and_inertia(
    store: FeatureStore,
    subjects: tuple[str, ...],
    *,
    centers: NDArray[np.float64],
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
    collect_labels: bool,
    pca_mean: NDArray[np.float64] | None = None,
    pca_components: NDArray[np.float64] | None = None,
) -> tuple[float, dict[int, tuple[list[NDArray[np.int64]], list[NDArray[np.int64]], list[NDArray[np.int64]]]]]:
    center_norm = np.einsum("ij,ij->i", centers, centers)
    inertia = 0.0
    labels_by_sequence: dict[
        int, tuple[list[NDArray[np.int64]], list[NDArray[np.int64]], list[NDArray[np.int64]]]
    ] = {}
    for chunk in _iter_chunks(store, subjects):
        values = (np.asarray(chunk.values, dtype=np.float64) - mean) / scale
        if pca_components is not None:
            assert pca_mean is not None
            values = (values - pca_mean) @ pca_components.T
        distances = (
            np.einsum("ij,ij->i", values, values)[:, None]
            + center_norm[None, :]
            - 2.0 * values @ centers.T
        )
        labels = np.argmin(distances, axis=1).astype(np.int64, copy=False)
        inertia += float(np.maximum(0.0, distances[np.arange(len(labels)), labels]).sum())
        if collect_labels:
            bucket = labels_by_sequence.setdefault(chunk.sequence_index, ([], [], []))
            bucket[0].append(np.array(labels, copy=True))
            bucket[1].append(np.array(chunk.sample_start_indices, dtype=np.int64, copy=True))
            bucket[2].append(np.array(chunk.sample_end_indices, dtype=np.int64, copy=True))
    if not labels_by_sequence and collect_labels:
        raise ValueError("no stored feature sequences match the requested subjects")
    return inertia, labels_by_sequence


def _assignments_from_store(
    store: FeatureStore,
    subjects: tuple[str, ...],
    *,
    model: KMeansStateModel,
) -> StateAssignments:
    _, labels_by_sequence = _labels_and_inertia(
        store,
        subjects,
        centers=model.clustering_centers,
        mean=model.feature_mean,
        scale=model.feature_scale,
        collect_labels=True,
        pca_mean=None if model.n_pca_components is None else model.pca_mean,
        pca_components=(
            None if model.n_pca_components is None else model.pca_components
        ),
    )
    sequences: list[StateLabelSequence] = []
    selected = set(subjects)
    for sequence_index, identity in enumerate(store.sequence_identities):
        if identity[0] not in selected:
            continue
        try:
            label_parts, start_parts, end_parts = labels_by_sequence[sequence_index]
        except KeyError as error:
            raise RuntimeError(f"missing labels for feature sequence {identity}") from error
        sequences.append(
            StateLabelSequence(
                labels=np.concatenate(label_parts),
                sample_start_indices=np.concatenate(start_parts),
                sample_end_indices=np.concatenate(end_parts),
                subject=identity[0],
                session=identity[1],
                acquisition_id=identity[2],
                segment_id=identity[3],
            )
        )
    return StateAssignments(
        sequences=tuple(sequences),
        n_states=model.n_states,
        source_contract=model.source_contract,
        sample_interval_seconds=model.sample_interval_seconds,
    )


def _selected_sequence_indices(
    store: FeatureStore,
    subjects: tuple[str, ...],
    *,
    minimum_length: int,
) -> tuple[int, ...]:
    selected = set(subjects)
    output = tuple(
        index
        for index, (identity, count) in enumerate(store.sequence_sample_counts)
        if identity[0] in selected and count >= minimum_length
    )
    if not output:
        raise ValueError("no stored feature sequence meets the requested minimum length")
    return output


def fit_incremental_pca_store(
    store: FeatureStore,
    *,
    n_components: int,
    batch_size: int = 4096,
    standardize_features: bool = True,
    subjects: Iterable[str] | None = None,
    minimum_sequence_length: int = 1,
) -> StreamingPCAModel:
    """Fit feature scaling and IncrementalPCA from memory-mapped store chunks."""
    n_components = _positive_int(n_components, "n_components")
    batch_size = _positive_int(batch_size, "batch_size")
    minimum_sequence_length = _positive_int(
        minimum_sequence_length, "minimum_sequence_length"
    )
    if not isinstance(standardize_features, (bool, np.bool_)):
        raise TypeError("standardize_features must be boolean")
    selected_subjects = _subjects(store, subjects)
    sequence_indices = _selected_sequence_indices(
        store,
        selected_subjects,
        minimum_length=minimum_sequence_length,
    )
    try:
        import sklearn
        from sklearn.decomposition import IncrementalPCA
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "streaming PCA fitting requires the 'states' extra: "
            "pip install 'dfc-kit[states]'"
        ) from error

    n_samples, observed_mean, m2 = _feature_statistics(
        store,
        selected_subjects,
        sequence_indices,
    )
    if n_samples < 2:
        raise ValueError("Streaming PCA requires at least two fitted samples")
    if standardize_features:
        mean = observed_mean
        scale = np.sqrt(m2 / n_samples)
    else:
        mean = np.zeros(store.n_features, dtype=np.float64)
        scale = np.ones(store.n_features, dtype=np.float64)
    maximum_components = min(n_samples, store.n_features)
    if n_components > maximum_components:
        raise ValueError(f"n_components must be between 1 and {maximum_components}")
    if standardize_features:
        invalid = np.flatnonzero((~np.isfinite(scale)) | (scale <= 1e-12))
        if len(invalid):
            raise ValueError(
                f"Streaming PCA feature scale is undefined for indices {invalid.tolist()}"
            )

    estimator = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for batch in _iter_batches(
        store,
        selected_subjects,
        total_samples=n_samples,
        minimum_rows=n_components,
        mean=mean,
        scale=scale,
        batch_size=batch_size,
        sequence_indices=sequence_indices,
    ):
        estimator.partial_fit(batch)
    fit_subjects = tuple(
        dict.fromkeys(store.sequence_identities[index][0] for index in sequence_indices)
    )
    return StreamingPCAModel(
        feature_mean=mean,
        feature_scale=scale,
        pca_mean=np.asarray(estimator.mean_, dtype=float),
        pca_components=np.asarray(estimator.components_, dtype=float),
        explained_variance_ratio=np.asarray(estimator.explained_variance_ratio_, dtype=float),
        feature_keys=store.feature_keys,
        source_contract=store.source_contract,
        sample_interval_seconds=store.sample_interval_seconds,
        n_components=n_components,
        standardize_features=bool(standardize_features),
        batch_size=batch_size,
        fit_subjects=fit_subjects,
        fit_sample_count=n_samples,
        fit_sequence_count=len(sequence_indices),
        implementation=f"scikit-learn {sklearn.__version__} IncrementalPCA",
    )


def iter_pca_store_chunks(
    model: StreamingPCAModel,
    store: FeatureStore,
    *,
    subjects: Iterable[str] | None = None,
    allow_fit_subjects: bool = False,
    sequence_indices: Iterable[int] | None = None,
) -> Iterator[TransformedFeatureChunk]:
    """Yield PCA-reduced chunks while retaining sequence and sample metadata."""
    _require_matching_feature_space(model, store, label="Streaming PCA")
    if not isinstance(allow_fit_subjects, (bool, np.bool_)):
        raise TypeError("allow_fit_subjects must be boolean")
    selected_subjects = _subjects(store, subjects)
    overlap = sorted(set(model.fit_subjects).intersection(selected_subjects))
    if overlap and not allow_fit_subjects:
        raise ValueError(f"prediction subjects overlap Streaming PCA fit subjects: {overlap}")
    for chunk in _iter_chunks(store, selected_subjects, sequence_indices):
        standardized = (
            np.asarray(chunk.values, dtype=np.float64) - model.feature_mean
        ) / model.feature_scale
        reduced = (standardized - model.pca_mean) @ model.pca_components.T
        yield replace(chunk, values=reduced)


def fit_minibatch_kmeans_store(
    store: FeatureStore,
    *,
    n_states: int,
    seed: int,
    n_init: int = 10,
    max_iter: int = 10,
    batch_size: int = 4096,
    standardize_features: bool = True,
    reassignment_ratio: float = 0.01,
    init_sample_size: int | None = None,
    n_pca_components: int | None = None,
    pca_batch_size: int = 4096,
    subjects: Iterable[str] | None = None,
    convergence_tol: float = 1e-4,
    convergence_patience: int = 3,
    minimum_passes: int = 2,
) -> KMeansFitResult:
    """Fit MiniBatchKMeans while keeping feature rows in a ``FeatureStore``.

    ``max_iter`` is the maximum number of complete passes through the selected
    store; each pass calls ``partial_fit`` for every bounded batch. Fitting
    stops earlier when relative centre drift remains below ``convergence_tol``
    for ``convergence_patience`` consecutive passes after at least
    ``minimum_passes``. Set ``convergence_tol=0`` to require all passes.
    K-means++ centres are initialized from one deterministic uniform global-row
    sample, and the best of ``n_init`` independent initializations is selected
    by a full-store inertia pass. The returned assignments contain labels and
    sample indices, not feature rows.
    """
    if not isinstance(n_states, (int, np.integer)) or n_states < 2:
        raise ValueError("n_states must be at least two")
    n_states = int(n_states)
    fit_subjects = _subjects(store, subjects)
    seed = _nonnegative_seed(seed)
    n_init = _positive_int(n_init, "n_init")
    max_iter = _positive_int(max_iter, "max_iter")
    batch_size = _positive_int(batch_size, "batch_size")
    if not isinstance(standardize_features, (bool, np.bool_)):
        raise TypeError("standardize_features must be boolean")
    if not np.isfinite(reassignment_ratio) or not 0.0 <= reassignment_ratio <= 1.0:
        raise ValueError("reassignment_ratio must be within [0, 1]")
    if init_sample_size is not None:
        init_sample_size = _positive_int(init_sample_size, "init_sample_size")
    if n_pca_components is not None:
        n_pca_components = _positive_int(n_pca_components, "n_pca_components")
        pca_batch_size = _positive_int(pca_batch_size, "pca_batch_size")
    if isinstance(convergence_tol, (bool, np.bool_)):
        raise TypeError("convergence_tol must be a real number")
    convergence_tol = float(convergence_tol)
    if not np.isfinite(convergence_tol) or convergence_tol < 0.0:
        raise ValueError("convergence_tol must be finite and non-negative")
    convergence_patience = _positive_int(
        convergence_patience, "convergence_patience"
    )
    minimum_passes = _positive_int(minimum_passes, "minimum_passes")

    try:
        import sklearn
        from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "streaming KMeans fitting requires the 'states' extra: "
            "pip install 'dfc-kit[states]'"
        ) from error

    pca_model = None
    if n_pca_components is not None:
        pca_model = fit_incremental_pca_store(
            store,
            n_components=n_pca_components,
            batch_size=pca_batch_size,
            standardize_features=bool(standardize_features),
            subjects=fit_subjects,
        )
        n_samples = pca_model.fit_sample_count
        mean = pca_model.feature_mean
        scale = pca_model.feature_scale
    else:
        n_samples, mean, scale = _normalization(
            store,
            fit_subjects,
            standardize_features=bool(standardize_features),
        )
    if n_samples < n_states:
        raise ValueError("n_states cannot exceed the number of fitted samples")
    if init_sample_size is None:
        init_sample_size = max(1000, 20 * n_states)
    init_sample_size = min(init_sample_size, n_samples)
    initialization_sample = _uniform_sample(
        store,
        fit_subjects,
        total_samples=n_samples,
        sample_size=init_sample_size,
        seed=seed,
        mean=mean,
        scale=scale,
        pca_mean=None if pca_model is None else pca_model.pca_mean,
        pca_components=None if pca_model is None else pca_model.pca_components,
    )

    best_centers: NDArray[np.float64] | None = None
    best_inertia = np.inf
    best_iterations = 0
    best_passes_completed = 0
    best_converged = False
    initialization_passes: list[int] = []
    for initialization in range(n_init):
        initialization_seed = int(
            np.random.SeedSequence([seed, initialization]).generate_state(1)[0]
        )
        initial_centers, _ = kmeans_plusplus(
            initialization_sample,
            n_clusters=n_states,
            random_state=initialization_seed,
        )
        estimator = MiniBatchKMeans(
            n_clusters=n_states,
            init=np.asarray(initial_centers, dtype=np.float64),
            n_init=1,
            max_iter=1,
            batch_size=batch_size,
            random_state=initialization_seed,
            reassignment_ratio=float(reassignment_ratio),
            compute_labels=False,
        )
        # Give the online estimator one globally representative batch before
        # traversing subject-ordered store chunks.  This establishes cluster
        # counts for every k-means++ centre and reduces first-subject ordering
        # bias without loading the complete feature matrix.
        estimator.partial_fit(initialization_sample)
        n_batches = 1
        previous_centers = np.asarray(estimator.cluster_centers_, dtype=np.float64).copy()
        stable_passes = 0
        converged = False
        passes_completed = 0
        for pass_index in range(1, max_iter + 1):
            for batch in _iter_batches(
                store,
                fit_subjects,
                total_samples=n_samples,
                minimum_rows=n_states,
                mean=mean,
                scale=scale,
                batch_size=batch_size,
                pca_mean=None if pca_model is None else pca_model.pca_mean,
                pca_components=(
                    None if pca_model is None else pca_model.pca_components
                ),
            ):
                estimator.partial_fit(batch)
                n_batches += 1
            passes_completed = pass_index
            current_centers = np.asarray(estimator.cluster_centers_, dtype=np.float64)
            relative_drift = np.linalg.norm(current_centers - previous_centers) / max(
                np.linalg.norm(previous_centers), np.finfo(np.float64).eps
            )
            stable_passes = stable_passes + 1 if relative_drift <= convergence_tol else 0
            previous_centers = current_centers.copy()
            if (
                convergence_tol > 0.0
                and passes_completed >= minimum_passes
                and stable_passes >= convergence_patience
            ):
                converged = True
                break
        initialization_passes.append(passes_completed)
        centers = np.asarray(estimator.cluster_centers_, dtype=np.float64)
        inertia, _ = _labels_and_inertia(
            store,
            fit_subjects,
            centers=centers,
            mean=mean,
            scale=scale,
            collect_labels=False,
            pca_mean=None if pca_model is None else pca_model.pca_mean,
            pca_components=(
                None if pca_model is None else pca_model.pca_components
            ),
        )
        if inertia < best_inertia:
            best_centers = centers.copy()
            best_inertia = inertia
            best_iterations = n_batches
            best_passes_completed = passes_completed
            best_converged = converged

    assert best_centers is not None
    if pca_model is None:
        clustering_centers = best_centers
        standardized_centers = best_centers
        pca_mean = np.zeros(store.n_features, dtype=float)
        pca_components = np.empty((0, store.n_features), dtype=float)
        pca_explained = np.empty(0, dtype=float)
    else:
        clustering_centers = best_centers
        pca_mean = pca_model.pca_mean
        pca_components = pca_model.pca_components
        pca_explained = pca_model.explained_variance_ratio
        standardized_centers = clustering_centers @ pca_components + pca_mean
    model = KMeansStateModel(
        centers=_readonly(standardized_centers * scale + mean),
        standardized_centers=_readonly(standardized_centers),
        clustering_centers=_readonly(clustering_centers),
        feature_mean=_readonly(mean),
        feature_scale=_readonly(scale),
        pca_mean=_readonly(pca_mean),
        pca_components=_readonly(pca_components),
        pca_explained_variance_ratio=_readonly(pca_explained),
        feature_keys=store.feature_keys,
        source_contract=store.source_contract,
        sample_interval_seconds=store.sample_interval_seconds,
        n_states=n_states,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        algorithm="minibatch",
        standardize_features=bool(standardize_features),
        n_pca_components=n_pca_components,
        pca_batch_size=None if pca_model is None else pca_batch_size,
        batch_size=batch_size,
        reassignment_ratio=float(reassignment_ratio),
        init_sample_size=init_sample_size,
        iterations=best_iterations,
        inertia=float(best_inertia),
        fit_subjects=fit_subjects,
        fit_sample_count=n_samples,
        implementation=(
            f"scikit-learn {sklearn.__version__} MiniBatchKMeans; "
            f"out-of-core partial_fit max_passes={max_iter}, "
            f"selected_passes={best_passes_completed}, convergence_tol={convergence_tol:g}, "
            f"patience={convergence_patience}, minimum_passes={minimum_passes}; "
            f"kmeans++ sample={init_sample_size}; "
            f"{'original standardized feature space' if pca_model is None else f'IncrementalPCA n_components={n_pca_components}'}"
        ),
    )
    assignments = _assignments_from_store(store, fit_subjects, model=model)
    counts = np.zeros(n_states, dtype=np.int64)
    for sequence in assignments.sequences:
        counts += np.bincount(sequence.labels, minlength=n_states)
    if np.any(counts == 0):
        raise RuntimeError(f"KMeans returned an empty state; fitted counts={counts.tolist()}")
    return KMeansFitResult(
        model=model,
        assignments=assignments,
        converged=None if convergence_tol == 0.0 else best_converged,
        passes_completed=best_passes_completed,
        initialization_passes=tuple(initialization_passes),
    )


def fit_kmeans_store_materialized(
    store: FeatureStore,
    *,
    n_states: int,
    seed: int,
    n_init: int = 20,
    max_iter: int = 300,
    algorithm: str = "minibatch",
    batch_size: int = 4096,
    standardize_features: bool = True,
    reassignment_ratio: float = 0.01,
    n_pca_components: int | None = None,
    subjects: Iterable[str] | None = None,
) -> KMeansFitResult:
    """Fit the exact in-memory sklearn path after reading a selected store cohort.

    This mode is intended for moderate stores whose historical analysis used one
    complete ``KMeans.fit`` or ``MiniBatchKMeans.fit`` call. It preserves the
    store's identities while making the higher memory cost
    explicit to callers.
    """
    fit_subjects = _subjects(store, subjects)
    dataset = store.read_dataset(subjects=fit_subjects)
    fit = fit_kmeans_states(
        dataset,
        n_states=n_states,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        algorithm=algorithm,
        standardize_features=standardize_features,
        batch_size=batch_size,
        reassignment_ratio=reassignment_ratio,
        n_pca_components=n_pca_components,
    )
    model = replace(
        fit.model,
        implementation=f"{fit.model.implementation}; materialized FeatureStore fit",
    )
    return KMeansFitResult(model=model, assignments=fit.assignments)


def predict_kmeans_store(
    model: KMeansStateModel,
    store: FeatureStore,
    *,
    subjects: Iterable[str] | None = None,
    allow_fit_subjects: bool = False,
) -> StateAssignments:
    """Assign a store's rows without materializing its feature matrix."""
    _require_matching_feature_space(model, store, label="KMeans")
    if not isinstance(allow_fit_subjects, (bool, np.bool_)):
        raise TypeError("allow_fit_subjects must be boolean")
    prediction_subjects = _subjects(store, subjects)
    overlap = sorted(set(model.fit_subjects).intersection(prediction_subjects))
    if overlap and not allow_fit_subjects:
        raise ValueError(f"prediction subjects overlap KMeans fit subjects: {overlap}")
    return _assignments_from_store(store, prediction_subjects, model=model)


def score_kmeans_store(
    model: KMeansStateModel,
    store: FeatureStore,
    *,
    subjects: Iterable[str] | None = None,
    allow_fit_subjects: bool = False,
) -> tuple[RunKMeansScore, ...]:
    """Score frozen KMeans centres on acquisitions in a feature store.

    Each row contributes its squared Euclidean distance to the nearest centre
    after applying the scaler fitted with the model. Scores are grouped by
    subject, session, and acquisition while censor-bounded sequences remain
    separately counted.
    """
    _require_matching_feature_space(model, store, label="KMeans")
    if not isinstance(allow_fit_subjects, (bool, np.bool_)):
        raise TypeError("allow_fit_subjects must be boolean")
    selected_subjects = _subjects(store, subjects)
    overlap = sorted(set(model.fit_subjects).intersection(selected_subjects))
    if overlap and not allow_fit_subjects:
        raise ValueError(f"scoring subjects overlap KMeans fit subjects: {overlap}")

    center_norm = np.einsum(
        "ij,ij->i",
        model.clustering_centers,
        model.clustering_centers,
    )
    totals: dict[tuple[str, str | None, str | None], float] = {}
    sample_counts: dict[tuple[str, str | None, str | None], int] = {}
    sequence_indices: dict[
        tuple[str, str | None, str | None], set[int]
    ] = {}
    for chunk in _iter_chunks(store, selected_subjects):
        values = (
            np.asarray(chunk.values, dtype=np.float64) - model.feature_mean
        ) / model.feature_scale
        if model.n_pca_components is not None:
            values = (values - model.pca_mean) @ model.pca_components.T
        distances = (
            np.einsum("ij,ij->i", values, values)[:, None]
            + center_norm[None, :]
            - 2.0 * values @ model.clustering_centers.T
        )
        nearest = np.maximum(0.0, np.min(distances, axis=1))
        identity = (chunk.subject, chunk.session, chunk.acquisition_id)
        totals[identity] = totals.get(identity, 0.0) + float(nearest.sum())
        sample_counts[identity] = sample_counts.get(identity, 0) + len(values)
        sequence_indices.setdefault(identity, set()).add(chunk.sequence_index)

    return tuple(
        RunKMeansScore(
            subject=subject,
            session=session,
            acquisition_id=acquisition_id,
            n_samples=sample_counts[identity],
            n_sequences=len(sequence_indices[identity]),
            total_squared_distance=total,
        )
        for identity, total in totals.items()
        for subject, session, acquisition_id in (identity,)
    )


__all__ = [
    "StreamingPCAModel",
    "TransformedFeatureChunk",
    "fit_incremental_pca_store",
    "fit_kmeans_store_materialized",
    "fit_minibatch_kmeans_store",
    "iter_pca_store_chunks",
    "predict_kmeans_store",
    "score_kmeans_store",
]
