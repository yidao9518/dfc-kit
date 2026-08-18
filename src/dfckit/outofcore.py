"""Bounded-memory state-model fitting from :class:`FeatureStore` objects.

The functions in this module deliberately keep the feature rows on disk.  A
feature store is traversed in memory-mapped chunks to estimate the global
standardization parameters, initialize MiniBatchKMeans, and score the fitted
centres.  Only state labels and their original sample indices are materialized
for the returned assignments.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .states.data import FeatureKey, StateAssignments, StateLabelSequence, _readonly
from .states.kmeans import KMeansFitResult, KMeansStateModel
from .states.scoring import RunKMeansScore
from .storage import FeatureStore, StoredFeatureChunk


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


@dataclass(frozen=True)
class TransformedFeatureChunk:
    """One in-memory transformed chunk retaining its source sequence metadata."""

    values: NDArray[np.float64]
    sample_start_indices: NDArray[np.int64]
    sample_end_indices: NDArray[np.int64]
    subject: str
    session: str | None
    acquisition_id: str | None
    segment_id: int
    sequence_index: int
    chunk_id: int
    start_in_sequence: int
    stop_in_sequence: int

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        starts = np.asarray(self.sample_start_indices, dtype=np.int64)
        ends = np.asarray(self.sample_end_indices, dtype=np.int64)
        if values.ndim != 2 or not len(values):
            raise ValueError("transformed feature values must be a non-empty matrix")
        if starts.shape != (len(values),) or ends.shape != (len(values),):
            raise ValueError("transformed feature sample indices do not align with rows")
        if not np.isfinite(values).all() or np.any(ends < starts):
            raise ValueError("transformed feature chunk contains invalid values or indices")
        object.__setattr__(self, "values", _readonly(values))
        object.__setattr__(self, "sample_start_indices", _readonly(starts))
        object.__setattr__(self, "sample_end_indices", _readonly(ends))


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


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
    count = 0
    mean = np.zeros(store.n_features, dtype=np.float64)
    m2 = np.zeros(store.n_features, dtype=np.float64)
    for chunk in _iter_chunks(store, subjects, sequence_indices):
        values = np.asarray(chunk.values, dtype=np.float64)
        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.einsum("ij,ij->j", centered, centered)
        combined = count + batch_count
        delta = batch_mean - mean
        weight = batch_count / combined
        m2 += batch_m2 + np.square(delta) * (count * batch_count / combined)
        mean += delta * weight
        count = combined
    if count == 0:
        raise ValueError("cannot fit a state model to an empty feature store selection")
    return count, mean, m2


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
                yield np.concatenate(pending, axis=0)
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
    return np.concatenate(pieces, axis=0)


def _labels_and_inertia(
    store: FeatureStore,
    subjects: tuple[str, ...],
    *,
    centers: NDArray[np.float64],
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
    collect_labels: bool,
) -> tuple[float, dict[int, tuple[list[NDArray[np.int64]], list[NDArray[np.int64]], list[NDArray[np.int64]]]]]:
    center_norm = np.einsum("ij,ij->i", centers, centers)
    inertia = 0.0
    labels_by_sequence: dict[
        int, tuple[list[NDArray[np.int64]], list[NDArray[np.int64]], list[NDArray[np.int64]]]
    ] = {}
    for chunk in _iter_chunks(store, subjects):
        values = (np.asarray(chunk.values, dtype=np.float64) - mean) / scale
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
        centers=model.standardized_centers,
        mean=model.feature_mean,
        scale=model.feature_scale,
        collect_labels=True,
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


def _validate_store_model(model: KMeansStateModel, store: FeatureStore) -> None:
    if store.feature_keys != model.feature_keys:
        raise ValueError("KMeans model and feature store use different feature identities or order")
    if store.source_contract != model.source_contract:
        raise ValueError("KMeans model and feature store use different source contracts")
    stored = store.sample_interval_seconds
    if (stored is None) != (model.sample_interval_seconds is None) or (
        stored is not None
        and model.sample_interval_seconds is not None
        and not np.isclose(stored, model.sample_interval_seconds, rtol=0.0, atol=1e-9)
    ):
        raise ValueError("KMeans model and feature store use different sample intervals")


def _validate_pca_store(model: StreamingPCAModel, store: FeatureStore) -> None:
    if store.feature_keys != model.feature_keys:
        raise ValueError("Streaming PCA and feature store use different feature identities or order")
    if store.source_contract != model.source_contract:
        raise ValueError("Streaming PCA and feature store use different source contracts")
    stored = store.sample_interval_seconds
    if (stored is None) != (model.sample_interval_seconds is None) or (
        stored is not None
        and model.sample_interval_seconds is not None
        and not np.isclose(stored, model.sample_interval_seconds, rtol=0.0, atol=1e-9)
    ):
        raise ValueError("Streaming PCA and feature store use different sample intervals")


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
    _validate_pca_store(model, store)
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
        yield TransformedFeatureChunk(
            values=reduced,
            sample_start_indices=chunk.sample_start_indices,
            sample_end_indices=chunk.sample_end_indices,
            subject=chunk.subject,
            session=chunk.session,
            acquisition_id=chunk.acquisition_id,
            segment_id=chunk.segment_id,
            sequence_index=chunk.sequence_index,
            chunk_id=chunk.chunk_id,
            start_in_sequence=chunk.start_in_sequence,
            stop_in_sequence=chunk.stop_in_sequence,
        )


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
    subjects: Iterable[str] | None = None,
) -> KMeansFitResult:
    """Fit MiniBatchKMeans while keeping feature rows in a ``FeatureStore``.

    ``max_iter`` is the number of complete passes through the selected store;
    each pass calls ``partial_fit`` for every bounded batch.  K-means++ centres
    are initialized from one deterministic uniform global-row sample, and the
    best of ``n_init`` independent initializations is selected by a full-store
    inertia pass.  The returned assignments contain labels and sample indices,
    not feature rows.
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

    try:
        import sklearn
        from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "streaming KMeans fitting requires the 'states' extra: "
            "pip install 'dfc-kit[states]'"
        ) from error

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
    )

    best_centers: NDArray[np.float64] | None = None
    best_inertia = np.inf
    best_iterations = 0
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
        for _ in range(max_iter):
            for batch in _iter_batches(
                store,
                fit_subjects,
                total_samples=n_samples,
                minimum_rows=n_states,
                mean=mean,
                scale=scale,
                batch_size=batch_size,
            ):
                estimator.partial_fit(batch)
                n_batches += 1
        centers = np.asarray(estimator.cluster_centers_, dtype=np.float64)
        inertia, _ = _labels_and_inertia(
            store,
            fit_subjects,
            centers=centers,
            mean=mean,
            scale=scale,
            collect_labels=False,
        )
        if inertia < best_inertia:
            best_centers = centers.copy()
            best_inertia = inertia
            best_iterations = n_batches

    assert best_centers is not None
    model = KMeansStateModel(
        centers=_readonly(best_centers * scale + mean),
        standardized_centers=_readonly(best_centers),
        feature_mean=_readonly(mean),
        feature_scale=_readonly(scale),
        feature_keys=store.feature_keys,
        source_contract=store.source_contract,
        sample_interval_seconds=store.sample_interval_seconds,
        n_states=n_states,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        algorithm="minibatch",
        standardize_features=bool(standardize_features),
        batch_size=batch_size,
        reassignment_ratio=float(reassignment_ratio),
        init_sample_size=init_sample_size,
        iterations=best_iterations,
        inertia=float(best_inertia),
        fit_subjects=fit_subjects,
        fit_sample_count=n_samples,
        training_data_fingerprint=store.data_fingerprint(subjects=fit_subjects),
        implementation=(
            f"scikit-learn {sklearn.__version__} MiniBatchKMeans; "
            f"out-of-core partial_fit passes={max_iter}; kmeans++ sample={init_sample_size}"
        ),
    )
    assignments = _assignments_from_store(store, fit_subjects, model=model)
    counts = np.zeros(n_states, dtype=np.int64)
    for sequence in assignments.sequences:
        counts += np.bincount(sequence.labels, minlength=n_states)
    if np.any(counts == 0):
        raise RuntimeError(f"KMeans returned an empty state; fitted counts={counts.tolist()}")
    return KMeansFitResult(model=model, assignments=assignments)


def predict_kmeans_store(
    model: KMeansStateModel,
    store: FeatureStore,
    *,
    subjects: Iterable[str] | None = None,
    allow_fit_subjects: bool = False,
) -> StateAssignments:
    """Assign a store's rows without materializing its feature matrix."""
    _validate_store_model(model, store)
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
    _validate_store_model(model, store)
    if not isinstance(allow_fit_subjects, (bool, np.bool_)):
        raise TypeError("allow_fit_subjects must be boolean")
    selected_subjects = _subjects(store, subjects)
    overlap = sorted(set(model.fit_subjects).intersection(selected_subjects))
    if overlap and not allow_fit_subjects:
        raise ValueError(f"scoring subjects overlap KMeans fit subjects: {overlap}")

    center_norm = np.einsum(
        "ij,ij->i",
        model.standardized_centers,
        model.standardized_centers,
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
        distances = (
            np.einsum("ij,ij->i", values, values)[:, None]
            + center_norm[None, :]
            - 2.0 * values @ model.standardized_centers.T
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
            mean_squared_distance=total / sample_counts[identity],
        )
        for identity, total in totals.items()
        for subject, session, acquisition_id in (identity,)
    )


__all__ = [
    "StreamingPCAModel",
    "TransformedFeatureChunk",
    "fit_incremental_pca_store",
    "fit_minibatch_kmeans_store",
    "iter_pca_store_chunks",
    "predict_kmeans_store",
    "score_kmeans_store",
]
