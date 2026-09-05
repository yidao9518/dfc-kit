"""Portable, pickle-free persistence for fitted dfc-kit models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..states.hmm import GaussianHMMStateModel
from ..states.kmeans import KMeansStateModel
from ..states.streaming import StreamingPCAModel
from ._fields import artifact_finite_float, artifact_integer
from ._numpy import load_numpy_artifact, write_numpy_artifact

FORMAT_NAME = "dfckit-fitted-model"
FORMAT_VERSION = 4

FittedModel: TypeAlias = KMeansStateModel | GaussianHMMStateModel | StreamingPCAModel

_KMEANS_ARRAYS = {
    "centers",
    "clustering_centers",
    "feature_mean",
    "feature_scale",
    "pca_components",
    "pca_explained_variance_ratio",
    "pca_mean",
    "standardized_centers",
}
_KMEANS_METADATA = {
    "algorithm",
    "batch_size",
    "feature_keys",
    "fit_sample_count",
    "fit_subjects",
    "implementation",
    "init_sample_size",
    "inertia",
    "iterations",
    "max_iter",
    "n_init",
    "n_pca_components",
    "pca_batch_size",
    "n_states",
    "reassignment_ratio",
    "sample_interval_seconds",
    "seed",
    "source_contract",
    "standardize_features",
}
_PCA_ARRAYS = {
    "explained_variance_ratio",
    "feature_mean",
    "feature_scale",
    "pca_components",
    "pca_mean",
}
_PCA_METADATA = {
    "batch_size",
    "feature_keys",
    "fit_sample_count",
    "fit_sequence_count",
    "fit_subjects",
    "implementation",
    "n_components",
    "sample_interval_seconds",
    "source_contract",
    "standardize_features",
}
_HMM_REQUIRED_ARRAYS = {
    "emission_means",
    "feature_mean",
    "feature_scale",
    "initialization_log_likelihoods",
    "pca_components",
    "pca_explained_variance_ratio",
    "pca_mean",
    "reduced_covariances",
    "reduced_means",
    "start_probabilities",
    "transition_matrix",
}
_HMM_METADATA = {
    "converged",
    "covariance_type",
    "feature_keys",
    "fit_sample_count",
    "fit_sequence_count",
    "fit_subjects",
    "implementation",
    "initialization_seeds",
    "iterations",
    "log_likelihood",
    "minimum_sequence_length",
    "n_init",
    "n_iter",
    "n_pca_components",
    "n_states",
    "omitted_short_sequence_count",
    "pca_batch_size",
    "sample_interval_seconds",
    "seed",
    "selected_initialization",
    "source_contract",
    "tol",
}


def _readonly_float(value: object, name: str) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"model array {name!r} contains non-finite values")
    array.setflags(write=False)
    return array


def _metadata_fields(metadata: object, expected: set[str]) -> dict[str, object]:
    if not isinstance(metadata, dict):
        raise TypeError("model artifact metadata must be a JSON object")
    observed = set(metadata)
    if observed != expected:
        raise ValueError(
            "model artifact metadata fields do not match the schema; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return metadata


def _integer(metadata: Mapping[str, object], name: str, *, minimum: int = 0) -> int:
    return artifact_integer(metadata[name], f"model metadata {name!r}", minimum=minimum)


def _finite_float(metadata: Mapping[str, object], name: str, *, positive: bool = False) -> float:
    return artifact_finite_float(
        metadata[name],
        f"model metadata {name!r}",
        positive=positive,
    )


def _optional_float(metadata: Mapping[str, object], name: str) -> float | None:
    return None if metadata[name] is None else _finite_float(metadata, name)


def _optional_integer(
    metadata: Mapping[str, object],
    name: str,
    *,
    minimum: int = 0,
) -> int | None:
    return None if metadata[name] is None else _integer(metadata, name, minimum=minimum)


def _boolean(metadata: Mapping[str, object], name: str) -> bool:
    value = metadata[name]
    if not isinstance(value, bool):
        raise TypeError(f"model metadata {name!r} must be boolean")
    return value


def _string(metadata: Mapping[str, object], name: str) -> str:
    value = metadata[name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model metadata {name!r} must be a non-empty string")
    return value


def _string_tuple(metadata: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = metadata[name]
    if not isinstance(value, list) or not value:
        raise TypeError(f"model metadata {name!r} must be a non-empty JSON list")
    output = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in output):
        raise ValueError(f"model metadata {name!r} contains an invalid string")
    if len(set(output)) != len(output):
        raise ValueError(f"model metadata {name!r} contains duplicates")
    return output


def _integer_tuple(metadata: Mapping[str, object], name: str) -> tuple[int, ...]:
    value = metadata[name]
    if not isinstance(value, list) or not value:
        raise TypeError(f"model metadata {name!r} must be a non-empty JSON list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"model metadata {name!r} contains an invalid integer")
    return tuple(value)


def _feature_keys(metadata: Mapping[str, object]) -> tuple[tuple[str, ...], ...]:
    value = metadata["feature_keys"]
    if not isinstance(value, list) or not value:
        raise TypeError("model metadata 'feature_keys' must be a non-empty JSON list")
    output = []
    for key in value:
        if not isinstance(key, list) or not key or any(not isinstance(part, str) for part in key):
            raise ValueError("model metadata 'feature_keys' contains an invalid key")
        output.append(tuple(key))
    result = tuple(output)
    if len(set(result)) != len(result):
        raise ValueError("model metadata 'feature_keys' contains duplicates")
    return result


def _sample_interval(metadata: Mapping[str, object]) -> float | None:
    value = _optional_float(metadata, "sample_interval_seconds")
    if value is not None and value <= 0.0:
        raise ValueError("model sample interval must be positive")
    return value


def _array_payload(model: FittedModel) -> tuple[str, dict[str, object], dict[str, NDArray]]:
    if not isinstance(model, (KMeansStateModel, GaussianHMMStateModel, StreamingPCAModel)):
        raise TypeError(f"unsupported fitted model type: {type(model).__name__}")
    common = {
        "feature_keys": [list(key) for key in model.feature_keys],
        "source_contract": str(model.source_contract),
        "sample_interval_seconds": model.sample_interval_seconds,
        "fit_subjects": list(model.fit_subjects),
        "fit_sample_count": int(model.fit_sample_count),
        "implementation": str(model.implementation),
    }
    if isinstance(model, KMeansStateModel):
        metadata = {
            **common,
            "n_states": int(model.n_states),
            "seed": int(model.seed),
            "n_init": int(model.n_init),
            "max_iter": int(model.max_iter),
            "algorithm": str(model.algorithm),
            "standardize_features": bool(model.standardize_features),
            "n_pca_components": (
                None
                if model.n_pca_components is None
                else int(model.n_pca_components)
            ),
            "pca_batch_size": (
                None if model.pca_batch_size is None else int(model.pca_batch_size)
            ),
            "batch_size": None if model.batch_size is None else int(model.batch_size),
            "reassignment_ratio": (
                None if model.reassignment_ratio is None else float(model.reassignment_ratio)
            ),
            "init_sample_size": (
                None if model.init_sample_size is None else int(model.init_sample_size)
            ),
            "iterations": int(model.iterations),
            "inertia": float(model.inertia),
        }
        arrays = {name: np.asarray(getattr(model, name)) for name in _KMEANS_ARRAYS}
        return "kmeans-state", metadata, arrays
    if isinstance(model, GaussianHMMStateModel):
        metadata = {
            **common,
            "n_states": int(model.n_states),
            "n_pca_components": int(model.n_pca_components),
            "covariance_type": str(model.covariance_type),
            "seed": int(model.seed),
            "n_init": int(model.n_init),
            "n_iter": int(model.n_iter),
            "tol": float(model.tol),
            "minimum_sequence_length": int(model.minimum_sequence_length),
            "pca_batch_size": (
                None if model.pca_batch_size is None else int(model.pca_batch_size)
            ),
            "selected_initialization": int(model.selected_initialization),
            "initialization_seeds": [int(value) for value in model.initialization_seeds],
            "iterations": int(model.iterations),
            "converged": bool(model.converged),
            "log_likelihood": float(model.log_likelihood),
            "fit_sequence_count": int(model.fit_sequence_count),
            "omitted_short_sequence_count": int(model.omitted_short_sequence_count),
        }
        arrays = {
            name: np.asarray(getattr(model, name)) for name in _HMM_REQUIRED_ARRAYS
        }
        if model.emission_covariances is not None:
            arrays["emission_covariances"] = np.asarray(model.emission_covariances)
        return "gaussian-hmm-state", metadata, arrays
    if isinstance(model, StreamingPCAModel):
        metadata = {
            **common,
            "n_components": int(model.n_components),
            "standardize_features": bool(model.standardize_features),
            "batch_size": int(model.batch_size),
            "fit_sequence_count": int(model.fit_sequence_count),
        }
        arrays = {name: np.asarray(getattr(model, name)) for name in _PCA_ARRAYS}
        return "streaming-pca", metadata, arrays
    raise TypeError(f"unsupported fitted model type: {type(model).__name__}")


def save_fitted_model(model: FittedModel, path: str | Path) -> Path:
    """Write one fitted model as JSON metadata plus pickle-free NumPy arrays."""
    kind, metadata, arrays = _array_payload(model)
    return write_numpy_artifact(
        path,
        {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "model_kind": kind,
            "array_names": sorted(arrays),
            "metadata": metadata,
        },
        arrays,
        label="model",
    )


def _load_manifest(root: Path) -> tuple[str, dict[str, object], dict[str, NDArray[np.float64]]]:
    expected_fields = {"array_names", "format", "format_version", "metadata", "model_kind"}
    manifest, arrays = load_numpy_artifact(
        root,
        label="model",
        manifest_fields=expected_fields,
    )
    arrays = {name: _readonly_float(value, name) for name, value in arrays.items()}
    if manifest["format"] != FORMAT_NAME:
        raise ValueError("not a dfc-kit fitted-model artifact")
    version = manifest["format_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != FORMAT_VERSION
    ):
        raise ValueError(f"unsupported model artifact format version: {manifest['format_version']!r}")
    kind = manifest["model_kind"]
    if kind not in {"kmeans-state", "gaussian-hmm-state", "streaming-pca"}:
        raise ValueError(f"unsupported fitted model kind: {kind!r}")
    metadata = manifest["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("model artifact metadata must be a JSON object")
    return str(kind), metadata, arrays


def _require_arrays(
    arrays: Mapping[str, NDArray[np.float64]],
    expected: set[str],
) -> None:
    observed = set(arrays)
    if observed != expected:
        raise ValueError(
            "model artifact arrays do not match the model schema; "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )


def _load_kmeans(
    metadata_value: object,
    arrays: dict[str, NDArray[np.float64]],
) -> KMeansStateModel:
    metadata = _metadata_fields(metadata_value, _KMEANS_METADATA)
    _require_arrays(arrays, _KMEANS_ARRAYS)
    keys = _feature_keys(metadata)
    n_states = _integer(metadata, "n_states", minimum=2)
    n_pca_components = _optional_integer(metadata, "n_pca_components", minimum=1)
    cluster_width = len(keys) if n_pca_components is None else n_pca_components
    shape = (n_states, len(keys))
    if arrays["centers"].shape != shape or arrays["standardized_centers"].shape != shape:
        raise ValueError("KMeans artifact centers have an invalid shape")
    if arrays["clustering_centers"].shape != (n_states, cluster_width):
        raise ValueError("KMeans artifact clustering_centers have an invalid shape")
    for name in ("feature_mean", "feature_scale"):
        if arrays[name].shape != (len(keys),):
            raise ValueError(f"KMeans artifact {name} has an invalid shape")
    if np.any(arrays["feature_scale"] <= 0.0):
        raise ValueError("KMeans artifact feature_scale must be positive")
    expected_pca_shapes = (
        ((0, len(keys)), (0,))
        if n_pca_components is None
        else ((n_pca_components, len(keys)), (n_pca_components,))
    )
    if (
        arrays["pca_mean"].shape != (len(keys),)
        or arrays["pca_components"].shape != expected_pca_shapes[0]
        or arrays["pca_explained_variance_ratio"].shape != expected_pca_shapes[1]
    ):
        raise ValueError("KMeans artifact PCA arrays have invalid shapes")
    algorithm = _string(metadata, "algorithm")
    if algorithm not in {"lloyd", "minibatch"}:
        raise ValueError("KMeans artifact algorithm is invalid")
    batch_size = _optional_integer(metadata, "batch_size", minimum=1)
    reassignment_ratio = _optional_float(metadata, "reassignment_ratio")
    init_sample_size = _optional_integer(metadata, "init_sample_size", minimum=1)
    if algorithm == "lloyd" and (batch_size is not None or reassignment_ratio is not None):
        raise ValueError("Lloyd KMeans artifact cannot contain mini-batch parameters")
    if algorithm == "lloyd" and init_sample_size is not None:
        raise ValueError("Lloyd KMeans artifact cannot contain init_sample_size")
    if algorithm == "minibatch" and (batch_size is None or reassignment_ratio is None):
        raise ValueError("MiniBatchKMeans artifact requires batch parameters")
    if reassignment_ratio is not None and not 0.0 <= reassignment_ratio <= 1.0:
        raise ValueError("KMeans artifact reassignment_ratio must be within [0, 1]")
    inertia = _finite_float(metadata, "inertia")
    if inertia < 0.0:
        raise ValueError("KMeans artifact inertia must be non-negative")
    fit_sample_count = _integer(metadata, "fit_sample_count", minimum=n_states)
    return KMeansStateModel(
        centers=arrays["centers"],
        standardized_centers=arrays["standardized_centers"],
        clustering_centers=arrays["clustering_centers"],
        feature_mean=arrays["feature_mean"],
        feature_scale=arrays["feature_scale"],
        pca_mean=arrays["pca_mean"],
        pca_components=arrays["pca_components"],
        pca_explained_variance_ratio=arrays["pca_explained_variance_ratio"],
        feature_keys=keys,
        source_contract=_string(metadata, "source_contract"),
        sample_interval_seconds=_sample_interval(metadata),
        n_states=n_states,
        seed=_integer(metadata, "seed"),
        n_init=_integer(metadata, "n_init", minimum=1),
        max_iter=_integer(metadata, "max_iter", minimum=1),
        algorithm=algorithm,
        standardize_features=_boolean(metadata, "standardize_features"),
        n_pca_components=n_pca_components,
        pca_batch_size=_optional_integer(metadata, "pca_batch_size", minimum=1),
        batch_size=batch_size,
        reassignment_ratio=reassignment_ratio,
        init_sample_size=init_sample_size,
        iterations=_integer(metadata, "iterations", minimum=1),
        inertia=inertia,
        fit_subjects=_string_tuple(metadata, "fit_subjects"),
        fit_sample_count=fit_sample_count,
        implementation=_string(metadata, "implementation"),
    )


def _load_pca(
    metadata_value: object,
    arrays: dict[str, NDArray[np.float64]],
) -> StreamingPCAModel:
    metadata = _metadata_fields(metadata_value, _PCA_METADATA)
    _require_arrays(arrays, _PCA_ARRAYS)
    keys = _feature_keys(metadata)
    n_components = _integer(metadata, "n_components", minimum=1)
    if n_components > len(keys):
        raise ValueError("Streaming PCA component count exceeds its feature count")
    if np.any(arrays["feature_scale"] <= 0.0):
        raise ValueError("Streaming PCA artifact feature_scale must be positive")
    explained = arrays["explained_variance_ratio"]
    if np.any(explained < 0.0) or explained.sum() > 1.0 + 1e-7:
        raise ValueError("Streaming PCA explained variance ratios are invalid")
    return StreamingPCAModel(
        feature_mean=arrays["feature_mean"],
        feature_scale=arrays["feature_scale"],
        pca_mean=arrays["pca_mean"],
        pca_components=arrays["pca_components"],
        explained_variance_ratio=arrays["explained_variance_ratio"],
        feature_keys=keys,
        source_contract=_string(metadata, "source_contract"),
        sample_interval_seconds=_sample_interval(metadata),
        n_components=n_components,
        standardize_features=_boolean(metadata, "standardize_features"),
        batch_size=_integer(metadata, "batch_size", minimum=1),
        fit_subjects=_string_tuple(metadata, "fit_subjects"),
        fit_sample_count=_integer(metadata, "fit_sample_count", minimum=1),
        fit_sequence_count=_integer(metadata, "fit_sequence_count", minimum=1),
        implementation=_string(metadata, "implementation"),
    )


def _load_hmm(
    metadata_value: object,
    arrays: dict[str, NDArray[np.float64]],
) -> GaussianHMMStateModel:
    metadata = _metadata_fields(metadata_value, _HMM_METADATA)
    expected = set(_HMM_REQUIRED_ARRAYS)
    if "emission_covariances" in arrays:
        expected.add("emission_covariances")
    _require_arrays(arrays, expected)
    keys = _feature_keys(metadata)
    n_features = len(keys)
    n_states = _integer(metadata, "n_states", minimum=2)
    n_components = _integer(metadata, "n_pca_components", minimum=1)
    if n_components > n_features:
        raise ValueError("Gaussian HMM PCA component count exceeds its feature count")
    n_init = _integer(metadata, "n_init", minimum=1)
    shapes = {
        "start_probabilities": (n_states,),
        "transition_matrix": (n_states, n_states),
        "reduced_means": (n_states, n_components),
        "reduced_covariances": (n_states, n_components, n_components),
        "emission_means": (n_states, n_features),
        "feature_mean": (n_features,),
        "feature_scale": (n_features,),
        "pca_mean": (n_features,),
        "pca_components": (n_components, n_features),
        "pca_explained_variance_ratio": (n_components,),
        "initialization_log_likelihoods": (n_init,),
    }
    if "emission_covariances" in arrays:
        shapes["emission_covariances"] = (n_states, n_features, n_features)
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Gaussian HMM artifact {name} has an invalid shape")
    start = arrays["start_probabilities"]
    transition = arrays["transition_matrix"]
    if np.any(start < 0.0) or not np.isclose(start.sum(), 1.0, atol=1e-7):
        raise ValueError("Gaussian HMM artifact start probabilities are invalid")
    if np.any(transition < 0.0) or not np.allclose(
        transition.sum(axis=1), 1.0, atol=1e-7
    ):
        raise ValueError("Gaussian HMM artifact transition matrix is invalid")
    if np.any(arrays["feature_scale"] <= 0.0):
        raise ValueError("Gaussian HMM artifact feature_scale must be positive")
    explained = arrays["pca_explained_variance_ratio"]
    if np.any(explained < 0.0) or explained.sum() > 1.0 + 1e-7:
        raise ValueError("Gaussian HMM PCA explained variance ratios are invalid")
    covariance_type = _string(metadata, "covariance_type")
    if covariance_type not in {"diag", "full"}:
        raise ValueError("Gaussian HMM artifact covariance_type is invalid")
    reduced_covariances = arrays["reduced_covariances"]
    if not np.allclose(
        reduced_covariances,
        np.swapaxes(reduced_covariances, 1, 2),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("Gaussian HMM reduced covariances must be symmetric")
    reduced_diagonal = np.diagonal(reduced_covariances, axis1=1, axis2=2)
    if np.any(reduced_diagonal <= 0.0):
        raise ValueError("Gaussian HMM reduced covariance diagonal must be positive")
    if covariance_type == "diag":
        diagonal_matrices = np.zeros_like(reduced_covariances)
        diagonal_indices = np.arange(n_components)
        diagonal_matrices[:, diagonal_indices, diagonal_indices] = reduced_diagonal
        if not np.allclose(reduced_covariances, diagonal_matrices, rtol=0.0, atol=1e-10):
            raise ValueError("diagonal Gaussian HMM covariances contain off-diagonal values")
    elif np.any(np.linalg.eigvalsh(reduced_covariances) <= 0.0):
        raise ValueError("full Gaussian HMM reduced covariances must be positive definite")
    emission_covariances = arrays.get("emission_covariances")
    if emission_covariances is not None and (
        not np.allclose(
            emission_covariances,
            np.swapaxes(emission_covariances, 1, 2),
            rtol=0.0,
            atol=1e-9,
        )
        or np.any(np.diagonal(emission_covariances, axis1=1, axis2=2) < -1e-10)
    ):
        raise ValueError("Gaussian HMM emission covariances are invalid")
    initialization_seeds = _integer_tuple(metadata, "initialization_seeds")
    if len(initialization_seeds) != n_init:
        raise ValueError("Gaussian HMM initialization seed count does not match n_init")
    selected = _integer(metadata, "selected_initialization")
    if selected >= n_init:
        raise ValueError("Gaussian HMM selected initialization is out of range")
    return GaussianHMMStateModel(
        start_probabilities=start,
        transition_matrix=transition,
        reduced_means=arrays["reduced_means"],
        reduced_covariances=reduced_covariances,
        emission_means=arrays["emission_means"],
        emission_covariances=emission_covariances,
        feature_mean=arrays["feature_mean"],
        feature_scale=arrays["feature_scale"],
        pca_mean=arrays["pca_mean"],
        pca_components=arrays["pca_components"],
        pca_explained_variance_ratio=arrays["pca_explained_variance_ratio"],
        feature_keys=keys,
        source_contract=_string(metadata, "source_contract"),
        sample_interval_seconds=_sample_interval(metadata),
        n_states=n_states,
        n_pca_components=n_components,
        covariance_type=covariance_type,
        seed=_integer(metadata, "seed"),
        n_init=n_init,
        n_iter=_integer(metadata, "n_iter", minimum=1),
        tol=_finite_float(metadata, "tol", positive=True),
        minimum_sequence_length=_integer(metadata, "minimum_sequence_length", minimum=1),
        pca_batch_size=_optional_integer(metadata, "pca_batch_size", minimum=1),
        selected_initialization=selected,
        initialization_seeds=initialization_seeds,
        initialization_log_likelihoods=arrays["initialization_log_likelihoods"],
        iterations=_integer(metadata, "iterations", minimum=1),
        converged=_boolean(metadata, "converged"),
        log_likelihood=_finite_float(metadata, "log_likelihood"),
        fit_subjects=_string_tuple(metadata, "fit_subjects"),
        fit_sample_count=_integer(metadata, "fit_sample_count", minimum=n_states),
        fit_sequence_count=_integer(metadata, "fit_sequence_count", minimum=1),
        omitted_short_sequence_count=_integer(metadata, "omitted_short_sequence_count"),
        implementation=_string(metadata, "implementation"),
    )


def load_fitted_model(path: str | Path) -> FittedModel:
    """Load and validate one model written by :func:`save_fitted_model`."""
    kind, metadata, arrays = _load_manifest(Path(path))
    if kind == "kmeans-state":
        return _load_kmeans(metadata, arrays)
    if kind == "streaming-pca":
        return _load_pca(metadata, arrays)
    return _load_hmm(metadata, arrays)
