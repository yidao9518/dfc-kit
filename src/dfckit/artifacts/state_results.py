"""Portable, pickle-free persistence for decoded state sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._arrays import readonly_copy as _readonly
from ..states.data import StateAssignments, StateLabelSequence
from ..states.metrics import summarize_state_assignments
from ._fields import artifact_integer
from ._json import write_json_atomic
from ._numpy import load_numpy_artifact, write_numpy_artifact

FORMAT_NAME = "dfckit-state-predictions"
FORMAT_VERSION = 4
METRICS_FORMAT_NAME = "dfckit-state-metrics"
METRICS_FORMAT_VERSION = 1


@dataclass(frozen=True)
class StatePredictions:
    """Decoded state assignments with optional Gaussian-HMM probabilities."""

    assignments: StateAssignments
    model_kind: str
    model_seed: int
    posterior_probabilities: tuple[ArrayLike, ...] | None = None
    log_likelihood: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.assignments, StateAssignments):
            raise TypeError("assignments must be a StateAssignments object")
        if self.model_kind not in {"kmeans-state", "gaussian-hmm-state"}:
            raise ValueError("model_kind must be 'kmeans-state' or 'gaussian-hmm-state'")
        if (
            isinstance(self.model_seed, (bool, np.bool_))
            or not isinstance(self.model_seed, (int, np.integer))
            or self.model_seed < 0
        ):
            raise ValueError("model_seed must be a non-negative integer")
        object.__setattr__(self, "model_seed", int(self.model_seed))
        if self.model_kind == "kmeans-state":
            if self.posterior_probabilities is not None or self.log_likelihood is not None:
                raise ValueError("KMeans predictions cannot contain posterior probabilities")
            return
        if self.posterior_probabilities is None or self.log_likelihood is None:
            raise ValueError("Gaussian HMM predictions require posteriors and log likelihood")
        posteriors = tuple(
            _readonly(np.asarray(values, dtype=float))
            for values in self.posterior_probabilities
        )
        if len(posteriors) != len(self.assignments.sequences):
            raise ValueError("posterior sequence count does not match state assignments")
        for posterior, sequence in zip(
            posteriors,
            self.assignments.sequences,
            strict=True,
        ):
            expected = (len(sequence.labels), self.assignments.n_states)
            if posterior.shape != expected:
                raise ValueError("posterior shape does not match sequence labels and states")
            if (
                not np.isfinite(posterior).all()
                or np.any(posterior < 0.0)
                or np.any(posterior > 1.0)
                or not np.allclose(posterior.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
            ):
                raise ValueError("posterior probabilities are invalid")
        if not np.isfinite(self.log_likelihood):
            raise ValueError("log_likelihood must be finite")
        object.__setattr__(self, "posterior_probabilities", posteriors)
        object.__setattr__(self, "log_likelihood", float(self.log_likelihood))

    @property
    def n_samples(self) -> int:
        return sum(len(sequence.labels) for sequence in self.assignments.sequences)

    @property
    def n_sequences(self) -> int:
        return len(self.assignments.sequences)

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(sequence.subject for sequence in self.assignments.sequences)
        )


def _sequence_metadata(assignments: StateAssignments) -> list[dict[str, object]]:
    return [
        {
            "subject": sequence.subject,
            "session": sequence.session,
            "acquisition_id": sequence.acquisition_id,
            "segment_id": sequence.segment_id,
            "n_samples": len(sequence.labels),
        }
        for sequence in assignments.sequences
    ]


def save_state_predictions(predictions: StatePredictions, path: str | Path) -> Path:
    """Write decoded labels, indices, and optional posteriors without pickle."""
    if not isinstance(predictions, StatePredictions):
        raise TypeError("predictions must be a StatePredictions object")
    assignments = predictions.assignments
    lengths = np.asarray(
        [len(sequence.labels) for sequence in assignments.sequences],
        dtype=np.int64,
    )
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    arrays: dict[str, NDArray] = {
        "labels": np.concatenate([sequence.labels for sequence in assignments.sequences]),
        "sample_start_indices": np.concatenate(
            [sequence.sample_start_indices for sequence in assignments.sequences]
        ),
        "sample_end_indices": np.concatenate(
            [sequence.sample_end_indices for sequence in assignments.sequences]
        ),
        "sequence_offsets": offsets,
    }
    if predictions.posterior_probabilities is not None:
        arrays["posterior_probabilities"] = np.concatenate(
            predictions.posterior_probabilities,
            axis=0,
        )
    return write_numpy_artifact(
        path,
        {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "model_kind": predictions.model_kind,
            "model_seed": predictions.model_seed,
            "array_names": sorted(arrays),
            "n_states": assignments.n_states,
            "source_contract": assignments.source_contract,
            "sample_interval_seconds": assignments.sample_interval_seconds,
            "log_likelihood": predictions.log_likelihood,
            "sequences": _sequence_metadata(assignments),
        },
        arrays,
        label="state-prediction",
    )


def _manifest(root: Path) -> tuple[dict[str, object], dict[str, NDArray]]:
    expected = {
        "array_names",
        "format",
        "format_version",
        "log_likelihood",
        "model_kind",
        "model_seed",
        "n_states",
        "sample_interval_seconds",
        "sequences",
        "source_contract",
    }
    manifest, arrays = load_numpy_artifact(
        root,
        label="state-prediction",
        manifest_fields=expected,
    )
    if manifest["format"] != FORMAT_NAME or manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported state-prediction artifact format or version")
    return manifest, arrays


def load_state_predictions(path: str | Path) -> StatePredictions:
    """Load and validate predictions written by :func:`save_state_predictions`."""
    manifest, arrays = _manifest(Path(path))
    model_kind = manifest["model_kind"]
    model_seed = artifact_integer(
        manifest["model_seed"], "state-prediction model_seed", minimum=0
    )
    expected_arrays = {
        "labels",
        "sample_start_indices",
        "sample_end_indices",
        "sequence_offsets",
    }
    if model_kind == "gaussian-hmm-state":
        expected_arrays.add("posterior_probabilities")
    elif model_kind != "kmeans-state":
        raise ValueError("state-prediction model_kind is invalid")
    if set(arrays) != expected_arrays:
        raise ValueError("state-prediction array set is invalid for its model kind")
    n_states = artifact_integer(
        manifest["n_states"], "state-prediction n_states", minimum=2
    )
    source_contract = manifest["source_contract"]
    if not isinstance(source_contract, str) or not source_contract.strip():
        raise ValueError("state-prediction source_contract must be non-empty")
    interval = manifest["sample_interval_seconds"]
    if interval is not None and (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not np.isfinite(interval)
        or interval <= 0.0
    ):
        raise ValueError("state-prediction sample interval is invalid")
    records = manifest["sequences"]
    if not isinstance(records, list) or not records:
        raise ValueError("state-prediction sequences must be a non-empty list")
    offsets = np.asarray(arrays["sequence_offsets"])
    labels = np.asarray(arrays["labels"])
    starts = np.asarray(arrays["sample_start_indices"])
    ends = np.asarray(arrays["sample_end_indices"])
    if (
        offsets.dtype.kind not in "iu"
        or offsets.shape != (len(records) + 1,)
        or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("state-prediction sequence offsets are invalid")
    total = int(offsets[-1])
    if (
        labels.dtype.kind not in "iu"
        or starts.dtype.kind not in "iu"
        or ends.dtype.kind not in "iu"
        or labels.shape != (total,)
        or starts.shape != (total,)
        or ends.shape != (total,)
    ):
        raise ValueError("state-prediction label or index arrays are invalid")
    sequences = []
    expected_record_fields = {
        "acquisition_id",
        "n_samples",
        "segment_id",
        "session",
        "subject",
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != expected_record_fields:
            raise ValueError("state-prediction sequence metadata is invalid")
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        if artifact_integer(
            record["n_samples"], "state-prediction sequence n_samples", minimum=1
        ) != stop - start:
            raise ValueError("state-prediction sequence length does not match offsets")
        sequences.append(
            StateLabelSequence(
                labels=labels[start:stop],
                sample_start_indices=starts[start:stop],
                sample_end_indices=ends[start:stop],
                subject=record["subject"],
                session=record["session"],
                acquisition_id=record["acquisition_id"],
                segment_id=record["segment_id"],
            )
        )
    assignments = StateAssignments(
        sequences=tuple(sequences),
        n_states=n_states,
        source_contract=source_contract,
        sample_interval_seconds=None if interval is None else float(interval),
    )
    posterior = None
    log_likelihood = manifest["log_likelihood"]
    if model_kind == "gaussian-hmm-state":
        pooled = np.asarray(arrays["posterior_probabilities"], dtype=float)
        if pooled.shape != (total, n_states):
            raise ValueError("state-prediction posterior array has an invalid shape")
        posterior = tuple(
            pooled[int(offsets[index]) : int(offsets[index + 1])]
            for index in range(len(records))
        )
    return StatePredictions(
        assignments=assignments,
        model_kind=str(model_kind),
        model_seed=model_seed,
        posterior_probabilities=posterior,
        log_likelihood=log_likelihood,
    )


def _nullable(values: NDArray) -> list:
    array = np.asarray(values)
    if array.ndim == 1:
        return [None if not np.isfinite(value) else float(value) for value in array]
    return [_nullable(row) for row in array]


def write_state_metrics(predictions: StatePredictions, path: str | Path) -> Path:
    """Summarize one prediction artifact as strict machine-readable JSON."""
    if not isinstance(predictions, StatePredictions):
        raise TypeError("predictions must be a StatePredictions object")
    runs = []
    for metric in summarize_state_assignments(predictions.assignments):
        runs.append(
            {
                "subject": metric.subject,
                "session": metric.session,
                "acquisition_id": metric.acquisition_id,
                "n_samples": metric.n_samples,
                "n_sequences": metric.n_sequences,
                "n_possible_transitions": metric.n_possible_transitions,
                "n_switches": metric.n_switches,
                "switch_rate": (
                    None if not np.isfinite(metric.switch_rate) else metric.switch_rate
                ),
                "occupancy": _nullable(metric.occupancy),
                "mean_dwell_samples": _nullable(metric.mean_dwell_samples),
                "mean_dwell_seconds": _nullable(metric.mean_dwell_seconds),
                "transition_counts": np.asarray(metric.transition_counts).tolist(),
                "transition_probabilities": _nullable(metric.transition_probabilities),
            }
        )
    payload = {
        "format": METRICS_FORMAT_NAME,
        "format_version": METRICS_FORMAT_VERSION,
        "model_kind": predictions.model_kind,
        "model_seed": predictions.model_seed,
        "n_states": predictions.assignments.n_states,
        "source_contract": predictions.assignments.source_contract,
        "sample_interval_seconds": predictions.assignments.sample_interval_seconds,
        "n_runs": len(runs),
        "runs": runs,
    }
    return write_json_atomic(path, payload)


__all__ = [
    "StatePredictions",
    "load_state_predictions",
    "save_state_predictions",
    "write_state_metrics",
]
