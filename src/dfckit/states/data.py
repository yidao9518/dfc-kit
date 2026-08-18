"""Feature sequences and state-label results with explicit gap boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..connectivity import LEiDAResult, WindowFCResult
from ..data import TimeSeriesDataset

FeatureKey = tuple[str, ...]


def _readonly(values: NDArray) -> NDArray:
    output = np.asarray(values).copy()
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class FeatureSequence:
    """One uninterrupted sequence of samples in a common feature space."""

    values: ArrayLike
    sample_start_indices: ArrayLike
    sample_end_indices: ArrayLike
    feature_keys: tuple[FeatureKey, ...]
    subject: str
    session: str | None
    segment_id: int
    source_contract: str
    sample_interval_seconds: float | None = None
    acquisition_id: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        starts = np.asarray(self.sample_start_indices, dtype=np.int64)
        ends = np.asarray(self.sample_end_indices, dtype=np.int64)
        keys = tuple(tuple(str(part) for part in key) for key in self.feature_keys)
        if values.ndim != 2 or not len(values) or not values.shape[1]:
            raise ValueError("feature values must be a non-empty samples-by-features array")
        if not np.isfinite(values).all():
            raise ValueError("feature values contain non-finite samples")
        if (
            starts.ndim != 1
            or ends.ndim != 1
            or len(starts) != len(values)
            or len(ends) != len(values)
        ):
            raise ValueError("sample indices must be one-dimensional and align with feature rows")
        if np.any(ends < starts):
            raise ValueError("sample end indices cannot precede sample start indices")
        if len(starts) > 1 and np.any(np.diff(starts) <= 0):
            raise ValueError("sample start indices must be strictly increasing")
        if len(keys) != values.shape[1] or any(not key for key in keys):
            raise ValueError("feature_keys must identify every feature")
        if len(set(keys)) != len(keys):
            raise ValueError("feature_keys must be unique")
        if not str(self.subject).strip():
            raise ValueError("subject must be a non-empty identifier")
        if self.session is not None and not str(self.session).strip():
            raise ValueError("session cannot be empty")
        if self.acquisition_id is not None and not str(self.acquisition_id).strip():
            raise ValueError("acquisition_id cannot be empty")
        if not isinstance(self.segment_id, (int, np.integer)) or self.segment_id < 0:
            raise ValueError("segment_id must be a non-negative integer")
        if not str(self.source_contract).strip():
            raise ValueError("source_contract must be non-empty")
        if self.sample_interval_seconds is not None and (
            not np.isfinite(self.sample_interval_seconds) or self.sample_interval_seconds <= 0
        ):
            raise ValueError("sample_interval_seconds must be finite and positive")
        object.__setattr__(self, "values", _readonly(values))
        object.__setattr__(self, "sample_start_indices", _readonly(starts))
        object.__setattr__(self, "sample_end_indices", _readonly(ends))
        object.__setattr__(self, "feature_keys", keys)
        object.__setattr__(self, "subject", str(self.subject))
        object.__setattr__(self, "session", None if self.session is None else str(self.session))
        object.__setattr__(
            self,
            "acquisition_id",
            None if self.acquisition_id is None else str(self.acquisition_id),
        )
        object.__setattr__(self, "segment_id", int(self.segment_id))
        object.__setattr__(self, "source_contract", str(self.source_contract))

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class FeatureSequenceDataset:
    """Sequences sharing one feature identity and construction contract."""

    sequences: Sequence[FeatureSequence]

    def __post_init__(self) -> None:
        sequences = tuple(self.sequences)
        if not sequences:
            raise ValueError("a feature-sequence dataset must contain at least one sequence")
        if any(not isinstance(sequence, FeatureSequence) for sequence in sequences):
            raise TypeError("sequences must contain only FeatureSequence objects")
        reference = sequences[0]
        identities: set[tuple[str, str | None, str | None, int]] = set()
        for sequence in sequences:
            if sequence.feature_keys != reference.feature_keys:
                raise ValueError("all sequences must use the same feature identity and order")
            if sequence.source_contract != reference.source_contract:
                raise ValueError("all sequences must use the same source contract")
            if (sequence.sample_interval_seconds is None) != (
                reference.sample_interval_seconds is None
            ):
                raise ValueError("sample interval must be known for every sequence or none")
            if (
                sequence.sample_interval_seconds is not None
                and reference.sample_interval_seconds is not None
                and not np.isclose(
                    sequence.sample_interval_seconds,
                    reference.sample_interval_seconds,
                    rtol=0.0,
                    atol=1e-9,
                )
            ):
                raise ValueError("all sequences must use the same sample interval")
            identity = (
                sequence.subject,
                sequence.session,
                sequence.acquisition_id,
                sequence.segment_id,
            )
            if identity in identities:
                raise ValueError(f"duplicate feature-sequence identity: {identity}")
            identities.add(identity)
        object.__setattr__(self, "sequences", sequences)

    @property
    def feature_keys(self) -> tuple[FeatureKey, ...]:
        return self.sequences[0].feature_keys

    @property
    def source_contract(self) -> str:
        return self.sequences[0].source_contract

    @property
    def sample_interval_seconds(self) -> float | None:
        return self.sequences[0].sample_interval_seconds

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(sequence.subject for sequence in self.sequences))

    @property
    def n_samples(self) -> int:
        return sum(sequence.n_samples for sequence in self.sequences)


@dataclass(frozen=True)
class StateLabelSequence:
    labels: ArrayLike
    sample_start_indices: ArrayLike
    sample_end_indices: ArrayLike
    subject: str
    session: str | None
    segment_id: int
    acquisition_id: str | None = None

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=np.int64)
        starts = np.asarray(self.sample_start_indices, dtype=np.int64)
        ends = np.asarray(self.sample_end_indices, dtype=np.int64)
        if labels.ndim != 1 or not len(labels):
            raise ValueError("state labels must be a non-empty one-dimensional array")
        if starts.shape != labels.shape or ends.shape != labels.shape:
            raise ValueError("state-label sample indices must align with labels")
        if np.any(labels < 0):
            raise ValueError("state labels cannot be negative")
        if np.any(ends < starts):
            raise ValueError("state-label end indices cannot precede start indices")
        if len(starts) > 1 and np.any(np.diff(starts) <= 0):
            raise ValueError("state-label start indices must be strictly increasing")
        if not str(self.subject).strip():
            raise ValueError("state-label subject must be non-empty")
        if self.session is not None and not str(self.session).strip():
            raise ValueError("state-label session cannot be empty")
        if self.acquisition_id is not None and not str(self.acquisition_id).strip():
            raise ValueError("state-label acquisition_id cannot be empty")
        if not isinstance(self.segment_id, (int, np.integer)) or self.segment_id < 0:
            raise ValueError("state-label segment_id must be a non-negative integer")
        object.__setattr__(self, "labels", _readonly(labels))
        object.__setattr__(self, "sample_start_indices", _readonly(starts))
        object.__setattr__(self, "sample_end_indices", _readonly(ends))
        object.__setattr__(self, "subject", str(self.subject))
        object.__setattr__(self, "session", None if self.session is None else str(self.session))
        object.__setattr__(self, "segment_id", int(self.segment_id))
        object.__setattr__(
            self,
            "acquisition_id",
            None if self.acquisition_id is None else str(self.acquisition_id),
        )


@dataclass(frozen=True)
class StateAssignments:
    sequences: tuple[StateLabelSequence, ...]
    n_states: int
    source_contract: str
    sample_interval_seconds: float | None

    def __post_init__(self) -> None:
        sequences = tuple(self.sequences)
        if not sequences:
            raise ValueError("state assignments must contain at least one sequence")
        if any(not isinstance(sequence, StateLabelSequence) for sequence in sequences):
            raise TypeError("state assignments must contain only StateLabelSequence objects")
        if not isinstance(self.n_states, (int, np.integer)) or self.n_states < 2:
            raise ValueError("n_states must be at least two")
        if any(np.any(sequence.labels >= self.n_states) for sequence in sequences):
            raise ValueError("state label exceeds n_states")
        if not str(self.source_contract).strip():
            raise ValueError("state assignment source_contract must be non-empty")
        if self.sample_interval_seconds is not None and (
            not np.isfinite(self.sample_interval_seconds) or self.sample_interval_seconds <= 0
        ):
            raise ValueError("state assignment sample interval must be finite and positive")
        identities = [
            (
                sequence.subject,
                sequence.session,
                sequence.acquisition_id,
                sequence.segment_id,
            )
            for sequence in sequences
        ]
        if len(set(identities)) != len(identities):
            raise ValueError("state assignments contain duplicate sequence identities")
        object.__setattr__(self, "sequences", sequences)
        object.__setattr__(self, "n_states", int(self.n_states))
        object.__setattr__(self, "source_contract", str(self.source_contract))


def window_fc_sequences(results: Sequence[WindowFCResult]) -> FeatureSequenceDataset:
    """Convert sliding-window outputs into one feature sequence per segment."""
    results = tuple(results)
    if not results:
        raise ValueError("at least one WindowFCResult is required")
    sequences: list[FeatureSequence] = []
    for result in results:
        if result.subject is None:
            raise ValueError("window state modeling requires subject IDs")
        keys = tuple(
            (result.roi_names[int(left)], result.roi_names[int(right)])
            for left, right in zip(result.edge_i, result.edge_j, strict=True)
        )
        contract = (
            f"window-fc:length={result.length};step={result.step};taper={result.taper}"
        )
        for segment_id in dict.fromkeys(result.segment_ids.tolist()):
            positions = np.flatnonzero(result.segment_ids == segment_id)
            sequences.append(
                FeatureSequence(
                    values=result.features[positions],
                    sample_start_indices=result.start_frames[positions],
                    sample_end_indices=result.end_frames[positions],
                    feature_keys=keys,
                    subject=result.subject,
                    session=result.session,
                    acquisition_id=result.acquisition_id,
                    segment_id=int(segment_id),
                    source_contract=contract,
                    sample_interval_seconds=(
                        None if result.tr is None else result.step * result.tr
                    ),
                )
            )
    return FeatureSequenceDataset(sequences)


def leida_sequences(results: Sequence[LEiDAResult]) -> FeatureSequenceDataset:
    """Convert LEiDA outputs into one leading-vector sequence per segment."""
    results = tuple(results)
    if not results:
        raise ValueError("at least one LEiDAResult is required")
    sequences: list[FeatureSequence] = []
    for result in results:
        if result.subject is None:
            raise ValueError("LEiDA state modeling requires subject IDs")
        keys = tuple((roi_name,) for roi_name in result.roi_names)
        contract = (
            "leida:hilbert=within-segment;"
            f"minimum-segment-length={result.minimum_segment_length};"
            f"orientation={result.orientation}"
        )
        for segment_id in dict.fromkeys(result.segment_ids.tolist()):
            positions = np.flatnonzero(result.segment_ids == segment_id)
            original = result.original_indices[positions]
            sequences.append(
                FeatureSequence(
                    values=result.leading_vectors[positions],
                    sample_start_indices=original,
                    sample_end_indices=original,
                    feature_keys=keys,
                    subject=result.subject,
                    session=result.session,
                    acquisition_id=result.acquisition_id,
                    segment_id=int(segment_id),
                    source_contract=contract,
                    sample_interval_seconds=result.tr,
                )
            )
    return FeatureSequenceDataset(sequences)


def timeseries_sequences(
    dataset: TimeSeriesDataset,
    *,
    minimum_segment_length: int = 2,
) -> FeatureSequenceDataset:
    """Convert raw ROI runs into one feature sequence per retained segment."""
    if isinstance(minimum_segment_length, (bool, np.bool_)) or not isinstance(
        minimum_segment_length, (int, np.integer)
    ):
        raise TypeError("minimum_segment_length must be an integer")
    if minimum_segment_length < 1:
        raise ValueError("minimum_segment_length must be at least one")
    dataset.require_subject_ids("ROI time-series sequence construction")
    keys = tuple((roi_name,) for roi_name in dataset.roi_names)
    sequences: list[FeatureSequence] = []
    for run in dataset.runs:
        assert run.subject is not None
        for segment_id, positions in enumerate(run.segments()):
            if len(positions) < minimum_segment_length:
                continue
            original = run.original_indices[positions]
            sequences.append(
                FeatureSequence(
                    values=run.values[positions],
                    sample_start_indices=original,
                    sample_end_indices=original,
                    feature_keys=keys,
                    subject=run.subject,
                    session=run.session,
                    acquisition_id=run.acquisition_id,
                    segment_id=segment_id,
                    source_contract=(
                        "roi-timeseries:raw-xcpd;"
                        f"minimum-segment-length={minimum_segment_length}"
                    ),
                    sample_interval_seconds=run.tr,
                )
            )
    if not sequences:
        raise ValueError("no retained segment meets minimum_segment_length")
    return FeatureSequenceDataset(sequences)
