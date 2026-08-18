"""Chunked, memory-mappable storage for feature sequences."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .connectivity.correlation import edge_index, weighted_correlation
from .connectivity.ets import segment_standardized_samples
from .connectivity.mtd import globally_standardized_derivatives
from .connectivity.windows import SlidingWindowFC
from .data import TimeSeriesDataset, TimeSeriesRun
from .states.data import FeatureKey, FeatureSequence, FeatureSequenceDataset

FORMAT_NAME = "dfckit-feature-store"
FORMAT_VERSION = 2
READABLE_FORMAT_VERSIONS = frozenset({1, FORMAT_VERSION})


def _validated_chunk_size(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("chunk_size must be an integer")
    if value < 1:
        raise ValueError("chunk_size must be positive")
    return int(value)


def _validated_dtype(value: str | np.dtype) -> np.dtype:
    dtype = np.dtype(value)
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("feature-store dtype must be float32 or float64")
    return dtype


def _validated_feature_keys(keys: Iterable[FeatureKey]) -> tuple[FeatureKey, ...]:
    output = tuple(tuple(str(part) for part in key) for key in keys)
    if not output or any(not key for key in output):
        raise ValueError("feature_keys must contain one non-empty key per feature")
    if len(set(output)) != len(output):
        raise ValueError("feature_keys must be unique")
    return output


def _write_array(path: Path, values: NDArray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    os.replace(temporary, path)


def _write_manifest(root: Path, manifest: dict[str, object]) -> None:
    target = root / "manifest.json"
    temporary = root / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _identity(
    subject: str,
    session: str | None,
    acquisition_id: str | None,
    segment_id: int,
) -> tuple[str, str | None, str | None, int]:
    return (
        str(subject),
        None if session is None else str(session),
        None if acquisition_id is None else str(acquisition_id),
        int(segment_id),
    )


def _record_identity(
    sequence: dict[str, object],
    *,
    format_version: int,
) -> tuple[str, str | None, str | None, int]:
    return _identity(
        str(sequence.get("subject", "")),
        sequence.get("session"),
        sequence.get("acquisition_id") if format_version >= 2 else None,
        int(sequence.get("segment_id", -1)),
    )


@dataclass(frozen=True)
class StoredFeatureChunk:
    """One memory-mapped or in-memory row chunk from a stored sequence."""

    values: NDArray[np.floating]
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


class FeatureStore:
    """Directory-backed, append-only feature sequences with atomic manifests.

    One writer may append at a time. Readers can iterate memory-mapped chunks or
    reconstruct selected sequences without loading unrelated rows.
    """

    def __init__(self, root: str | Path, manifest: dict[str, object]) -> None:
        self.root = Path(root)
        self._manifest = manifest

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        feature_keys: Iterable[FeatureKey],
        source_contract: str,
        sample_interval_seconds: float | None,
        dtype: str | np.dtype = "float64",
    ) -> FeatureStore:
        path = Path(root)
        keys = _validated_feature_keys(feature_keys)
        if not str(source_contract).strip():
            raise ValueError("source_contract must be non-empty")
        if sample_interval_seconds is not None and (
            not np.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0.0
        ):
            raise ValueError("sample_interval_seconds must be finite and positive")
        normalized_dtype = _validated_dtype(dtype)
        path.mkdir(parents=True, exist_ok=False)
        (path / "chunks").mkdir()
        manifest: dict[str, object] = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "dtype": normalized_dtype.name,
            "feature_keys": [list(key) for key in keys],
            "source_contract": str(source_contract),
            "sample_interval_seconds": sample_interval_seconds,
            "n_features": len(keys),
            "n_samples": 0,
            "sequences": [],
            "chunks": [],
        }
        _write_manifest(path, manifest)
        return cls(path, manifest)

    @classmethod
    def open(cls, root: str | Path) -> FeatureStore:
        path = Path(root)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"feature-store manifest does not exist: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"cannot read feature-store manifest: {error}") from error
        cls._validate_manifest(manifest)
        return cls(path, manifest)

    @staticmethod
    def _validate_manifest(manifest: object) -> None:
        if not isinstance(manifest, dict):
            raise TypeError("feature-store manifest must be a JSON object")
        if manifest.get("format") != FORMAT_NAME:
            raise ValueError("not a dfc-kit feature store")
        format_version = manifest.get("format_version")
        if format_version not in READABLE_FORMAT_VERSIONS:
            raise ValueError("unsupported feature-store format version")
        _validated_dtype(str(manifest.get("dtype")))
        keys = _validated_feature_keys(
            tuple(tuple(key) for key in manifest.get("feature_keys", ()))
        )
        if manifest.get("n_features") != len(keys):
            raise ValueError("manifest n_features does not match feature_keys")
        if not str(manifest.get("source_contract", "")).strip():
            raise ValueError("manifest source_contract is empty")
        interval = manifest.get("sample_interval_seconds")
        if interval is not None and (
            not isinstance(interval, (int, float))
            or not np.isfinite(interval)
            or interval <= 0.0
        ):
            raise ValueError("manifest sample interval is invalid")
        sequences = manifest.get("sequences")
        chunks = manifest.get("chunks")
        if not isinstance(sequences, list) or not isinstance(chunks, list):
            raise TypeError("manifest sequences and chunks must be lists")
        identities = []
        sample_total = 0
        for index, sequence in enumerate(sequences):
            if not isinstance(sequence, dict) or sequence.get("sequence_index") != index:
                raise ValueError("manifest sequence indices are invalid")
            if int(format_version) >= 2 and "acquisition_id" not in sequence:
                raise ValueError("format v2 sequence is missing acquisition_id")
            identity = _record_identity(sequence, format_version=int(format_version))
            if not identity[0].strip() or identity[3] < 0:
                raise ValueError("manifest contains an invalid sequence identity")
            if identity[2] is not None and not identity[2].strip():
                raise ValueError("manifest contains an empty acquisition_id")
            identities.append(identity)
            n_samples = int(sequence.get("n_samples", -1))
            if n_samples < 1:
                raise ValueError("manifest sequence sample count must be positive")
            sample_total += n_samples
        if len(set(identities)) != len(identities):
            raise ValueError("manifest contains duplicate sequence identities")
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or chunk.get("chunk_id") != index:
                raise ValueError("manifest chunk indices are invalid")
        if manifest.get("n_samples") != sample_total:
            raise ValueError("manifest n_samples does not match sequence totals")

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(str(self._manifest["dtype"]))

    @property
    def feature_keys(self) -> tuple[FeatureKey, ...]:
        return tuple(tuple(key) for key in self._manifest["feature_keys"])

    @property
    def source_contract(self) -> str:
        return str(self._manifest["source_contract"])

    @property
    def sample_interval_seconds(self) -> float | None:
        value = self._manifest["sample_interval_seconds"]
        return None if value is None else float(value)

    @property
    def n_features(self) -> int:
        return int(self._manifest["n_features"])

    @property
    def n_samples(self) -> int:
        return int(self._manifest["n_samples"])

    @property
    def n_sequences(self) -> int:
        return len(self._manifest["sequences"])

    @property
    def n_chunks(self) -> int:
        return len(self._manifest["chunks"])

    @property
    def format_version(self) -> int:
        return int(self._manifest["format_version"])

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(str(sequence["subject"]) for sequence in self._manifest["sequences"])
        )

    @property
    def sequence_identities(
        self,
    ) -> tuple[tuple[str, str | None, str | None, int], ...]:
        return tuple(
            _record_identity(
                sequence,
                format_version=int(self._manifest["format_version"]),
            )
            for sequence in self._manifest["sequences"]
        )

    @property
    def sequence_sample_counts(
        self,
    ) -> tuple[tuple[tuple[str, str | None, str | None, int], int], ...]:
        """Return each sequence identity and its stored row count."""
        return tuple(
            (
                _record_identity(
                    sequence,
                    format_version=int(self._manifest["format_version"]),
                ),
                int(sequence["n_samples"]),
            )
            for sequence in self._manifest["sequences"]
        )

    def data_fingerprint(
        self,
        *,
        subjects: Iterable[str] | None = None,
        minimum_sequence_length: int = 1,
    ) -> str:
        """Hash selected sequence identities, sample indices, and feature values."""
        minimum = _validated_chunk_size(minimum_sequence_length)
        selected = None if subjects is None else tuple(dict.fromkeys(str(item) for item in subjects))
        if selected is not None:
            missing = sorted(set(selected) - set(self.subjects))
            if missing:
                raise ValueError(f"fingerprint subjects are absent from the store: {missing}")
        selected_set = None if selected is None else set(selected)
        sequence_indices = tuple(
            index
            for index, sequence in enumerate(self._manifest["sequences"])
            if (selected_set is None or sequence["subject"] in selected_set)
            and int(sequence["n_samples"]) >= minimum
        )
        if not sequence_indices:
            raise ValueError("no feature-store sequence is eligible for fingerprinting")

        digest = hashlib.sha256()
        header = {
            "format": "dfckit-feature-data-fingerprint",
            "version": 1,
            "dtype": self.dtype.str,
            "feature_keys": [list(key) for key in self.feature_keys],
            "source_contract": self.source_contract,
            "sample_interval_seconds": self.sample_interval_seconds,
        }
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        value_dtype = self.dtype.newbyteorder("<")
        index_dtype = np.dtype("<i8")
        chunks = self._manifest["chunks"]
        for sequence_index in sequence_indices:
            sequence = self._manifest["sequences"][sequence_index]
            identity = _record_identity(
                sequence,
                format_version=int(self._manifest["format_version"]),
            )
            record = {
                "identity": list(identity),
                "n_samples": int(sequence["n_samples"]),
            }
            digest.update(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            value_digest = hashlib.sha256()
            start_digest = hashlib.sha256()
            end_digest = hashlib.sha256()
            observed = 0
            for chunk_id in sequence["chunk_ids"]:
                loaded = self._load_chunk(chunks[int(chunk_id)], mmap=True)
                value_digest.update(
                    np.ascontiguousarray(loaded.values, dtype=value_dtype).tobytes(order="C")
                )
                start_digest.update(
                    np.ascontiguousarray(
                        loaded.sample_start_indices,
                        dtype=index_dtype,
                    ).tobytes(order="C")
                )
                end_digest.update(
                    np.ascontiguousarray(
                        loaded.sample_end_indices,
                        dtype=index_dtype,
                    ).tobytes(order="C")
                )
                observed += len(loaded.values)
            if observed != int(sequence["n_samples"]):
                raise ValueError("feature-store sequence chunks do not match its sample count")
            digest.update(value_digest.digest())
            digest.update(start_digest.digest())
            digest.update(end_digest.digest())
        return digest.hexdigest()

    def feature_contract_fingerprint(self) -> str:
        """Hash the exact stored feature identity and numeric representation."""
        contract = {
            "format": "dfckit-feature-contract-fingerprint",
            "version": 1,
            "dtype": self.dtype.str,
            "feature_keys": [list(key) for key in self.feature_keys],
            "source_contract": self.source_contract,
            "sample_interval_seconds": self.sample_interval_seconds,
        }
        return hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def require_contract(
        self,
        *,
        feature_keys: Iterable[FeatureKey],
        source_contract: str,
        sample_interval_seconds: float | None,
    ) -> None:
        if _validated_feature_keys(feature_keys) != self.feature_keys:
            raise ValueError("feature store and appended data use different feature identities")
        if str(source_contract) != self.source_contract:
            raise ValueError("feature store and appended data use different source contracts")
        stored = self.sample_interval_seconds
        intervals_match = (stored is None and sample_interval_seconds is None) or (
            stored is not None
            and sample_interval_seconds is not None
            and np.isclose(stored, sample_interval_seconds, rtol=0.0, atol=1e-9)
        )
        if not intervals_match:
            raise ValueError("feature store and appended data use different sample intervals")

    def append_sequence(
        self,
        sequence: FeatureSequence,
        *,
        chunk_size: int = 4096,
    ) -> None:
        size = _validated_chunk_size(chunk_size)
        self.require_contract(
            feature_keys=sequence.feature_keys,
            source_contract=sequence.source_contract,
            sample_interval_seconds=sequence.sample_interval_seconds,
        )
        parts = (
            (
                sequence.values[start : start + size],
                sequence.sample_start_indices[start : start + size],
                sequence.sample_end_indices[start : start + size],
            )
            for start in range(0, sequence.n_samples, size)
        )
        self.append_sequence_parts(
            parts,
            subject=sequence.subject,
            session=sequence.session,
            acquisition_id=sequence.acquisition_id,
            segment_id=sequence.segment_id,
        )

    def append_dataset(
        self,
        dataset: FeatureSequenceDataset,
        *,
        chunk_size: int = 4096,
    ) -> None:
        self.require_contract(
            feature_keys=dataset.feature_keys,
            source_contract=dataset.source_contract,
            sample_interval_seconds=dataset.sample_interval_seconds,
        )
        for sequence in dataset.sequences:
            self.append_sequence(sequence, chunk_size=chunk_size)

    def append_sequence_parts(
        self,
        parts: Iterable[tuple[ArrayLike, ArrayLike, ArrayLike]],
        *,
        subject: str,
        session: str | None,
        segment_id: int,
        acquisition_id: str | None = None,
    ) -> None:
        identity = _identity(subject, session, acquisition_id, segment_id)
        if not identity[0].strip() or (identity[1] is not None and not identity[1].strip()):
            raise ValueError("subject and non-null session identifiers must be non-empty")
        if identity[2] is not None and not identity[2].strip():
            raise ValueError("acquisition_id must be non-empty when provided")
        if identity[3] < 0:
            raise ValueError("segment_id must be non-negative")
        if identity in self.sequence_identities:
            raise ValueError(f"feature store already contains sequence identity {identity}")

        sequence_index = self.n_sequences
        next_chunk_id = self.n_chunks
        new_chunks: list[dict[str, object]] = []
        written_paths: list[Path] = []
        total = 0
        previous_start: int | None = None
        try:
            for raw_values, raw_starts, raw_ends in parts:
                values = np.asarray(raw_values, dtype=self.dtype)
                starts = np.asarray(raw_starts, dtype=np.int64)
                ends = np.asarray(raw_ends, dtype=np.int64)
                if values.ndim != 2 or not len(values) or values.shape[1] != self.n_features:
                    raise ValueError("each chunk must be a non-empty rows-by-features array")
                if starts.shape != (len(values),) or ends.shape != (len(values),):
                    raise ValueError("chunk sample indices must align with feature rows")
                if not np.isfinite(values).all():
                    raise ValueError("chunk feature values contain non-finite samples")
                if np.any(ends < starts) or (len(starts) > 1 and np.any(np.diff(starts) <= 0)):
                    raise ValueError("chunk sample indices are invalid or not increasing")
                if previous_start is not None and int(starts[0]) <= previous_start:
                    raise ValueError("sample starts must increase strictly across chunks")
                previous_start = int(starts[-1])

                chunk_id = next_chunk_id + len(new_chunks)
                prefix = f"chunks/{chunk_id:08d}"
                files = {
                    "values_file": f"{prefix}.values.npy",
                    "starts_file": f"{prefix}.starts.npy",
                    "ends_file": f"{prefix}.ends.npy",
                }
                for key, array in (
                    ("values_file", values),
                    ("starts_file", starts),
                    ("ends_file", ends),
                ):
                    path = self.root / files[key]
                    _write_array(path, array)
                    written_paths.append(path)
                new_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "sequence_index": sequence_index,
                        "start_in_sequence": total,
                        "stop_in_sequence": total + len(values),
                        **files,
                    }
                )
                total += len(values)
            if total == 0:
                raise ValueError("a stored sequence must contain at least one chunk")

            new_manifest = copy.deepcopy(self._manifest)
            if int(new_manifest["format_version"]) == 1:
                new_manifest["format_version"] = FORMAT_VERSION
                for existing in new_manifest["sequences"]:
                    existing["acquisition_id"] = None
            new_manifest["chunks"].extend(new_chunks)
            new_manifest["sequences"].append(
                {
                    "sequence_index": sequence_index,
                    "subject": identity[0],
                    "session": identity[1],
                    "acquisition_id": identity[2],
                    "segment_id": identity[3],
                    "n_samples": total,
                    "chunk_ids": [chunk["chunk_id"] for chunk in new_chunks],
                }
            )
            new_manifest["n_samples"] = self.n_samples + total
            _write_manifest(self.root, new_manifest)
            self._manifest = new_manifest
        except Exception:
            for path in written_paths:
                path.unlink(missing_ok=True)
            raise

    def _load_chunk(self, record: dict[str, object], *, mmap: bool) -> StoredFeatureChunk:
        mode = "r" if mmap else None
        try:
            values = np.load(self.root / record["values_file"], mmap_mode=mode, allow_pickle=False)
            starts = np.load(self.root / record["starts_file"], mmap_mode=mode, allow_pickle=False)
            ends = np.load(self.root / record["ends_file"], mmap_mode=mode, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot load feature chunk {record['chunk_id']}: {error}") from error
        expected = int(record["stop_in_sequence"]) - int(record["start_in_sequence"])
        if values.shape != (expected, self.n_features):
            raise ValueError(f"feature chunk {record['chunk_id']} has an invalid value shape")
        if starts.shape != (expected,) or ends.shape != (expected,):
            raise ValueError(f"feature chunk {record['chunk_id']} has invalid sample indices")
        sequence = self._manifest["sequences"][int(record["sequence_index"])]
        return StoredFeatureChunk(
            values=values,
            sample_start_indices=starts,
            sample_end_indices=ends,
            subject=str(sequence["subject"]),
            session=sequence["session"],
            acquisition_id=sequence.get("acquisition_id"),
            segment_id=int(sequence["segment_id"]),
            sequence_index=int(record["sequence_index"]),
            chunk_id=int(record["chunk_id"]),
            start_in_sequence=int(record["start_in_sequence"]),
            stop_in_sequence=int(record["stop_in_sequence"]),
        )

    def iter_chunks(
        self,
        *,
        subjects: Iterable[str] | None = None,
        mmap: bool = True,
    ) -> Iterator[StoredFeatureChunk]:
        selected = None if subjects is None else {str(subject) for subject in subjects}
        for record in self._manifest["chunks"]:
            sequence = self._manifest["sequences"][int(record["sequence_index"])]
            if selected is not None and sequence["subject"] not in selected:
                continue
            yield self._load_chunk(record, mmap=mmap)

    def read_sequence(
        self,
        subject: str,
        session: str | None,
        segment_id: int,
        *,
        acquisition_id: str | None = None,
        sample_slice: slice | None = None,
    ) -> FeatureSequence:
        identity = _identity(subject, session, acquisition_id, segment_id)
        try:
            sequence_index = self.sequence_identities.index(identity)
        except ValueError as error:
            raise KeyError(f"feature store has no sequence identity {identity}") from error
        sequence = self._manifest["sequences"][sequence_index]
        n_samples = int(sequence["n_samples"])
        requested = slice(None) if sample_slice is None else sample_slice
        start, stop, step = requested.indices(n_samples)
        if step != 1:
            raise ValueError("sample_slice step must equal one")
        if stop <= start:
            raise ValueError("sample_slice must select at least one sample")

        values_parts = []
        start_parts = []
        end_parts = []
        chunks = self._manifest["chunks"]
        for chunk_id in sequence["chunk_ids"]:
            record = chunks[int(chunk_id)]
            chunk_start = int(record["start_in_sequence"])
            chunk_stop = int(record["stop_in_sequence"])
            overlap_start = max(start, chunk_start)
            overlap_stop = min(stop, chunk_stop)
            if overlap_stop <= overlap_start:
                continue
            loaded = self._load_chunk(record, mmap=True)
            local = slice(overlap_start - chunk_start, overlap_stop - chunk_start)
            values_parts.append(loaded.values[local])
            start_parts.append(loaded.sample_start_indices[local])
            end_parts.append(loaded.sample_end_indices[local])
        return FeatureSequence(
            values=np.concatenate(values_parts, axis=0),
            sample_start_indices=np.concatenate(start_parts),
            sample_end_indices=np.concatenate(end_parts),
            feature_keys=self.feature_keys,
            subject=identity[0],
            session=identity[1],
            acquisition_id=identity[2],
            segment_id=identity[3],
            source_contract=self.source_contract,
            sample_interval_seconds=self.sample_interval_seconds,
        )

    def read_dataset(
        self,
        *,
        subjects: Iterable[str] | None = None,
    ) -> FeatureSequenceDataset:
        selected = None if subjects is None else {str(subject) for subject in subjects}
        sequences = [
            self.read_sequence(
                identity[0],
                identity[1],
                identity[3],
                acquisition_id=identity[2],
            )
            for identity in self.sequence_identities
            if selected is None or identity[0] in selected
        ]
        if not sequences:
            raise ValueError("no stored feature sequences match the requested subjects")
        return FeatureSequenceDataset(sequences)


def _edge_feature_keys(roi_names: Sequence[str]) -> tuple[FeatureKey, ...]:
    left, right = edge_index(len(roi_names))
    return tuple(
        (roi_names[int(i)], roi_names[int(j)])
        for i, j in zip(left, right, strict=True)
    )


def _roi_feature_keys(roi_names: Sequence[str]) -> tuple[FeatureKey, ...]:
    return tuple((str(name),) for name in roi_names)


def _window_fc_contract(estimator: SlidingWindowFC) -> str:
    return f"window-fc:length={estimator.length};step={estimator.step};taper={estimator.taper}"


def append_window_fc(
    store: FeatureStore,
    run: TimeSeriesRun,
    estimator: SlidingWindowFC,
    *,
    chunk_size: int = 128,
) -> None:
    """Compute and append sliding-window FC without materializing the full result."""
    if run.subject is None:
        raise ValueError("stored window FC requires a subject identifier")
    size = _validated_chunk_size(chunk_size)
    keys = _edge_feature_keys(run.roi_names)
    interval = None if run.tr is None else estimator.step * run.tr
    store.require_contract(
        feature_keys=keys,
        source_contract=_window_fc_contract(estimator),
        sample_interval_seconds=interval,
    )
    edge_i, edge_j = edge_index(run.n_rois)
    weights = estimator._weights()
    appended = 0
    for segment_id, positions in enumerate(run.segments()):
        n_windows = max(0, (len(positions) - estimator.length) // estimator.step + 1)
        if not n_windows:
            continue

        def parts(
            segment_positions: NDArray[np.int64] = positions,
            window_count: int = n_windows,
        ) -> Iterator[tuple[NDArray, NDArray, NDArray]]:
            for first in range(0, window_count, size):
                last = min(first + size, window_count)
                features = []
                starts = np.empty(last - first, dtype=np.int64)
                ends = np.empty(last - first, dtype=np.int64)
                for output_index, window_index in enumerate(range(first, last)):
                    start = window_index * estimator.step
                    selected = segment_positions[start : start + estimator.length]
                    correlation = weighted_correlation(run.values[selected], weights)
                    features.append(
                        np.arctanh(
                            np.clip(correlation[edge_i, edge_j], -0.999999, 0.999999)
                        )
                    )
                    original = run.original_indices[selected]
                    starts[output_index] = original[0]
                    ends[output_index] = original[-1]
                yield np.stack(features), starts, ends

        store.append_sequence_parts(
            parts(),
            subject=run.subject,
            session=run.session,
            segment_id=segment_id,
            acquisition_id=run.acquisition_id,
        )
        appended += n_windows
    if not appended:
        raise ValueError("no contiguous segment is long enough for the requested window")


def write_window_fc_store(
    root: str | Path,
    runs: Sequence[TimeSeriesRun],
    estimator: SlidingWindowFC,
    *,
    chunk_size: int = 128,
    dtype: str | np.dtype = "float64",
) -> FeatureStore:
    """Create a store and stream sliding-window FC from compatible runs into it."""
    dataset = TimeSeriesDataset(runs)
    dataset.require_subject_ids("stored window FC")
    store = FeatureStore.create(
        root,
        feature_keys=_edge_feature_keys(dataset.roi_names),
        source_contract=_window_fc_contract(estimator),
        sample_interval_seconds=None if dataset.tr is None else estimator.step * dataset.tr,
        dtype=dtype,
    )
    for run in dataset.runs:
        append_window_fc(store, run, estimator, chunk_size=chunk_size)
    return store


def append_mtd(
    store: FeatureStore,
    run: TimeSeriesRun,
    *,
    chunk_size: int = 128,
) -> None:
    """Compute and append run-standardized MTD in bounded row chunks."""
    if run.subject is None:
        raise ValueError("stored MTD requires a subject identifier")
    size = _validated_chunk_size(chunk_size)
    keys = _edge_feature_keys(run.roi_names)
    contract = "mtd:difference=within-segment;normalization=run"
    store.require_contract(
        feature_keys=keys,
        source_contract=contract,
        sample_interval_seconds=run.tr,
    )
    derivatives, starts, ends, segment_ids = globally_standardized_derivatives(run)
    edge_i, edge_j = edge_index(run.n_rois)
    for segment_id in dict.fromkeys(segment_ids.tolist()):
        positions = np.flatnonzero(segment_ids == segment_id)

        def parts(
            segment_positions: NDArray[np.int64] = positions,
        ) -> Iterator[tuple[NDArray, NDArray, NDArray]]:
            for first in range(0, len(segment_positions), size):
                selected = segment_positions[first : first + size]
                values = derivatives[selected]
                yield values[:, edge_i] * values[:, edge_j], starts[selected], ends[selected]

        store.append_sequence_parts(
            parts(),
            subject=run.subject,
            session=run.session,
            segment_id=int(segment_id),
            acquisition_id=run.acquisition_id,
        )


def write_mtd_store(
    root: str | Path,
    runs: Sequence[TimeSeriesRun],
    *,
    chunk_size: int = 128,
    dtype: str | np.dtype = "float64",
) -> FeatureStore:
    """Create a store and stream run-standardized MTD from compatible runs."""
    dataset = TimeSeriesDataset(runs)
    dataset.require_subject_ids("stored MTD")
    store = FeatureStore.create(
        root,
        feature_keys=_edge_feature_keys(dataset.roi_names),
        source_contract="mtd:difference=within-segment;normalization=run",
        sample_interval_seconds=dataset.tr,
        dtype=dtype,
    )
    for run in dataset.runs:
        append_mtd(store, run, chunk_size=chunk_size)
    return store


def append_cap(
    store: FeatureStore,
    run: TimeSeriesRun,
    *,
    chunk_size: int = 128,
) -> None:
    """Append segment-standardized instantaneous ROI patterns in bounded chunks."""
    if run.subject is None:
        raise ValueError("stored CAP requires a subject identifier")
    size = _validated_chunk_size(chunk_size)
    keys = _roi_feature_keys(run.roi_names)
    contract = "cap:within-segment-roi-zscore-ddof0"
    store.require_contract(
        feature_keys=keys,
        source_contract=contract,
        sample_interval_seconds=run.tr,
    )
    standardized, original, segment_ids = segment_standardized_samples(
        run, method_name="CAP"
    )
    for segment_id in dict.fromkeys(segment_ids.tolist()):
        positions = np.flatnonzero(segment_ids == segment_id)

        def parts(
            segment_positions: NDArray[np.int64] = positions,
        ) -> Iterator[tuple[NDArray, NDArray, NDArray]]:
            for first in range(0, len(segment_positions), size):
                selected = segment_positions[first : first + size]
                indices = original[selected]
                yield standardized[selected], indices, indices

        store.append_sequence_parts(
            parts(),
            subject=run.subject,
            session=run.session,
            segment_id=int(segment_id),
            acquisition_id=run.acquisition_id,
        )


def write_cap_store(
    root: str | Path,
    runs: Sequence[TimeSeriesRun],
    *,
    chunk_size: int = 128,
    dtype: str | np.dtype = "float64",
) -> FeatureStore:
    """Create a store of segment-standardized instantaneous ROI patterns."""
    dataset = TimeSeriesDataset(runs)
    dataset.require_subject_ids("stored CAP")
    store = FeatureStore.create(
        root,
        feature_keys=_roi_feature_keys(dataset.roi_names),
        source_contract="cap:within-segment-roi-zscore-ddof0",
        sample_interval_seconds=dataset.tr,
        dtype=dtype,
    )
    for run in dataset.runs:
        append_cap(store, run, chunk_size=chunk_size)
    return store


def append_ets(
    store: FeatureStore,
    run: TimeSeriesRun,
    *,
    chunk_size: int = 128,
) -> None:
    """Compute and append complete ETS in bounded row chunks."""
    if run.subject is None:
        raise ValueError("stored ETS requires a subject identifier")
    size = _validated_chunk_size(chunk_size)
    keys = _edge_feature_keys(run.roi_names)
    store.require_contract(
        feature_keys=keys,
        source_contract="ets:normalization=segment",
        sample_interval_seconds=run.tr,
    )
    standardized, original, segment_ids = segment_standardized_samples(run)
    edge_i, edge_j = edge_index(run.n_rois)
    for segment_id in dict.fromkeys(segment_ids.tolist()):
        positions = np.flatnonzero(segment_ids == segment_id)

        def parts(
            segment_positions: NDArray[np.int64] = positions,
        ) -> Iterator[tuple[NDArray, NDArray, NDArray]]:
            for first in range(0, len(segment_positions), size):
                selected = segment_positions[first : first + size]
                values = standardized[selected]
                features = values[:, edge_i] * values[:, edge_j]
                indices = original[selected]
                yield features, indices, indices

        store.append_sequence_parts(
            parts(),
            subject=run.subject,
            session=run.session,
            segment_id=int(segment_id),
            acquisition_id=run.acquisition_id,
        )


def write_ets_store(
    root: str | Path,
    runs: Sequence[TimeSeriesRun],
    *,
    chunk_size: int = 128,
    dtype: str | np.dtype = "float64",
) -> FeatureStore:
    """Create a store and stream complete ETS from compatible runs into it."""
    dataset = TimeSeriesDataset(runs)
    dataset.require_subject_ids("stored ETS")
    store = FeatureStore.create(
        root,
        feature_keys=_edge_feature_keys(dataset.roi_names),
        source_contract="ets:normalization=segment",
        sample_interval_seconds=dataset.tr,
        dtype=dtype,
    )
    for run in dataset.runs:
        append_ets(store, run, chunk_size=chunk_size)
    return store
