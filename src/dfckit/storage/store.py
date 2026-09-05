"""Chunked, memory-mappable storage for feature sequences."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._arrays import readonly_copy

FeatureKey = tuple[str, ...]

if TYPE_CHECKING:
    from ..states.data import FeatureSequence, FeatureSequenceDataset

FORMAT_NAME = "dfckit-feature-store"
FORMAT_VERSION = 2


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
) -> tuple[str, str | None, str | None, int]:
    return _identity(
        str(sequence.get("subject", "")),
        sequence.get("session"),
        sequence.get("acquisition_id"),
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

    def __post_init__(self) -> None:
        values = np.asanyarray(self.values)
        starts = np.asanyarray(self.sample_start_indices, dtype=np.int64)
        ends = np.asanyarray(self.sample_end_indices, dtype=np.int64)
        if values.ndim != 2 or not len(values):
            raise ValueError("feature chunk values must be a non-empty matrix")
        if starts.shape != (len(values),) or ends.shape != (len(values),):
            raise ValueError("feature chunk sample indices do not align with rows")
        if not np.isfinite(values).all() or np.any(ends < starts):
            raise ValueError("feature chunk contains invalid values or indices")
        for name, array in (
            ("values", values), ("sample_start_indices", starts), ("sample_end_indices", ends)
        ):
            object.__setattr__(self, name, array if isinstance(array, np.memmap) else readonly_copy(array))


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
        if format_version != FORMAT_VERSION:
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
            if "acquisition_id" not in sequence:
                raise ValueError("feature-store sequence is missing acquisition_id")
            identity = _record_identity(sequence)
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
            _record_identity(sequence)
            for sequence in self._manifest["sequences"]
        )

    @property
    def sequence_sample_counts(
        self,
    ) -> tuple[tuple[tuple[str, str | None, str | None, int], int], ...]:
        """Return each sequence identity and its stored row count."""
        return tuple(
            (
                _record_identity(sequence),
                int(sequence["n_samples"]),
            )
            for sequence in self._manifest["sequences"]
        )

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
        from ..states.data import FeatureSequence

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
        from ..states.data import FeatureSequenceDataset

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
