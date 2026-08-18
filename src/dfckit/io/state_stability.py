"""Machine-readable persistence for aligned repeated-fit state stability."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..states.alignment import StateAlignment
from ..states.stability import RunStateStability

FORMAT_NAME = "dfckit-state-stability"
FORMAT_VERSION = 1


def _fingerprint(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal string")
    return value


def _seed(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    output = float(value)
    if not np.isfinite(output) or output <= 0.0:
        raise ValueError("sample_interval_seconds must be finite and positive")
    return output


def _json_values(values: NDArray[np.float64]) -> object:
    array = np.asarray(values, dtype=float)
    if np.isinf(array).any():
        raise ValueError("state stability metrics cannot contain infinite values")
    if array.ndim == 0:
        return None if np.isnan(array) else float(array)
    return [_json_values(item) for item in array]


def _aggregate(values: NDArray[np.float64]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or array.shape[0] < 2:
        raise ValueError("state stability aggregation requires at least two fitted values")
    if np.isinf(array).any():
        raise ValueError("state stability metrics cannot contain infinite values")
    finite = np.isfinite(array)
    counts = finite.sum(axis=0)
    total = np.where(finite, array, 0.0).sum(axis=0)
    mean = np.divide(
        total,
        counts,
        out=np.full(counts.shape, np.nan, dtype=float),
        where=counts > 0,
    )
    centered = np.where(finite, array - mean, 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=0),
        counts,
        out=np.full(counts.shape, np.nan, dtype=float),
        where=counts > 1,
    )
    standard_deviation = np.sqrt(variance)
    return {
        "by_fit": _json_values(array),
        "mean": _json_values(mean),
        "standard_deviation": _json_values(standard_deviation),
        "valid_fit_count": np.asarray(counts, dtype=int).tolist(),
    }


def _fit_records(
    *,
    n_states: int,
    reference_model_fingerprint: str,
    reference_seed: int,
    candidate_model_fingerprints: Sequence[str],
    candidate_seeds: Sequence[int],
    alignments: Sequence[StateAlignment],
) -> list[dict[str, object]]:
    fingerprints = (
        _fingerprint(reference_model_fingerprint, "reference_model_fingerprint"),
        *(
            _fingerprint(value, "candidate_model_fingerprint")
            for value in candidate_model_fingerprints
        ),
    )
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("state stability model fingerprints must be unique")
    seeds = (
        _seed(reference_seed, "reference_seed"),
        *(_seed(value, "candidate_seed") for value in candidate_seeds),
    )
    alignments = tuple(alignments)
    if len(fingerprints) < 2 or len(seeds) != len(fingerprints):
        raise ValueError("state stability requires matching identities for at least two fits")
    if len(alignments) != len(fingerprints) - 1:
        raise ValueError("state stability alignment count does not match candidate fits")
    records = [
        {
            "fit_index": 0,
            "role": "reference",
            "model_fingerprint": fingerprints[0],
            "model_seed": seeds[0],
            "candidate_to_reference": list(range(n_states)),
            "matched_correlations": [1.0] * n_states,
            "correlation_matrix": np.eye(n_states).tolist(),
            "alignment_method": "identity reference numbering",
        }
    ]
    for index, (fingerprint, seed, alignment) in enumerate(
        zip(fingerprints[1:], seeds[1:], alignments, strict=True),
        start=1,
    ):
        if len(alignment.candidate_to_reference) != n_states:
            raise ValueError("state stability alignment does not match n_states")
        if alignment.reference_seed != seeds[0] or alignment.candidate_seed != seed:
            raise ValueError("state stability alignment seed identity is inconsistent")
        records.append(
            {
                "fit_index": index,
                "role": "candidate",
                "model_fingerprint": fingerprint,
                "model_seed": seed,
                "candidate_to_reference": alignment.candidate_to_reference.tolist(),
                "matched_correlations": alignment.matched_correlations.tolist(),
                "correlation_matrix": alignment.correlation_matrix.tolist(),
                "alignment_method": alignment.method,
            }
        )
    return records


def state_stability_payload(
    runs: Sequence[RunStateStability],
    *,
    model_kind: str,
    reference_model_fingerprint: str,
    reference_seed: int,
    candidate_model_fingerprints: Sequence[str],
    candidate_seeds: Sequence[int],
    alignments: Sequence[StateAlignment],
    training_data_fingerprint: str,
    source_contract: str,
    sample_interval_seconds: float | None,
    allow_fit_subjects: bool,
) -> dict[str, object]:
    """Build a JSON-safe report with reference-numbered metrics for every fit."""
    runs = tuple(runs)
    if not runs:
        raise ValueError("state stability report requires at least one run")
    if model_kind not in {"kmeans-state", "gaussian-hmm-state"}:
        raise ValueError("state stability model_kind is invalid")
    if not str(source_contract).strip():
        raise ValueError("state stability source_contract must be non-empty")
    if not isinstance(allow_fit_subjects, (bool, np.bool_)):
        raise TypeError("allow_fit_subjects must be boolean")
    interval = _optional_float(sample_interval_seconds)
    training_fingerprint = _fingerprint(
        training_data_fingerprint,
        "training_data_fingerprint",
    )
    n_fits = runs[0].n_fits
    n_states = runs[0].n_states
    if any(run.n_fits != n_fits or run.n_states != n_states for run in runs):
        raise ValueError("state stability runs must have identical fit and state counts")
    fit_records = _fit_records(
        n_states=n_states,
        reference_model_fingerprint=reference_model_fingerprint,
        reference_seed=reference_seed,
        candidate_model_fingerprints=candidate_model_fingerprints,
        candidate_seeds=candidate_seeds,
        alignments=alignments,
    )
    if len(fit_records) != n_fits:
        raise ValueError("state stability fit identities do not match metric fit count")
    run_records = []
    for run in runs:
        run_records.append(
            {
                "subject": run.subject,
                "session": run.session,
                "acquisition_id": run.acquisition_id,
                "n_samples": run.n_samples,
                "n_sequences": run.n_sequences,
                "occupancy": _aggregate(run.occupancy_by_fit),
                "mean_dwell_samples": _aggregate(run.mean_dwell_samples_by_fit),
                "mean_dwell_seconds": _aggregate(run.mean_dwell_seconds_by_fit),
                "switch_rate": _aggregate(run.switch_rate_by_fit),
                "transition_probabilities": _aggregate(
                    run.transition_probabilities_by_fit
                ),
            }
        )
    subjects = list(dict.fromkeys(run.subject for run in runs))
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model_kind": model_kind,
        "reference_model_fingerprint": fit_records[0]["model_fingerprint"],
        "reference_seed": fit_records[0]["model_seed"],
        "state_numbering": "reference-model state indices",
        "dispersion_standard_deviation": (
            "population standard deviation across fits (ddof=0); null when fewer "
            "than two finite values are available"
        ),
        "n_fits": n_fits,
        "n_states": n_states,
        "n_runs": len(runs),
        "n_sequences": sum(run.n_sequences for run in runs),
        "n_samples": sum(run.n_samples for run in runs),
        "subjects": subjects,
        "training_data_fingerprint": training_fingerprint,
        "source_contract": str(source_contract),
        "sample_interval_seconds": interval,
        "allow_fit_subjects": bool(allow_fit_subjects),
        "fits": fit_records,
        "runs": run_records,
    }


def write_state_stability(
    runs: Sequence[RunStateStability],
    path: str | Path,
    **metadata: object,
) -> Path:
    """Atomically write an aligned repeated-fit stability report as strict JSON."""
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"state-stability output path already exists: {target}")
    payload = state_stability_payload(runs, **metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"state-stability output path already exists: {target}")
        os.rename(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "state_stability_payload",
    "write_state_stability",
]
