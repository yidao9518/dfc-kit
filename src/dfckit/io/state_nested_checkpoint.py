"""Persistence for resumable nested state-count workflow contracts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

FORMAT_NAME = "dfckit-nested-state-count-checkpoint"
FORMAT_VERSION = 1


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON field in nested checkpoint: {key}")
        output[key] = value
    return output


def _invalid_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant in nested checkpoint: {value}")


def _validate(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("nested checkpoint must be a JSON object")
    expected_fields = {
        "candidate_n_states",
        "cohort_data_fingerprint",
        "feature_contract_fingerprint",
        "fit_configuration",
        "format",
        "format_version",
        "inner_validation",
        "method",
        "model_kind",
        "model_seeds",
        "outer_split",
        "sample_interval_seconds",
        "selection_policy",
        "source_contract",
        "subjects",
    }
    if set(payload) != expected_fields:
        raise ValueError("nested checkpoint fields do not match the schema")
    if payload["format"] != FORMAT_NAME or payload["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported nested checkpoint format or version")
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("nested checkpoint must contain finite JSON values") from error
    return payload


def load_nested_state_count_checkpoint(path: str | Path) -> dict[str, object]:
    """Load a strict resumable-workflow contract."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"nested checkpoint manifest does not exist: {target}")
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_invalid_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read nested checkpoint {target}: {error}") from error
    return _validate(payload)


def write_nested_state_count_checkpoint(
    payload: dict[str, object],
    path: str | Path,
) -> Path:
    """Atomically create a nested-workflow checkpoint manifest."""
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"nested checkpoint manifest already exists: {target}")
    validated = _validate(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(validated, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"nested checkpoint manifest already exists: {target}")
        os.rename(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "load_nested_state_count_checkpoint",
    "write_nested_state_count_checkpoint",
]
