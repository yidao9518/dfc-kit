"""Strict JSON persistence shared by artifact modules."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def strict_object_hook(context: str) -> Callable[[list[tuple[str, object]]], dict[str, object]]:
    """Return an object-pairs hook that rejects duplicate JSON fields."""

    def hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON field in {context}: {key}")
            output[key] = value
        return output

    return hook


def nonstandard_constant_hook(context: str) -> Callable[[str], object]:
    """Return a parse hook that rejects NaN and infinite JSON constants."""

    def hook(value: str) -> object:
        raise ValueError(f"non-standard JSON constant in {context}: {value}")

    return hook


def load_json_object(path: str | Path, *, context: str) -> dict[str, Any]:
    """Read one finite JSON object while rejecting duplicate fields."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{context} does not exist: {source}")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object_hook(context),
            parse_constant=nonstandard_constant_hook(context),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {context} {source}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def write_json_atomic(
    path: str | Path,
    payload: object,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write finite JSON, creating or replacing one regular file."""
    target = Path(path)
    if not overwrite and (target.exists() or target.is_symlink()):
        raise FileExistsError(f"JSON output already exists: {target}")
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
        if overwrite:
            os.replace(temporary, target)
        else:
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"JSON output already exists: {target}")
            os.rename(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "load_json_object",
    "nonstandard_constant_hook",
    "strict_object_hook",
    "write_json_atomic",
]
