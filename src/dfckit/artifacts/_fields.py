"""Scalar and collection validation for persisted artifact schemas."""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def artifact_integer(value: object, name: str, *, minimum: int = 0) -> int:
    """Validate an integer artifact field using schema-style errors."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def artifact_finite_float(
    value: object,
    name: str,
    *,
    positive: bool = False,
) -> float:
    """Validate a finite numeric artifact field."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    output = float(value)
    if not np.isfinite(output) or (positive and output <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return output


def artifact_integer_grid(
    value: object,
    name: str,
    *,
    minimum: int,
    minimum_count: int,
) -> tuple[int, ...]:
    """Validate a strictly increasing integer grid in an artifact."""
    if not isinstance(value, list) or len(value) < minimum_count:
        raise ValueError(f"{name} must contain at least {minimum_count} values")
    output = tuple(artifact_integer(item, name, minimum=minimum) for item in value)
    if len(set(output)) != len(output) or any(
        right <= left for left, right in pairwise(output)
    ):
        raise ValueError(f"{name} must be strictly increasing and unique")
    return output


def sample_intervals_match(left: object, right: float | None) -> bool:
    """Compare nullable positive sample intervals with fixed absolute tolerance."""
    if left is None or right is None:
        return left is None and right is None
    try:
        observed = artifact_finite_float(
            left,
            "sample_interval_seconds",
            positive=True,
        )
    except (TypeError, ValueError):
        return False
    return bool(np.isclose(observed, right, rtol=0.0, atol=1e-9))
