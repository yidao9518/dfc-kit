"""Internal validation shared across method families."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


def validated_integer(value: object, *, label: str, minimum: int = 0) -> int:
    """Return a Python integer after type and lower-bound validation."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return int(value)


def validated_seed(value: object, *, label: str) -> int:
    """Return a non-negative Python integer suitable as a random seed."""
    return validated_integer(value, label=label, minimum=0)


def validated_positive_integer(value: object, label: str) -> int:
    """Validate an integer using the legacy positive-value error contract."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return int(value)


def validated_nonnegative_integer(value: object, label: str) -> int:
    """Validate an integer using the legacy non-negative error contract."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return int(value)


def validated_roi_indices(
    nodes: Iterable[int],
    *,
    n_rois: int,
    label: str,
    minimum: int = 1,
) -> NDArray[np.int64]:
    """Validate unique integer ROI indices against a known ROI axis."""
    raw = tuple(nodes)
    if len(raw) < minimum:
        quantity = "one ROI index" if minimum == 1 else f"{minimum} ROI indices"
        raise ValueError(f"{label} must contain at least {quantity}")
    if any(
        isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer))
        for node in raw
    ):
        raise TypeError(f"{label} must contain integer ROI indices")
    output = np.asarray(raw, dtype=np.int64)
    if len(set(output.tolist())) != len(output):
        raise ValueError(f"{label} contains duplicate ROI indices")
    if np.any(output < 0) or np.any(output >= n_rois):
        raise ValueError(f"{label} contains an ROI index outside [0, {n_rois})")
    return output


def validated_subject_labels(subjects: Iterable[str], *, n_observations: int) -> tuple[str, ...]:
    """Validate one non-empty subject label per observation, allowing repeats."""
    output = tuple(str(subject) for subject in subjects)
    if len(output) != n_observations:
        raise ValueError("subjects must contain one identifier per observation")
    if any(not subject.strip() for subject in output):
        raise ValueError("subject identifiers must be non-empty")
    return output


def validated_subject_ids(subject_ids: Sequence[str], n_observations: int) -> tuple[str, ...]:
    """Validate one unique, non-empty participant ID per observation."""
    identifiers = tuple(str(subject) for subject in subject_ids)
    if len(identifiers) != n_observations:
        raise ValueError("subject_ids must match the number of observations")
    if any(not subject.strip() for subject in identifiers):
        raise ValueError("subject_ids cannot contain empty identifiers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("subject_ids must contain one unique entry per participant")
    return identifiers
