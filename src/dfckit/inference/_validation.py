"""Shared validation helpers for inferential units."""

from __future__ import annotations

from collections.abc import Sequence


def validated_subject_ids(subject_ids: Sequence[str], n_observations: int) -> tuple[str, ...]:
    identifiers = tuple(str(subject) for subject in subject_ids)
    if len(identifiers) != n_observations:
        raise ValueError("subject_ids must match the number of observations")
    if any(not subject.strip() for subject in identifiers):
        raise ValueError("subject_ids cannot contain empty identifiers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("subject_ids must contain one unique entry per participant")
    return identifiers
