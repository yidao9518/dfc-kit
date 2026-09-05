"""Run-level held-out scores for fitted state models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


def _validate_run(
    *,
    subject: str,
    session: str | None,
    acquisition_id: str | None,
    n_samples: int,
    n_sequences: int,
) -> None:
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("state-model score subject must be non-empty")
    if session is not None and (not isinstance(session, str) or not session.strip()):
        raise ValueError("state-model score session must be null or non-empty")
    if acquisition_id is not None and (
        not isinstance(acquisition_id, str) or not acquisition_id.strip()
    ):
        raise ValueError("state-model score acquisition_id must be null or non-empty")
    for name, value in (("n_samples", n_samples), ("n_sequences", n_sequences)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"state-model score {name} must be an integer")
        if value < 1:
            raise ValueError(f"state-model score {name} must be positive")
    if n_sequences > n_samples:
        raise ValueError("state-model score cannot contain more sequences than samples")


def _finite(value: float, name: str) -> float:
    output = float(value)
    if not np.isfinite(output):
        raise ValueError(f"state-model score {name} must be finite")
    return output


@dataclass(frozen=True)
class RunKMeansScore:
    """KMeans quantization error for one held-out acquisition.

    Distances are squared Euclidean distances in the model's frozen,
    training-standardized feature space. ``mean_squared_distance`` divides
    the total by samples, not by samples times features.
    """

    subject: str
    session: str | None
    acquisition_id: str | None
    n_samples: int
    n_sequences: int
    total_squared_distance: float

    def __post_init__(self) -> None:
        _validate_run(
            subject=self.subject,
            session=self.session,
            acquisition_id=self.acquisition_id,
            n_samples=self.n_samples,
            n_sequences=self.n_sequences,
        )
        total = _finite(self.total_squared_distance, "total_squared_distance")
        if total < 0.0:
            raise ValueError("KMeans squared distances must be non-negative")
        object.__setattr__(self, "total_squared_distance", total)

    @property
    def mean_squared_distance(self) -> float:
        return self.total_squared_distance / self.n_samples


@dataclass(frozen=True)
class RunGaussianHMMScore:
    """Gaussian-HMM log likelihood for one held-out acquisition.

    Every censor-bounded sequence starts from the fitted initial-state
    distribution. Run likelihood is the sum across those independent
    sequences, so no transition is introduced across a censor gap.
    """

    subject: str
    session: str | None
    acquisition_id: str | None
    n_samples: int
    n_sequences: int
    log_likelihood: float

    def __post_init__(self) -> None:
        _validate_run(
            subject=self.subject,
            session=self.session,
            acquisition_id=self.acquisition_id,
            n_samples=self.n_samples,
            n_sequences=self.n_sequences,
        )
        total = _finite(self.log_likelihood, "log_likelihood")
        object.__setattr__(self, "log_likelihood", total)

    @property
    def log_likelihood_per_sample(self) -> float:
        return self.log_likelihood / self.n_samples


RunStateModelScore: TypeAlias = RunKMeansScore | RunGaussianHMMScore


__all__ = ["RunGaussianHMMScore", "RunKMeansScore", "RunStateModelScore"]
