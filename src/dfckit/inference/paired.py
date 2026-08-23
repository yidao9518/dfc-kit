"""Participant-level paired randomization and bootstrap inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .._validation import validated_subject_ids


@dataclass(frozen=True)
class PairedSignFlipResult:
    estimate: float
    pvalue: float
    alternative: str
    subject_ids: tuple[str, ...]
    permutation_unit: str
    n_permutations_requested: int
    n_permutations_performed: int
    exact: bool
    seed: int | None
    null_mean: float
    null_sd: float


@dataclass(frozen=True)
class BootstrapCIResult:
    estimate: float
    lower: float
    upper: float
    confidence: float
    subject_ids: tuple[str, ...]
    resampling_unit: str
    n_resamples: int
    seed: int
    method: str = "percentile"


def _paired_values(
    differences: ArrayLike,
    subject_ids: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(
            "paired differences must be one-dimensional with at least two participants"
        )
    if not np.isfinite(values).all():
        raise ValueError("paired differences contain non-finite values")
    return values, validated_subject_ids(subject_ids, len(values))


def _extreme_count(null: np.ndarray, observed: float, alternative: str) -> int:
    if alternative == "two-sided":
        return int(np.count_nonzero(np.abs(null) >= abs(observed)))
    if alternative == "greater":
        return int(np.count_nonzero(null >= observed))
    if alternative == "less":
        return int(np.count_nonzero(null <= observed))
    raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")


def paired_sign_flip(
    differences: ArrayLike,
    subject_ids: Sequence[str],
    *,
    n_permutations: int = 10_000,
    seed: int | None = None,
    alternative: str = "two-sided",
    exact: bool = False,
) -> PairedSignFlipResult:
    """Test a paired mean by independently changing each participant's sign."""
    values, identifiers = _paired_values(differences, subject_ids)
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if exact and len(values) > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 participants")
    if not exact and seed is None:
        raise ValueError("Monte Carlo sign-flip testing requires an explicit seed")

    observed = float(values.mean())
    extreme = 0
    total = 2 ** len(values) if exact else int(n_permutations)
    null_sum = 0.0
    null_sum_squares = 0.0
    rng = None if exact else np.random.default_rng(seed)

    for start in range(0, total, 4096):
        stop = min(start + 4096, total)
        if exact:
            codes = np.arange(start, stop, dtype=np.uint64)[:, None]
            bits = (codes >> np.arange(len(values), dtype=np.uint64)) & 1
            signs = np.where(bits == 1, 1.0, -1.0)
        else:
            assert rng is not None
            signs = rng.choice((-1.0, 1.0), size=(stop - start, len(values)))
        null = (signs * values).mean(axis=1)
        extreme += _extreme_count(null, observed, alternative)
        null_sum += float(null.sum())
        null_sum_squares += float(np.square(null).sum())

    pvalue = extreme / total if exact else (extreme + 1.0) / (total + 1.0)
    null_mean = null_sum / total
    null_variance = max(null_sum_squares / total - null_mean**2, 0.0)
    return PairedSignFlipResult(
        estimate=observed,
        pvalue=float(pvalue),
        alternative=alternative,
        subject_ids=identifiers,
        permutation_unit="participant",
        n_permutations_requested=int(n_permutations),
        n_permutations_performed=total,
        exact=bool(exact),
        seed=None if exact else int(seed),
        null_mean=float(null_mean),
        null_sd=float(np.sqrt(null_variance)),
    )


def paired_bootstrap_mean_ci(
    differences: ArrayLike,
    subject_ids: Sequence[str],
    *,
    n_resamples: int = 10_000,
    seed: int,
    confidence: float = 0.95,
) -> BootstrapCIResult:
    """Return a participant-resampled percentile interval for the paired mean."""
    values, identifiers = _paired_values(differences, subject_ids)
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, 4096):
        stop = min(start + 4096, n_resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return BootstrapCIResult(
        estimate=float(values.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        subject_ids=identifiers,
        resampling_unit="participant",
        n_resamples=int(n_resamples),
        seed=int(seed),
    )
