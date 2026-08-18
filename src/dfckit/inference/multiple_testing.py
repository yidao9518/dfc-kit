"""Explicit multiple-comparison families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class FDRResult:
    """Benjamini-Hochberg adjusted values and their declared family."""

    pvalues: NDArray[np.float64]
    adjusted_pvalues: NDArray[np.float64]
    family: str
    n_tests: int
    method: str = "benjamini-hochberg"


def benjamini_hochberg(pvalues: ArrayLike, *, family: str) -> FDRResult:
    """Adjust one explicitly named family while preserving missing entries."""
    if not str(family).strip():
        raise ValueError("family must be a non-empty description of the correction scope")
    probabilities = np.asarray(pvalues, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    invalid = np.isfinite(probabilities) & (
        (probabilities < 0.0) | (probabilities > 1.0)
    )
    if np.any(invalid):
        raise ValueError("finite pvalues must lie within [0, 1]")

    adjusted = np.full(len(probabilities), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(probabilities))
    if len(finite):
        order = finite[np.argsort(probabilities[finite], kind="stable")]
        ranked = probabilities[order]
        correction = ranked * len(order) / np.arange(1, len(order) + 1)
        correction = np.minimum.accumulate(correction[::-1])[::-1]
        adjusted[order] = np.minimum(correction, 1.0)

    probabilities = probabilities.copy()
    adjusted.setflags(write=False)
    probabilities.setflags(write=False)
    return FDRResult(
        pvalues=probabilities,
        adjusted_pvalues=adjusted,
        family=str(family),
        n_tests=len(finite),
    )
