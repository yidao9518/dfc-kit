"""Dependency-light ordinary least squares with HC3 standard errors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import validated_subject_ids


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 300
    epsilon = 3e-14
    minimum = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        doubled = 2 * iteration
        coefficient = iteration * (b - iteration) * x / (
            (qam + doubled) * (a + doubled)
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c

        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + doubled) * (qap + doubled)
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        change = d * c
        result *= change
        if abs(change - 1.0) < epsilon:
            return result
    raise ArithmeticError("regularized incomplete beta did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not a > 0.0 or not b > 0.0 or not 0.0 <= x <= 1.0:
        raise ValueError("invalid regularized incomplete beta arguments")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_two_sided_p(statistic: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    if math.isnan(statistic):
        raise ValueError("statistic must not be NaN")
    if math.isinf(statistic):
        return 0.0
    x = degrees_of_freedom / (degrees_of_freedom + statistic * statistic)
    return _regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)


def _readonly(values: NDArray) -> NDArray:
    output = np.asarray(values).copy()
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class HC3Result:
    coefficients: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    statistics: NDArray[np.float64]
    pvalues: NDArray[np.float64]
    covariance: NDArray[np.float64]
    column_names: tuple[str, ...]
    degrees_of_freedom: int
    rank: int
    n_observations: int
    r_squared: float | None
    standard_error_type: str = "HC3"


@dataclass(frozen=True)
class PairedHC3Result:
    model: HC3Result
    subject_ids: tuple[str, ...]
    difference_direction: str
    estimand: str


def ols_hc3(
    outcome: ArrayLike,
    design: ArrayLike,
    *,
    column_names: Sequence[str],
) -> HC3Result:
    """Fit OLS and calculate finite-sample t tests from an HC3 covariance."""
    y = np.asarray(outcome, dtype=float)
    x = np.asarray(design, dtype=float)
    names = tuple(str(name) for name in column_names)
    if y.ndim != 1:
        raise ValueError("outcome must be one-dimensional")
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("design must be two-dimensional and align with outcome")
    if x.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("column_names must uniquely identify every design column")
    if any(not name.strip() for name in names):
        raise ValueError("column_names cannot contain empty names")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("HC3 inputs must be finite")
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise ValueError("HC3 design matrix is rank deficient")
    degrees_of_freedom = len(y) - rank
    if degrees_of_freedom <= 0:
        raise ValueError("HC3 requires positive residual degrees of freedom")

    inverse = np.linalg.inv(x.T @ x)
    coefficients = inverse @ x.T @ y
    fitted = x @ coefficients
    residual = y - fitted
    leverage = np.sum((x @ inverse) * x, axis=1)
    one_minus_leverage = 1.0 - leverage
    if not np.isfinite(leverage).all() or np.any(one_minus_leverage <= 1e-12):
        raise ValueError("HC3 is undefined because at least one observation has leverage h=1")
    scaled = residual / one_minus_leverage
    covariance = inverse @ (x.T @ (x * scaled[:, None] ** 2)) @ inverse
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if not np.isfinite(standard_errors).all() or np.any(standard_errors <= 0.0):
        raise ValueError("HC3 standard errors are zero or non-finite")
    statistics = coefficients / standard_errors
    pvalues = np.asarray(
        [_student_t_two_sided_p(float(value), degrees_of_freedom) for value in statistics]
    )
    total_sum_squares = float(np.square(y - y.mean()).sum())
    r_squared = (
        None
        if total_sum_squares <= 0.0
        else float(1.0 - np.square(residual).sum() / total_sum_squares)
    )
    return HC3Result(
        coefficients=_readonly(coefficients),
        standard_errors=_readonly(standard_errors),
        statistics=_readonly(statistics),
        pvalues=_readonly(pvalues),
        covariance=_readonly(covariance),
        column_names=names,
        degrees_of_freedom=degrees_of_freedom,
        rank=rank,
        n_observations=len(y),
        r_squared=r_squared,
    )


def paired_hc3(
    differences: ArrayLike,
    covariates: ArrayLike,
    subject_ids: Sequence[str],
    *,
    covariate_names: Sequence[str],
    difference_direction: str,
) -> PairedHC3Result:
    """Model one paired difference as an intercept plus paired covariates."""
    values = np.asarray(differences, dtype=float)
    covariate_values = np.asarray(covariates, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("paired differences must be one-dimensional")
    if covariate_values.ndim == 1:
        covariate_values = covariate_values[:, None]
    if covariate_values.ndim != 2 or len(covariate_values) != len(values):
        raise ValueError("covariates must align with paired differences")
    names = tuple(str(name) for name in covariate_names)
    if len(names) != covariate_values.shape[1]:
        raise ValueError("covariate_names must identify every covariate column")
    if not str(difference_direction).strip():
        raise ValueError("difference_direction must explicitly describe the paired subtraction")
    identifiers = validated_subject_ids(subject_ids, len(values))
    design = np.column_stack((np.ones(len(values)), covariate_values))
    model = ols_hc3(values, design, column_names=("intercept", *names))
    return PairedHC3Result(
        model=model,
        subject_ids=identifiers,
        difference_direction=str(difference_direction),
        estimand="intercept: paired difference when every supplied covariate equals zero",
    )
