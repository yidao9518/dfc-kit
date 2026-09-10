"""Censor-aware conditional Granger predictability for ROI time series."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._arrays import readonly_copy
from .._validation import validated_positive_integer, validated_roi_indices
from ..data import TimeSeriesRun


@dataclass(frozen=True)
class ResidualWhiteness:
    """Autocorrelation diagnostic for unrestricted VAR residuals."""

    max_absolute_autocorrelation: float
    min_ljung_box_p: float
    diagnostic_lags: int


@dataclass(frozen=True)
class ConditionalGrangerResult:
    """Conditional source-to-target lagged-predictability result.

    ``geweke`` is the log ratio of restricted to unrestricted target-residual
    generalized variance. It measures predictive information in the fitted
    linear model; it does not establish neural causality.
    """

    geweke: float
    lag_order: int
    n_observations: int
    n_segments: int
    unrestricted_rank: int
    unrestricted_residual_df: int
    restricted_rank: int
    restricted_residual_df: int
    unrestricted_target_covariance: NDArray[np.float64]
    restricted_target_covariance: NDArray[np.float64]
    spectral_radius: float
    stable: bool
    whiteness: ResidualWhiteness

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "unrestricted_target_covariance",
            readonly_copy(self.unrestricted_target_covariance),
        )
        object.__setattr__(
            self,
            "restricted_target_covariance",
            readonly_copy(self.restricted_target_covariance),
        )


def _standardize(values: NDArray[np.float64]) -> NDArray[np.float64]:
    scale = values.std(axis=0, ddof=0)
    if np.any(scale <= np.finfo(float).eps):
        raise ValueError("each selected ROI must vary within every retained segment")
    return (values - values.mean(axis=0)) / scale


def _lagged_rows(
    run: TimeSeriesRun,
    variables: NDArray[np.int64],
    lag_order: int,
    *,
    standardize_segments: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, tuple[int, ...]]:
    responses: list[NDArray[np.float64]] = []
    designs: list[NDArray[np.float64]] = []
    used_segments = 0
    segment_row_counts: list[int] = []
    for positions in run.segments(min_length=lag_order + 1):
        values = np.asarray(run.values[np.ix_(positions, variables)], dtype=float)
        if standardize_segments:
            values = _standardize(values)
        response = values[lag_order:]
        lagged = np.concatenate(
            [values[lag_order - lag : -lag] for lag in range(1, lag_order + 1)],
            axis=1,
        )
        responses.append(response)
        designs.append(lagged)
        used_segments += 1
        segment_row_counts.append(len(response))
    if not responses:
        raise ValueError("no retained segment is long enough for the requested lag order")
    return (
        np.concatenate(responses),
        np.concatenate(designs),
        used_segments,
        tuple(segment_row_counts),
    )


def _fit_ols(
    design: NDArray[np.float64], response: NDArray[np.float64], *, label: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int, int]:
    design_with_intercept = np.column_stack([np.ones(len(design)), design])
    rank = int(np.linalg.matrix_rank(design_with_intercept))
    if rank < design_with_intercept.shape[1]:
        raise ValueError(f"{label} design matrix is rank deficient")
    residual_df = len(design) - rank
    if residual_df <= 0:
        raise ValueError(f"{label} model has no residual degrees of freedom")
    coefficients, _, _, _ = np.linalg.lstsq(design_with_intercept, response, rcond=None)
    residuals = response - design_with_intercept @ coefficients
    covariance = np.atleast_2d(residuals.T @ residuals / len(residuals))
    return coefficients, residuals, covariance, rank, residual_df


def _spectral_radius(coefficients: NDArray[np.float64], n_variables: int, lag_order: int) -> float:
    lag_coefficients = coefficients[1:].T
    companion = np.zeros((n_variables * lag_order, n_variables * lag_order))
    companion[:n_variables] = lag_coefficients
    if lag_order > 1:
        companion[n_variables:, :-n_variables] = np.eye(n_variables * (lag_order - 1))
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _whiteness(
    residuals: NDArray[np.float64],
    diagnostic_lags: int,
    segment_row_counts: tuple[int, ...],
) -> ResidualWhiteness:
    try:
        from scipy.stats import chi2
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Conditional Granger diagnostics require the 'inference' extra: "
            "pip install 'dfc-kit[inference]'"
        ) from error
    n = len(residuals)
    maximum = min(diagnostic_lags, n - 1)
    if maximum < 1:
        return ResidualWhiteness(float("nan"), float("nan"), 0)
    correlations: list[float] = []
    p_values: list[float] = []
    for column in residuals.T:
        centered = column - column.mean()
        denominator = float(centered @ centered)
        segment_columns = []
        first = 0
        for count in segment_row_counts:
            segment_columns.append(centered[first : first + count])
            first += count
        acf = np.asarray(
            [
                sum(
                    float(segment[lag:] @ segment[:-lag])
                    for segment in segment_columns
                    if len(segment) > lag
                )
                / denominator
                for lag in range(1, maximum + 1)
            ]
        )
        correlations.extend(np.abs(acf).tolist())
        q = n * (n + 2) * float(np.sum(acf**2 / (n - np.arange(1, maximum + 1))))
        p_values.append(float(chi2.sf(q, maximum)))
    return ResidualWhiteness(float(max(correlations)), float(min(p_values)), maximum)


def conditional_granger(
    run: TimeSeriesRun,
    *,
    source: Sequence[int],
    target: Sequence[int],
    conditioning: Sequence[int] = (),
    lag_order: int,
    standardize_segments: bool = True,
    whiteness_lags: int = 10,
) -> ConditionalGrangerResult:
    """Estimate source-to-target conditional linear Granger predictability.

    Lagged rows are constructed independently inside every contiguous retained
    frame segment and then pooled for one OLS VAR fit. The restricted target
    model removes every source lag while retaining target and conditioning lags.
    """
    lag_order = validated_positive_integer(lag_order, "lag_order")
    whiteness_lags = validated_positive_integer(whiteness_lags, "whiteness_lags")
    source_indices = validated_roi_indices(source, n_rois=run.n_rois, label="source")
    target_indices = validated_roi_indices(target, n_rois=run.n_rois, label="target")
    condition_indices = validated_roi_indices(
        conditioning, n_rois=run.n_rois, label="conditioning", minimum=0
    )
    groups = [source_indices, target_indices, condition_indices]
    selected = np.concatenate(groups)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("source, target, and conditioning ROI sets must be disjoint")

    response, lagged, n_segments, segment_row_counts = _lagged_rows(
        run, selected, lag_order, standardize_segments=standardize_segments
    )
    unrestricted_coefficients, unrestricted_residuals, _, unrestricted_rank, unrestricted_df = (
        _fit_ols(lagged, response, label="unrestricted")
    )

    n_source = len(source_indices)
    n_target = len(target_indices)
    n_variables = len(selected)
    target_response = response[:, n_source : n_source + n_target]
    target_columns = np.arange(n_source, n_source + n_target)
    condition_columns = np.arange(n_source + n_target, n_variables)
    kept_per_lag = np.concatenate([target_columns, condition_columns])
    restricted_columns = np.concatenate(
        [kept_per_lag + lag * n_variables for lag in range(lag_order)]
    )
    _, _, restricted_covariance, restricted_rank, restricted_df = _fit_ols(
        lagged[:, restricted_columns], target_response, label="restricted"
    )
    target_unrestricted_residuals = unrestricted_residuals[:, n_source : n_source + n_target]
    unrestricted_covariance = np.atleast_2d(
        target_unrestricted_residuals.T
        @ target_unrestricted_residuals
        / len(target_unrestricted_residuals)
    )
    unrestricted_sign, unrestricted_logdet = np.linalg.slogdet(unrestricted_covariance)
    restricted_sign, restricted_logdet = np.linalg.slogdet(restricted_covariance)
    if unrestricted_sign <= 0 or restricted_sign <= 0:
        raise ValueError("target residual covariance must be positive definite")
    spectral_radius = _spectral_radius(unrestricted_coefficients, n_variables, lag_order)
    return ConditionalGrangerResult(
        geweke=float(restricted_logdet - unrestricted_logdet),
        lag_order=lag_order,
        n_observations=len(response),
        n_segments=n_segments,
        unrestricted_rank=unrestricted_rank,
        unrestricted_residual_df=unrestricted_df,
        restricted_rank=restricted_rank,
        restricted_residual_df=restricted_df,
        unrestricted_target_covariance=unrestricted_covariance,
        restricted_target_covariance=restricted_covariance,
        spectral_radius=spectral_radius,
        stable=spectral_radius < 1.0,
        whiteness=_whiteness(unrestricted_residuals, whiteness_lags, segment_row_counts),
    )
