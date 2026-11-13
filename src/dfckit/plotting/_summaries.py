"""Compact summary plots with statistics supplied explicitly by callers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._validation import validated_positive_integer, validated_seed
from ._common import _axes, _clean_axes, _colors, _density, _errors, _vector
from ._style import CONDITION_PALETTE, INK_COLOR, NEGATIVE_COLOR

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _labels(labels: Sequence[str], size: int) -> list[str]:
    output = [str(label) for label in labels]
    if len(output) != size:
        raise ValueError(f"labels must contain {size} values")
    return output


def plot_paired(
    values,
    *,
    labels,
    estimates=None,
    errors=None,
    error_label=None,
    colors=None,
    jitter=0.06,
    seed=0,
    ax: Axes | None = None,
) -> Axes:
    """Plot repeated observations with one line per subject.

    ``values`` must have shape ``(subjects, conditions)`` and contain no
    missing, masked, or non-finite entries. A subject receives one seeded
    horizontal offset shared across all conditions, so connecting lines retain
    row identity. ``estimates`` and ``errors`` are drawn only when explicitly
    supplied; ``errors`` are distances and require a descriptive
    ``error_label`` such as ``"SE"``. No summary or test is calculated.
    """
    if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
        raise ValueError("values must not contain masked observations")
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 2
        or not np.isfinite(values).all()
    ):
        raise ValueError("values must be a finite subjects-by-conditions matrix")
    n_subjects, n_conditions = values.shape
    label_values = _labels(labels, n_conditions)
    color_values = _colors(
        colors if colors is not None else [
            CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i in range(n_conditions)
        ],
        n_conditions,
    )

    jitter = float(jitter)
    if not np.isfinite(jitter) or not 0 <= jitter < 0.5:
        raise ValueError("jitter must be finite and in [0, 0.5)")
    seed = validated_seed(seed, label="seed")

    if estimates is None:
        if errors is not None:
            raise ValueError("errors require estimates")
        if error_label is not None:
            raise ValueError("error_label requires errors")
        estimate_values = None
        error_values = None
    else:
        estimate_values = _vector(estimates, "estimates", size=n_conditions)
        error_values = _errors(errors, n_conditions, error_label)

    ax = _axes(ax, figsize=(5.0, 3.5))
    positions = np.arange(n_conditions, dtype=float)
    offsets = np.random.default_rng(seed).uniform(-jitter, jitter, n_subjects)
    subject_x = positions[None, :] + offsets[:, None]
    for row_x, row_y in zip(subject_x, values, strict=True):
        ax.plot(row_x, row_y, color="#5A5F66", linewidth=0.55, alpha=0.20, zorder=1)
    for index, color in enumerate(color_values):
        ax.scatter(
            subject_x[:, index],
            values[:, index],
            s=12,
            color=color,
            alpha=0.38,
            edgecolors="none",
            zorder=2,
        )
    if estimate_values is not None:
        ax.errorbar(
            positions,
            estimate_values,
            yerr=error_values,
            fmt="D",
            markersize=5,
            color=INK_COLOR,
            markerfacecolor="white",
            markeredgewidth=1.1,
            capsize=2,
            linewidth=1.1,
            label=error_label if error_values is not None else None,
            zorder=3,
        )
    ax.set_xticks(positions, label_values)
    ax.set_xlim(-0.5, n_conditions - 0.5)
    return _clean_axes(ax)


def plot_effects(
    estimates,
    labels,
    *,
    errors=None,
    error_label=None,
    samples=None,
    density=True,
    annotations=None,
    colors=None,
    reference=0.0,
    row_bands=False,
    ax: Axes | None = None,
) -> Axes:
    """Plot caller-supplied effects in labeled horizontal rows.

    The first estimate is placed at the top. ``errors`` are symmetric ``(n,)``
    or lower/upper ``(2, n)`` distances and require ``error_label``. Optional
    ``samples`` are finite per-row vectors shown with vertical jitter;
    ``density=True`` adds a pale half-density above each supplied sample row,
    with observations below and an open diamond at the supplied estimate.
    Singleton or constant samples retain their points without a density.
    ``annotations`` are displayed verbatim at the right edge. Alternating
    ``row_bands`` are visual guides only. No averaging or inference is done.
    """
    estimate_values = _vector(estimates, "estimates")
    size = estimate_values.size
    label_values = _labels(labels, size)
    error_values = _errors(errors, size, error_label)
    color_values = _colors(colors, size)
    y_positions = np.arange(size - 1, -1, -1, dtype=float)

    if annotations is not None:
        annotation_values = _labels(annotations, size)
    else:
        annotation_values = None
    if samples is not None:
        if len(samples) != size:
            raise ValueError("samples must contain one vector per estimate")
        sample_values = [
            _vector(sample, f"samples[{index}]") for index, sample in enumerate(samples)
        ]
    else:
        sample_values = None
    if reference is not None:
        reference = float(reference)
        if not np.isfinite(reference):
            raise ValueError("reference must be finite or None")

    ax = _axes(ax, figsize=(6.0, max(2.5, 0.48 * size + 1.0)))
    if row_bands:
        for index, y_position in enumerate(y_positions):
            if index % 2 == 0:
                ax.axhspan(y_position - 0.5, y_position + 0.5, color="#000000", alpha=0.045)
    if reference is not None:
        ax.axvline(reference, color=INK_COLOR, linewidth=0.6, zorder=1)
    if sample_values is not None:
        rng = np.random.default_rng(0)
        for y_position, sample, color in zip(
            y_positions, sample_values, color_values, strict=True
        ):
            if density:
                curve = _density(sample)
                if curve is not None:
                    grid, height = curve
                    height = height / height.max() * 0.26
                    ax.fill_between(grid, y_position + 0.06, y_position + 0.06 + height,
                                    color=color, alpha=0.13, linewidth=0, zorder=1)
                    ax.plot(grid, y_position + 0.06 + height,
                            color=color, alpha=0.45, linewidth=0.6, zorder=1)
                sample_y = y_position - 0.08 - rng.uniform(0.0, 0.12, sample.size)
            else:
                sample_y = y_position + rng.uniform(-0.12, 0.12, sample.size)
            ax.scatter(sample, sample_y, s=9, color=color, alpha=0.32,
                       edgecolors="none", zorder=2)
    for index, (estimate, y_position, color) in enumerate(
        zip(estimate_values, y_positions, color_values, strict=True)
    ):
        y_error = None if error_values is None else error_values[..., index : index + 1]
        ax.errorbar(
            [estimate],
            [y_position],
            xerr=y_error,
            fmt="D",
            markersize=4.5,
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.1,
            capsize=2,
            linewidth=1.1,
            label=error_label if index == 0 and error_values is not None else None,
            zorder=3,
        )
    if annotation_values is not None:
        for y_position, annotation in zip(y_positions, annotation_values, strict=True):
            ax.text(
                1.01,
                y_position,
                annotation,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                clip_on=False,
            )
    ax.set_yticks(y_positions, label_values)
    ax.set_ylim(-0.6, size - 0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return _clean_axes(ax)


def plot_association(
    x,
    y,
    *,
    fit=None,
    color=NEGATIVE_COLOR,
    marginals=False,
    ax: Axes | None = None,
) -> Axes:
    """Plot paired finite vectors and an optional explicit linear fit.

    ``fit`` is ``(slope, intercept)`` and is never estimated here. Its line is
    drawn across the observed x range. No confidence interval, correlation, or
    p-value is calculated or annotated.

    ``marginals=True`` adds filled KDEs above/right of the axes. Leave space
    for these in the caller's layout; they are available as ``ax.child_axes``.
    Constant/singleton vectors are shown as a single marginal tick.
    """
    x_values = _vector(x, "x")
    y_values = _vector(y, "y", size=x_values.size)
    if fit is not None:
        if x_values.size < 2 or np.ptp(x_values) == 0:
            raise ValueError("a fit requires at least two distinct x values")
        fit_values = _vector(fit, "fit", size=2)
    else:
        fit_values = None

    ax = _axes(ax, figsize=(4.5, 3.5))
    ax.scatter(x_values, y_values, s=18, color=color, alpha=0.65,
               edgecolors="white", linewidths=0.4, zorder=2)
    if fit_values is not None:
        slope, intercept = fit_values
        line_x = np.array([np.min(x_values), np.max(x_values)])
        ax.plot(line_x, slope * line_x + intercept, color=INK_COLOR, linewidth=1.2)
    _clean_axes(ax)
    if marginals:
        top = ax.inset_axes([0.0, 1.04, 1.0, 0.18], sharex=ax)
        right = ax.inset_axes([1.04, 0.0, 0.18, 1.0], sharey=ax)
        for marginal, values, horizontal in ((top, x_values, True), (right, y_values, False)):
            curve = _density(values)
            if curve is None:
                if horizontal:
                    marginal.axvline(values[0], color=color, linewidth=1)
                else:
                    marginal.axhline(values[0], color=color, linewidth=1)
            else:
                grid, height = curve
                if horizontal:
                    marginal.fill_between(grid, height, color=color, alpha=0.25, linewidth=0)
                    marginal.plot(grid, height, color=color, linewidth=0.8)
                    marginal.set_ylim(bottom=0)
                else:
                    marginal.fill_betweenx(grid, height, color=color, alpha=0.25, linewidth=0)
                    marginal.plot(height, grid, color=color, linewidth=0.8)
                    marginal.set_xlim(left=0)
            marginal.set_axis_off()
    return ax


def plot_curve(
    x,
    y,
    *,
    errors=None,
    error_label=None,
    color=NEGATIVE_COLOR,
    label=None,
    ax: Axes | None = None,
) -> Axes:
    """Plot points in the caller's x/y order with optional uncertainty.

    ``errors`` are symmetric or lower/upper distances and require a descriptive
    ``error_label``. This is suitable for validation-error or explained-
    variance curves, but it neither sorts x nor selects an optimum.
    """
    x_values = _vector(x, "x")
    y_values = _vector(y, "y", size=x_values.size)
    error_values = _errors(errors, x_values.size, error_label)
    ax = _axes(ax, figsize=(4.5, 3.5))
    ax.errorbar(
        x_values,
        y_values,
        yerr=error_values,
        fmt="D-",
        color=color,
        capsize=3 if error_values is not None else 0,
        linewidth=1.3,
        markersize=4,
        markerfacecolor="white",
        label=(
            f"{label} ({error_label})"
            if label is not None and error_values is not None
            else label if label is not None else error_label
        ),
    )
    return _clean_axes(ax)


def plot_null_distribution(
    null_values,
    observed,
    *,
    tail="greater",
    bins=30,
    color=NEGATIVE_COLOR,
    ax: Axes | None = None,
) -> Axes:
    """Plot actual null draws and highlight an explicitly selected tail.

    Histograms show counts using one shared set of bin edges. ``greater`` and
    ``less`` include draws tied with ``observed``; ``two-sided`` uses the
    absolute-statistic convention ``abs(null) >= abs(observed)``. The function
    does not generate a surrogate distribution or calculate a p-value.
    """
    null_values = _vector(null_values, "null_values")
    observed = float(observed)
    if not np.isfinite(observed):
        raise ValueError("observed must be finite")
    bins = validated_positive_integer(bins, "bins")
    if tail == "greater":
        extreme = null_values >= observed
    elif tail == "less":
        extreme = null_values <= observed
    elif tail == "two-sided":
        extreme = np.abs(null_values) >= abs(observed)
    else:
        raise ValueError("tail must be 'greater', 'less', or 'two-sided'")

    ax = _axes(ax, figsize=(4.5, 3.5))
    _, bin_edges = np.histogram(null_values, bins=bins)
    ax.hist(null_values, bins=bin_edges, color="#B8C4CE", alpha=0.8, label="All null draws")
    if np.any(extreme):
        ax.hist(
            null_values[extreme],
            bins=bin_edges,
            color=color,
            alpha=0.85,
            label="Extreme null draws",
        )
    else:
        ax.hist([], bins=bin_edges, color=color, alpha=0.85, label="Extreme null draws")
    ax.axvline(observed, color="#111111", linewidth=1.4, label="Observed")
    ax.set_ylabel("Null draws (count)")
    ax.legend(frameon=False)
    return _clean_axes(ax)
