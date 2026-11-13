"""Small shared array and axes helpers; no statistical inference."""

from __future__ import annotations

import numpy as np

from ._style import INK_COLOR, NEGATIVE_COLOR, POSITIVE_COLOR, SYSTEM_PALETTE


def _axes(ax, *, figsize=(5.0, 3.5)):
    if ax is None:
        try:
            from matplotlib import pyplot as plt
        except ImportError as error:
            raise ImportError(
                "Plotting requires matplotlib; install dfc-kit[plotting]"
            ) from error
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _vector(values, name, *, size=None):
    if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
        raise ValueError(f"{name} must not contain masked observations")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    if size is not None and values.size != size:
        raise ValueError(f"{name} must contain {size} values")
    return values


def _errors(errors, size, error_label):
    if errors is None:
        if error_label is not None:
            raise ValueError("error_label requires errors")
        return None
    if np.ma.isMaskedArray(errors) and np.ma.getmaskarray(errors).any():
        raise ValueError("errors must not contain masked observations")
    errors = np.asarray(errors, dtype=float)
    if errors.shape not in ((size,), (2, size)):
        raise ValueError("errors must have shape (n,) or (2, n)")
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("errors must be finite non-negative distances from estimates")
    if not isinstance(error_label, str) or not error_label.strip():
        raise ValueError("supply error_label, for example 'SE', 'SD', or '95% CI'")
    return errors


def _colors(colors, size):
    if colors is None:
        palette = (NEGATIVE_COLOR, POSITIVE_COLOR, *SYSTEM_PALETTE)
        return [palette[i % len(palette)] for i in range(size)]
    if isinstance(colors, str):
        return [colors] * size
    if len(colors) != size:
        raise ValueError("colors must contain one color per item")
    return list(colors)


def _density(values, *, points=192):
    """Display-only Gaussian KDE; retain all observations, bound temporary memory.

    Singleton or constant samples have no estimated density. Their observations
    are still displayed by the caller. NumPy suffices; SciPy is not required.
    """
    if values.size < 2:
        return None
    bandwidth = 1.06 * values.std(ddof=1) * values.size ** (-0.2)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        return None
    grid = np.linspace(values.min() - bandwidth, values.max() + bandwidth, points)
    density = np.zeros(points)
    for start in range(0, values.size, 512):
        z = (grid[:, None] - values[None, start : start + 512]) / bandwidth
        density += np.exp(-0.5 * z**2).sum(axis=1)
    density /= values.size * bandwidth * np.sqrt(2 * np.pi)
    return grid, density


def _clean_axes(ax):
    """Apply the same unobtrusive frame to numerical summary plots."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_COLOR)
        ax.spines[side].set_linewidth(0.6)
    return ax
