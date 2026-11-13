"""Shared publication colours and an opt-in, temporary Matplotlib style."""

from __future__ import annotations

from math import isfinite

POSITIVE_COLOR = "#B2182B"
NEGATIVE_COLOR = "#2B7BBA"
INK_COLOR = "#1A1A1A"
SYSTEM_PALETTE = ("#4B58A6", "#C9592B", "#1E9E86", "#A03C74", "#D6A626")
CONDITION_PALETTE = ("#D39A3A", "#00798C", "#5A5F66")


def publication_style(*, font_size=7.0, font_family=None):
    """Return a temporary, white-background style for ``with`` statements.

    Keeps SVG labels as editable text and embeds TrueType fonts in PDF.
    Importing this module does not import Matplotlib or change global settings;
    leaving the context restores the caller's style. No backend is selected.
    """
    try:
        from matplotlib import rc_context
    except ImportError as error:
        raise ImportError(
            "Plotting requires matplotlib; install dfc-kit[plotting]"
        ) from error
    font_size = float(font_size)
    if not isfinite(font_size) or font_size <= 0:
        raise ValueError("font_size must be finite and positive")
    if font_family is None:
        fonts = ["Arial", "Liberation Sans", "DejaVu Sans"]
    else:
        fonts = [font_family] if isinstance(font_family, str) else list(font_family)
        if not fonts or any(not isinstance(name, str) or not name.strip() for name in fonts):
            raise ValueError("font_family must contain non-empty font names")
    return rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": fonts,
        "font.size": font_size,
        "axes.titlesize": font_size + 0.5,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 0.5 if font_size > 1 else font_size,
        "ytick.labelsize": font_size - 0.5 if font_size > 1 else font_size,
        "legend.fontsize": font_size - 0.5 if font_size > 1 else font_size,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.edgecolor": INK_COLOR,
        "axes.labelcolor": INK_COLOR,
        "xtick.color": INK_COLOR,
        "ytick.color": INK_COLOR,
        "text.color": INK_COLOR,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
