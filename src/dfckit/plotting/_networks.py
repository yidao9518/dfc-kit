"""Network and matrix drawings with explicit ordering and signed scales."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._validation import validated_positive_integer
from ._style import NEGATIVE_COLOR, POSITIVE_COLOR

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _matplotlib():
    try:
        from matplotlib import colors
        from matplotlib import pyplot as plt
        from matplotlib.patches import PathPatch, Wedge
        from matplotlib.path import Path
    except ImportError as error:  # pragma: no cover - depends on the installed extras
        raise ImportError(
            "Plotting requires matplotlib; install dfc-kit[plotting]"
        ) from error
    return colors, plt, Path, PathPatch, Wedge


def _ordered_values(
    values: Sequence[str] | Mapping[str, str] | None,
    names: tuple[str, ...],
    *,
    label: str,
    default: str,
) -> tuple[str, ...]:
    if values is None:
        return (default,) * len(names)
    if isinstance(values, Mapping):
        missing = [name for name in names if name not in values]
        extra = [name for name in values if name not in set(names)]
        if missing or extra:
            raise ValueError(f"{label} keys must match node_names exactly")
        return tuple(values[name] for name in names)
    output = tuple(values)
    if len(output) != len(names):
        raise ValueError(f"{label} must contain one value per node")
    return output


def plot_ring_network(
    node_names: Sequence[str],
    edges: Sequence[tuple[str, str]],
    *,
    edge_values: Sequence[float] | None = None,
    value_limit: float | None = None,
    node_colors: Sequence[str] | Mapping[str, str] | None = None,
    labels: Sequence[str] | Mapping[str, str] | None = None,
    ax: Axes | None = None,
    start_angle: float = 90.0,
    ring_width: float = 0.10,
    boundary_color: str = "white",
    boundary_linewidth: float = 0.6,
    positive_color: str = POSITIVE_COLOR,
    negative_color: str = NEGATIVE_COLOR,
    neutral_color: str = "#666666",
    edge_width_range: tuple[float, float] = (0.3, 2.7),
    edge_alpha: float = 0.85,
    unconnected_alpha: float = 0.28,
    label_unconnected: bool = True,
) -> Axes:
    """Plot ROI-level edges inside a segmented annular node layout.

    Nodes follow ``node_names`` counterclockwise, with the first node centred
    at ``start_angle``. ``edges`` must
    contain exact node-name pairs; no thresholding, sorting, or edge selection
    is performed. When ``edge_values`` are supplied, ``value_limit`` is
    required so that callers can use one explicit width scale across panels.
    Positive and negative values control edge colour, while absolute magnitude
    controls width.

    Parameters
    ----------
    node_names:
        Unique node names in the exact display order.
    edges:
        Undirected node-name pairs. Each supplied edge is drawn once.
    edge_values:
        Optional signed value for each edge.
    value_limit:
        Positive absolute scale limit. Required with ``edge_values``; values
        outside the limit are rejected rather than silently clipped.
    node_colors:
        One Matplotlib colour per node, either in order or keyed by node name.
    labels:
        Optional display label per node. Omit to draw no text labels.
    unconnected_alpha:
        Mix unconnected ROI colours with white, retaining this fraction of
        their original colour. The full supplied ROI frame is retained.
        Set to 1 for equal saturation of all segments.
    label_unconnected:
        Whether supplied labels also identify ROIs without fitted edges.
    ax:
        Existing axes, or ``None`` to create one.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing editable vector artists.
    """
    colors, plt, Path, PathPatch, Wedge = _matplotlib()

    names = tuple(str(name) for name in node_names)
    if not names or any(not name for name in names):
        raise ValueError("node_names must contain at least one non-empty name")
    if len(set(names)) != len(names):
        raise ValueError("node_names must be unique")
    name_to_index = {name: index for index, name in enumerate(names)}

    pairs = tuple(tuple(edge) for edge in edges)
    if any(len(edge) != 2 for edge in pairs):
        raise ValueError("each edge must contain exactly two node names")
    seen: set[frozenset[str]] = set()
    for source, target in pairs:
        if source not in name_to_index or target not in name_to_index:
            raise ValueError("edge node names must occur in node_names")
        if source == target:
            raise ValueError("self-edges are not supported")
        key = frozenset((source, target))
        if key in seen:
            raise ValueError("edges must not contain duplicate undirected pairs")
        seen.add(key)

    if edge_values is None:
        if value_limit is not None:
            raise ValueError("value_limit is only valid when edge_values are supplied")
        values = None
    else:
        values = np.asarray(edge_values, dtype=float)
        if values.ndim != 1 or len(values) != len(pairs):
            raise ValueError("edge_values must contain one value per edge")
        if not np.all(np.isfinite(values)):
            raise ValueError("edge_values must be finite")
        if value_limit is None:
            raise ValueError("value_limit is required when edge_values are supplied")
        limit = float(value_limit)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("value_limit must be finite and positive")
        if np.any(np.abs(values) > limit):
            raise ValueError("absolute edge_values must not exceed value_limit")

    width_min, width_max = (float(value) for value in edge_width_range)
    if not np.isfinite([width_min, width_max]).all() or not 0 < width_min <= width_max:
        raise ValueError("edge_width_range must contain positive increasing values")
    if not np.isfinite(start_angle):
        raise ValueError("start_angle must be finite")
    if not 0.0 < ring_width < 1.0:
        raise ValueError("ring_width must be between zero and one")
    if not np.isfinite(boundary_linewidth) or boundary_linewidth < 0.0:
        raise ValueError("boundary_linewidth must be finite and non-negative")
    if not np.isfinite(edge_alpha) or not 0.0 <= edge_alpha <= 1.0:
        raise ValueError("edge_alpha must be between zero and one")
    if not np.isfinite(unconnected_alpha) or not 0.0 <= unconnected_alpha <= 1.0:
        raise ValueError("unconnected_alpha must be between zero and one")

    node_colour_values = _ordered_values(
        node_colors, names, label="node_colors", default="#808080"
    )
    label_values = (
        None if labels is None else _ordered_values(labels, names, label="labels", default="")
    )
    for colour in (
        *node_colour_values,
        boundary_color,
        positive_color,
        negative_color,
        neutral_color,
    ):
        colors.to_rgba(colour)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 4.0))

    count = len(names)
    step = 360.0 / count
    centres_deg = start_angle + np.arange(count, dtype=float) * step
    connected = {name for pair in pairs for name in pair}
    for name, centre, colour in zip(names, centres_deg, node_colour_values, strict=True):
        rgba = colors.to_rgba(colour)
        if name not in connected:
            rgba = tuple(1.0 - unconnected_alpha * (1.0 - c) for c in rgba[:3]) + rgba[3:]
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                1.0,
                centre - step / 2.0,
                centre + step / 2.0,
                width=ring_width,
                facecolor=rgba,
                edgecolor="none",
            )
        )

    inner_radius = 1.0 - ring_width
    boundary_angles = np.deg2rad(centres_deg - step / 2.0)
    for angle in boundary_angles:
        cosine, sine = np.cos(angle), np.sin(angle)
        ax.plot(
            [inner_radius * cosine, cosine],
            [inner_radius * sine, sine],
            color=boundary_color,
            linewidth=boundary_linewidth,
            solid_capstyle="butt",
            zorder=3,
        )

    centres_rad = np.deg2rad(centres_deg)
    if values is None:
        edge_colours = (neutral_color,) * len(pairs)
        edge_widths = (width_min,) * len(pairs)
    else:
        edge_colours = tuple(
            positive_color if value > 0 else negative_color if value < 0 else neutral_color
            for value in values
        )
        edge_widths = width_min + (width_max - width_min) * np.abs(values) / float(value_limit)

    for (source, target), colour, linewidth in zip(
        pairs, edge_colours, edge_widths, strict=True
    ):
        source_angle = centres_rad[name_to_index[source]]
        target_angle = centres_rad[name_to_index[target]]
        start = (inner_radius * np.cos(source_angle), inner_radius * np.sin(source_angle))
        end = (inner_radius * np.cos(target_angle), inner_radius * np.sin(target_angle))
        path = Path((start, (0.0, 0.0), end), (Path.MOVETO, Path.CURVE3, Path.CURVE3))
        ax.add_patch(
            PathPatch(
                path,
                facecolor="none",
                edgecolor=colour,
                linewidth=float(linewidth),
                alpha=edge_alpha,
                capstyle="round",
                zorder=1,
            )
        )

    if label_values is not None:
        label_radius = 1.12
        for name, angle, text in zip(names, centres_rad, label_values, strict=True):
            if not label_unconnected and name not in connected:
                continue
            x, y = label_radius * np.cos(angle), label_radius * np.sin(angle)
            align = "left" if x > 0.1 else "right" if x < -0.1 else "center"
            ax.text(x, y, text, ha=align, va="center", fontsize="small", clip_on=False)

    ax.set_xlim(-1.45 if label_values is not None else -1.04,
                1.45 if label_values is not None else 1.04)
    ax.set_ylim(-1.25 if label_values is not None else -1.04,
                1.25 if label_values is not None else 1.04)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_connectivity_matrix(
    matrix,
    *,
    value_limit: float,
    negative_limit: float | None = None,
    labels: Sequence[str] | None = None,
    node_colors: Sequence[str] | Mapping[str, str] | None = None,
    group_sizes: Sequence[int] | None = None,
    ax: Axes | None = None,
    cmap=None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
) -> Axes:
    """Plot a signed square connectivity matrix without changing its data.

    The matrix order, signs, diagonal, zeros, and existing mask are preserved.
    NaNs are displayed as unselected cells. This function does not threshold,
    symmetrize, sort, or otherwise transform the supplied values.

    ``value_limit`` defines the positive colour limit. The default negative
    limit is its opposite; supply ``negative_limit`` for an explicit asymmetric
    scale centred at zero. Optional node colour strips follow the
    matrix order. A mapping can be used for ``node_colors`` only when
    ``labels`` provide the corresponding names.
    """
    colors, plt, _, _, _ = _matplotlib()
    from matplotlib.patches import Rectangle

    values = np.ma.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not values.shape[0]:
        raise ValueError("matrix must be a non-empty square array")
    raw = np.asarray(np.ma.getdata(values), dtype=float)
    if np.isinf(raw).any():
        raise ValueError("matrix must not contain infinite values")
    values = np.ma.masked_invalid(values, copy=True)

    limit = float(value_limit)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("value_limit must be finite and positive")
    negative = limit if negative_limit is None else float(negative_limit)
    if not np.isfinite(negative) or negative <= 0.0:
        raise ValueError("negative_limit must be finite and positive")
    visible = values.compressed()
    if visible.size and (np.any(visible > limit) or np.any(visible < -negative)):
        raise ValueError("visible matrix values must remain within the supplied colour limits")

    count = values.shape[0]
    if labels is None:
        names = None
    else:
        names = tuple(str(label) for label in labels)
        if len(names) != count:
            raise ValueError("labels must contain one value per matrix row")
        if any(not name for name in names):
            raise ValueError("labels must not contain empty values")

    if node_colors is None:
        node_colour_values = None
    elif isinstance(node_colors, Mapping):
        if names is None:
            raise ValueError("labels are required when node_colors is a mapping")
        if len(set(names)) != len(names):
            raise ValueError("labels must be unique when node_colors is a mapping")
        node_colour_values = _ordered_values(
            node_colors, names, label="node_colors", default="#808080"
        )
    else:
        node_colour_values = tuple(node_colors)
        if len(node_colour_values) != count:
            raise ValueError("node_colors must contain one value per matrix row")
    if node_colour_values is not None:
        for colour in node_colour_values:
            colors.to_rgba(colour)

    boundaries: tuple[int, ...] = ()
    if group_sizes is not None:
        sizes = tuple(
            validated_positive_integer(size, f"group_sizes[{index}]")
            for index, size in enumerate(group_sizes)
        )
        if sum(sizes) != count:
            raise ValueError("group_sizes must sum to the matrix size")
        boundaries = tuple(np.cumsum(sizes)[:-1])

    if colorbar_label is not None and not colorbar:
        raise ValueError("colorbar_label requires colorbar=True")
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 4.0))

    if cmap is None:
        negative_rgb = np.array(colors.to_rgb(NEGATIVE_COLOR))
        positive_rgb = np.array(colors.to_rgb(POSITIVE_COLOR))
        cmap = colors.LinearSegmentedColormap.from_list(
            "dfckit_signed",
            [negative_rgb, negative_rgb + (1 - negative_rgb) * 0.55, "#F4F4F2",
             positive_rgb + (1 - positive_rgb) * 0.55, positive_rgb],
        )
    colour_map = plt.get_cmap(cmap).with_extremes(bad="#E5E5E5")
    image = ax.imshow(
        values,
        cmap=colour_map,
        norm=colors.TwoSlopeNorm(vmin=-negative, vcenter=0.0, vmax=limit),
        origin="upper",
        interpolation="none",
    )

    if names is None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        positions = np.arange(count)
        ax.set_xticks(positions, names, rotation=90)
        ax.set_yticks(positions, names)

    for boundary in boundaries:
        position = float(boundary) - 0.5
        ax.axvline(position, color="white", linewidth=0.6)
        ax.axhline(position, color="white", linewidth=0.6)

    if node_colour_values is not None:
        strip = 0.16
        for index, colour in enumerate(node_colour_values):
            ax.add_patch(
                Rectangle(
                    (index - 0.5, -0.5 - strip),
                    1.0,
                    strip,
                    facecolor=colour,
                    edgecolor="none",
                    clip_on=False,
                )
            )
            ax.add_patch(
                Rectangle(
                    (-0.5 - strip, index - 0.5),
                    strip,
                    1.0,
                    facecolor=colour,
                    edgecolor="none",
                    clip_on=False,
                )
            )

    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    if colorbar:
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.outline.set_visible(False)
        if colorbar_label is not None:
            bar.set_label(colorbar_label)
    return ax


__all__ = ["plot_connectivity_matrix", "plot_ring_network"]
