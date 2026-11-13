"""Render the publication-style plotting helpers using synthetic values only.

Run from an installed source checkout:
    python examples/plotting_gallery.py --output /tmp/dfckit-plotting-gallery
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from dfckit.plotting import (
    CONDITION_PALETTE,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    SYSTEM_PALETTE,
    plot_association,
    plot_connectivity_matrix,
    plot_curve,
    plot_effects,
    plot_null_distribution,
    plot_paired,
    plot_ring_network,
    publication_style,
)


def make_gallery():
    """Use the shared style, with ordinary Matplotlib composition."""
    rng = np.random.default_rng(21)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), layout="constrained")
    axes = axes.ravel()
    fig.suptitle("dfc-kit publication style | SYNTHETIC EXAMPLES", fontsize=12)

    nodes = ("V1-L", "V1-R", "M1-L", "M1-R", "PM-L", "PM-R",
             "PUT-L", "PUT-R", "THA-L", "THA-R")
    colors = tuple(color for color in SYSTEM_PALETTE for _ in range(2))
    edges = ((nodes[0], nodes[1]), (nodes[0], nodes[6]), (nodes[1], nodes[3]),
             (nodes[2], nodes[6]), (nodes[6], nodes[7]))
    weights = np.array([0.7, -0.4, -0.2, 0.3, 0.5])
    plot_ring_network(nodes, edges, edge_values=weights, value_limit=0.8,
                      node_colors=colors, labels=nodes, label_unconnected=False, ax=axes[0])
    matrix = np.full((len(nodes), len(nodes)), np.nan)
    lookup = {node: i for i, node in enumerate(nodes)}
    for (a, b), value in zip(edges, weights, strict=True):
        matrix[lookup[a], lookup[b]] = matrix[lookup[b], lookup[a]] = value
    plot_connectivity_matrix(matrix, value_limit=0.8, negative_limit=0.45,
                             labels=nodes, node_colors=colors, group_sizes=[2] * 5,
                             colorbar_label="FC (Fisher-z)", ax=axes[1])

    before = rng.normal(0.3, 0.07, 40)
    after = before + rng.normal(0.08, 0.09, 40)
    paired = np.column_stack([before, after])
    plot_paired(paired, labels=["OFF", "ON"], colors=CONDITION_PALETTE[:2],
                estimates=paired.mean(axis=0),
                errors=paired.std(axis=0, ddof=1) / np.sqrt(len(paired)),
                error_label="SE", ax=axes[2])
    axes[2].set_ylabel("FC (Fisher-z)")
    axes[2].legend()

    samples = [after - before, rng.normal(-0.09, 0.10, 40), rng.normal(0.03, 0.09, 40)]
    plot_effects([sample.mean() for sample in samples], ["Effect A", "Effect B", "Effect C"],
                 samples=samples, errors=[sample.std(ddof=1) / np.sqrt(len(sample))
                                          for sample in samples], error_label="SE",
                 colors=[POSITIVE_COLOR, NEGATIVE_COLOR, SYSTEM_PALETTE[3]], ax=axes[3])
    axes[3].set_xlabel("ON minus OFF (Fisher-z)")
    axes[3].legend(loc="lower right")

    x = np.linspace(-0.5, 0.7, 30)
    y = 0.4 * x + rng.normal(0, 0.12, len(x))
    # Explicitly supplied demonstration line; the plotter does not fit it.
    plot_association(x, y, fit=(0.4, 0.0), ax=axes[4])
    axes[4].set(xlabel="Imaging measure", ylabel="Clinical measure")
    plot_curve([2, 3, 4, 5, 6], [0.85, 0.71, 0.64, 0.59, 0.56],
               errors=[0.05, 0.04, 0.04, 0.035, 0.035], error_label="SE", ax=axes[5])
    axes[5].set(xlabel="Number of states K", ylabel="Held-out error")
    axes[5].legend()
    # Synthetic null draws are generated HERE, never by the plotting helper.
    plot_null_distribution(rng.normal(size=1500), 1.8, bins=30, ax=axes[6])
    axes[6].set_xlabel("Permutation statistic")

    # Ordinary axes composition: supplied example R² values, not model fitting.
    base, increment, full = 12.0, 20.0, 32.0
    heights = [base, increment, full]
    bottom = [0.0, base, 0.0]
    axes[7].bar([0, 1, 2], heights, bottom=bottom,
                color=["#B4B8BC", NEGATIVE_COLOR, "#26303F"], width=0.62)
    axes[7].errorbar([0, 1, 2], [base, full, full], yerr=[[0, 0, 0], [4, 5, 7]],
                     fmt="none", color="#1A1A1A", capsize=3, linewidth=1.1)
    for i, (height, offset, label) in enumerate(zip(
        heights, bottom, ["12%", "+20 pp", "32%"], strict=True
    )):
        axes[7].text(i, offset + height / 2, label, ha="center", va="center",
                      color="white" if i else "#1A1A1A", fontweight="bold")
    axes[7].set(xticks=[0, 1, 2], xticklabels=["Base", "+Predictor", "Full"],
                 ylabel="Illustrative R² (%)", ylim=(0, 45))
    titles = ("a  Complete ROI ring", "b  Signed matrix", "c  Paired observations",
              "d  Raincloud effects", "e  Association", "f  Comparison curve",
              "g  Permutation reference", "h  Model-fit composition")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=12)
    return fig


def make_association_example():
    """Leave explicit room for the two filled marginal distributions."""
    rng = np.random.default_rng(43)
    x = rng.normal(size=40)
    y = -0.45 * x + rng.normal(0, 0.6, size=x.size)
    fig = plt.figure(figsize=(4.6, 4.3))
    ax = fig.add_axes([0.18, 0.16, 0.61, 0.61])
    plot_association(x, y, fit=(-0.45, 0.0), marginals=True, ax=ax)
    ax.set(xlabel="Imaging measure", ylabel="Clinical measure")
    fig.suptitle("Filled marginals | SYNTHETIC EXAMPLE", fontsize=10)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/dfckit-plotting-gallery"))
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    with publication_style(font_size=8):
        for name, fig in (("plotting_gallery", make_gallery()),
                          ("association_marginals", make_association_example())):
            for suffix in ("png", "svg", "pdf"):
                fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
            plt.close(fig)
    print(f"Synthetic gallery written to {output}")


if __name__ == "__main__":
    main()
