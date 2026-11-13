"""Optional, composable plots; statistical inference stays with the caller."""

from ._networks import plot_connectivity_matrix, plot_ring_network
from ._style import (
    CONDITION_PALETTE,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    SYSTEM_PALETTE,
    publication_style,
)
from ._summaries import (
    plot_association,
    plot_curve,
    plot_effects,
    plot_null_distribution,
    plot_paired,
)

__all__ = [
    "CONDITION_PALETTE",
    "NEGATIVE_COLOR",
    "POSITIVE_COLOR",
    "SYSTEM_PALETTE",
    "plot_association",
    "plot_connectivity_matrix",
    "plot_curve",
    "plot_effects",
    "plot_null_distribution",
    "plot_paired",
    "plot_ring_network",
    "publication_style",
]
