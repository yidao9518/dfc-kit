"""Connectivity estimators and graph summaries of connectivity matrices."""

from ._edge_products import cross_block_products, edge_products, edge_rss, within_block_products
from .correlation import edge_index, fisher_z_edges, weighted_correlation
from .instantaneous import (
    ETS,
    MTD,
    InstantaneousEdgeResult,
)
from .leida import (
    LEiDA,
    LEiDAResult,
    analytic_phase,
    cross_block_phase_coherence,
    leading_phase_eigenvectors,
    within_block_phase_coherence,
)
from .lowrank import (
    LowRankCovariance,
    LowRankCovarianceResult,
    StandardizedPCA,
    bidirectional_heldout_r2,
    effective_rank,
    eigen_concentration,
    fit_standardized_pca,
    heldout_reconstruction_r2,
    mean_projector_basis,
    subspace_distance,
    subspace_similarity,
)
from .partition import (
    FixedPartitionGraph,
    FixedPartitionGraphResult,
    PartitionEdgeSummary,
    achieved_density,
    fixed_partition_modularity,
    participation_coefficient,
    partition_edge_summary,
    positive_proportional_adjacency,
)
from .windows import SlidingWindowFC, WindowFCResult

__all__ = [
    "ETS",
    "MTD",
    "FixedPartitionGraph",
    "FixedPartitionGraphResult",
    "InstantaneousEdgeResult",
    "LEiDA",
    "LEiDAResult",
    "LowRankCovariance",
    "LowRankCovarianceResult",
    "PartitionEdgeSummary",
    "SlidingWindowFC",
    "StandardizedPCA",
    "WindowFCResult",
    "achieved_density",
    "analytic_phase",
    "bidirectional_heldout_r2",
    "cross_block_phase_coherence",
    "cross_block_products",
    "edge_index",
    "edge_products",
    "edge_rss",
    "effective_rank",
    "eigen_concentration",
    "fisher_z_edges",
    "fit_standardized_pca",
    "fixed_partition_modularity",
    "heldout_reconstruction_r2",
    "leading_phase_eigenvectors",
    "mean_projector_basis",
    "participation_coefficient",
    "partition_edge_summary",
    "positive_proportional_adjacency",
    "subspace_distance",
    "subspace_similarity",
    "weighted_correlation",
    "within_block_phase_coherence",
    "within_block_products",
]
