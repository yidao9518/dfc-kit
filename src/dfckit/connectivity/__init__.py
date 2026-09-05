"""Connectivity estimators and graph summaries of connectivity matrices."""

from .correlation import (
    edge_index,
    edge_vector_to_symmetric_matrix,
    fisher_z_edges,
    weighted_correlation,
)
from .instantaneous import (
    ETS,
    MTD,
    InstantaneousEdgeResult,
)
from .leida import (
    LEiDA,
    LEiDAResult,
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
    summarize_lowrank_dataset,
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
from .windows import (
    AdjacentWindowSimilarityResult,
    SlidingWindowFC,
    WindowFCResult,
    adjacent_window_pattern_similarity,
    all_pair_window_pattern_similarity,
    summarize_window_pattern_dataset,
    window_pattern_adjacency_excess,
)

__all__ = [
    "ETS",
    "MTD",
    "AdjacentWindowSimilarityResult",
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
    "adjacent_window_pattern_similarity",
    "all_pair_window_pattern_similarity",
    "bidirectional_heldout_r2",
    "edge_index",
    "edge_vector_to_symmetric_matrix",
    "effective_rank",
    "eigen_concentration",
    "fisher_z_edges",
    "fit_standardized_pca",
    "fixed_partition_modularity",
    "heldout_reconstruction_r2",
    "mean_projector_basis",
    "participation_coefficient",
    "partition_edge_summary",
    "positive_proportional_adjacency",
    "subspace_distance",
    "subspace_similarity",
    "summarize_lowrank_dataset",
    "summarize_window_pattern_dataset",
    "weighted_correlation",
    "window_pattern_adjacency_excess",
]
