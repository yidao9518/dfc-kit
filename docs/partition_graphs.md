# Fixed-partition graph metrics

`dfckit.networks` provides graph metrics for an anatomical or functional
partition supplied by the caller. No disease-specific ROI names or community
assignments are built into the package.

## Positive proportional thresholding

`positive_proportional_adjacency` ranks the upper-triangular positive weights
of a symmetric connectivity matrix. At nominal density `d`, the target number
of edges is

```text
round(d * n_nodes * (n_nodes - 1) / 2)
```

with a minimum target of one. If fewer positive edges exist, all available
positive edges are retained. `achieved_density` therefore belongs alongside
the nominal density in every result. Negative and zero edges are not converted
into graph edges.

The function does not apply Fisher transformation. Pass raw Pearson weights or
another symmetric measure deliberately; monotonic transformations preserve the
edge ranking but can change weighted modularity and node strength.

## Fixed-partition metrics

For a positive symmetric adjacency matrix `A`, node strength is the row sum
`k_i`. With `2m = A.sum()`, `fixed_partition_modularity` computes

```text
Q = (1 / 2m) * sum_ij [A_ij - (k_i * k_j / 2m)] * same_partition(i, j)
```

This evaluates one declared partition. It does not search the data for
communities.

`participation_coefficient` computes, for every non-isolated node,

```text
P_i = 1 - sum_c (k_i,c / k_i)^2
```

where `k_i,c` is the positive strength from node `i` to community `c`.
Isolated nodes receive zero.

`partition_edge_summary` works on any finite symmetric edge-weight matrix,
including signed Fisher-z FC. It counts each undirected edge once and returns
the overall within-community mean, between-community mean, their difference,
the ratio `(within - between) / within`, and community-specific block means.

## Density AUC

`FixedPartitionGraph` evaluates modularity, participation, node strength, and
achieved density at a strictly increasing density grid. Its AUC values are
trapezoidal integrals divided by the density range, so they remain on the
metric's original scale.

```python
from dfckit.networks import FixedPartitionGraph

partition = (
    "visual",
    "visual",
    "sensorimotor",
    "sensorimotor",
    "putamen",
    "putamen",
)

result = FixedPartitionGraph(
    densities=(0.10, 0.15, 0.20, 0.25, 0.30)
).transform(correlation_matrix, partition)

print(result.modularity_auc)
print(result.participation_auc)
print(result.achieved_densities)
```

The result preserves first-seen community order in `community_labels` and
returns integer `partition_codes` aligned with the matrix node axis.
