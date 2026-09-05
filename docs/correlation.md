# Correlation and sliding-window FC

`weighted_correlation` computes a finite weighted Pearson correlation matrix
from a frames-by-ROI array. `fisher_z_edges` extracts the upper triangle in
NumPy order and applies the Fisher transform after clipping exact endpoints.

`SlidingWindowFC` combines these kernels with the `TimeSeriesRun` topology:

```python
from dfckit.connectivity import SlidingWindowFC

result = SlidingWindowFC(
    length=60,
    step=10,
    taper="hamming",
).transform(run)
```

The Hamming taper is periodic. Use `taper="uniform"` for ordinary unweighted
Pearson FC. A window is emitted only when all of its original frame indices are
consecutive, so neither a window nor its correlation can cross a censor gap.

`WindowFCResult.features` has shape `windows x edges`. The aligned `edge_i` and
`edge_j` arrays map every feature back to its ROI pair. `start_frames`,
`end_frames`, and `segment_ids` preserve original acquisition positions.

Use `edge_vector_to_symmetric_matrix` to restore an indexed edge vector to a
symmetric matrix. `adjacent_window_pattern_similarity(result)` returns the
Pearson similarity of consecutive whole-edge patterns, along with the paired
window indices. It only forms pairs whose windows have the same `segment_id`,
so an adjacency summary cannot cross a censor gap.

`all_pair_window_pattern_similarity(result)` averages every unordered window
pair separately within each retained segment, then weights segment means by
`m - 1`. It therefore measures order-insensitive within-acquisition pattern
homogeneity on the same segment-weighting scale as the adjacent-pair mean.
`window_pattern_adjacency_excess(result)` subtracts that all-pair expectation
from the observed adjacent-pair mean. Reporting both separates a generally
more homogeneous configuration repertoire from similarity that is specific
to the observed temporal order.

`summarize_window_pattern_dataset(dataset, estimator)` writes those quantities
as three named acquisition-level endpoints. The same path is available from
XCP-D derivatives through the `window-pattern-endpoints` command.

The estimator returns statistical association, not directionality, effective
connectivity, or information flow. Window length and step determine temporal
resolution and dependence between adjacent observations and should be reported.
