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

The estimator returns statistical association, not directionality, effective
connectivity, or information flow. Window length and step determine temporal
resolution and dependence between adjacent observations and should be reported.
