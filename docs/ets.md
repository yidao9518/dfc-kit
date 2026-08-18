# Edge time series

ETS represents instantaneous co-fluctuation between every ROI pair. Within
each uninterrupted retained segment, ROI signals are centered and divided by
their population standard deviation:

```text
z_i(t) = (x_i(t) - segment_mean_i) / segment_sd_i
ETS_ij(t) = z_i(t) * z_j(t)
```

Single-frame censor islands cannot define a within-segment scale and are not
included. Original frame indices and segment IDs remain attached to all
returned samples.

```python
from dfckit.connectivity import ETS

edges = ETS().transform(run)
rss_only = ETS().rss(run)
```

`transform` returns the complete frame-by-edge matrix. `rss` computes the
root-sum-square co-fluctuation amplitude with the algebraic identity

```text
RSS(t)^2 = ((sum_i z_i(t)^2)^2 - sum_i z_i(t)^4) / 2
```

and therefore does not allocate the edge matrix. This path is preferable for
whole-atlas event-amplitude analyses.

An event threshold is a cohort-fitted quantity, not a property of one run:

```python
from dfckit.connectivity import fit_ets_event_threshold

threshold = fit_ets_event_threshold(training_dataset, percentile=95)
events = threshold.event_mask(rss_only)
print(threshold.fit_subjects)
```

The fitted object records the participant IDs used to estimate the percentile.
Training and test runs should first be checked with
`validate_subject_disjoint`.
