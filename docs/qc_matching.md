# Within-subject motion matching

`dfckit.qc` matches observations from two conditions within the same
participant on continuous QC variables. The implementation is condition-
neutral: callers provide left and right arrays rather than ON/OFF labels, and
the covariates need not be limited to framewise displacement.

Install the assignment dependency with:

```bash
python -m pip install -e '.[qc]'
```

## Window motion summaries

`summarize_window_motion` accepts a full-length framewise-displacement vector
and inclusive original-frame start/end indices. It returns mean, 95th
percentile, maximum, and finite frame count for each window. By default one
non-finite value is allowed, covering the commonly undefined FD at the first
volume of a run. Windows with additional missing values are rejected.

The input must be indexed on the original acquisition axis. Do not pass a
censored/compressed FD vector together with original-frame window indices.

## Matching objective

```python
from dfckit.qc import match_within_subject

matches = match_within_subject(
    off_motion,
    off_subjects,
    on_motion,
    on_subjects,
    calipers=(0.025, 0.075),
    covariate_names=("mean_fd", "p95_fd"),
    ceilings=(0.20, float("inf")),
    minimum_pairs=10,
)
```

An observation is eligible when every covariate is less than or equal to its
ceiling. A candidate pair is valid when every absolute left-right difference
is less than or equal to its caliper.

Within each participant, `scipy.optimize.linear_sum_assignment` solves a
lexicographic objective:

1. maximize the number of valid one-to-one pairs;
2. among those solutions, minimize the sum of
   `abs(left - right) / caliper` across covariates.

An observation can appear in at most one pair. If the final number of pairs is
below `minimum_pairs`, every pair from that participant is removed. No match
is ever formed across participants.

The result retains original left/right row indices, participant IDs, absolute
differences, scaled costs, calipers, ceilings, covariate names, input row
counts, and the SciPy implementation version.

## Endpoint differences

`matched_subject_differences` applies an existing match to one or more endpoint
arrays. It computes the pairwise right-minus-left difference and then averages
within participant, returning one row per retained participant and the number
of matched pairs.

```python
from dfckit.qc import matched_subject_differences

deltas = matched_subject_differences(
    matches,
    off_endpoints,
    on_endpoints,
    endpoint_names=("state_0_occupancy", "effective_rank"),
)
```

Matching is a sensitivity design, not an automatic rule for accepting or
rejecting a scientific endpoint. The library reports balance and matched
differences; the application defines the inference family and interpretation.
