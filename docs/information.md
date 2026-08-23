# Fixed-length MI and CMI

`dfckit.information.FixedLengthInformation` estimates symmetric statistical
dependence between two predefined ROI blocks on equal-length retained-frame
windows. It implements scalar Kraskov mutual information (MI) and optional
Frenzel-Pompe conditional mutual information (CMI) with `scipy.spatial.cKDTree`.

MI and CMI are not directed. They do not identify information flow, effective
connectivity, or a causal pathway. CMI asks whether the statistical dependence
between two series remains after conditioning on another observed series; it
does not prove that the conditioning variable mediates either signal.

## Implementation boundary

There is one MI/CMI estimator implementation. The scalar Kraskov and
Frenzel-Pompe kernels, block aggregation, and single-run fixed-window estimator
live in `dfckit.information`. The same package selects equal-length windows
across acquisitions, resolves named ROI groups, and saves or loads the
resulting analysis artifact. There is no second connectivity-level information
implementation.

## Why fixed lengths are required

kNN information estimators depend on the number of temporal samples. Comparing
an estimate from 120 frames with one from 300 frames can therefore mix a sample-
length effect with the intended biological comparison. This API samples every
estimate at one exact length. Use several explicitly reported lengths as a
sensitivity analysis rather than pooling unequal-length segments.

Sampling is with replacement and uniform over all valid start positions. A long
segment therefore contributes in proportion to its number of eligible starts;
the algorithm does not first choose a segment uniformly. Every sampled window
stays inside one contiguous `TimeSeriesRun` segment and records its draw number,
segment ID, original start/end frame, length, number of eligible starts, and
sampling seed.

## Estimator contract

- `k=3` is the default number of neighbours.
- Joint and marginal neighbourhoods use the Chebyshev norm required by these
  Kraskov and Frenzel-Pompe estimators. It is fixed internally rather than
  exposed as a configurable metric.
- `jitter=1e-10` with a separately recorded seed deterministically breaks exact
  ties. Set `jitter=0` only when ties are known to be absent.
- Each ROI column is independently centered and divided by its population SD by
  default. Constant columns are rejected.
- `jobs` may parallelize independent draw-level estimates; executor order is
  preserved, so it does not change the seeded windows or artifact row order.
- When several conditioning ROI indices are supplied, their standardized time
  series are averaged into one scalar conditioning signal.
- Finite-sample kNN estimates may be slightly negative and are not clipped.

## Example

```python
from dfckit.information import FixedLengthInformation

estimator = FixedLengthInformation(
    length=180,
    draws=20,
    sample_seed=20260813,
    k=3,
)
result = estimator.transform(
    run,
    left=[0, 1, 2],
    right=[3, 4],
    conditioning=[5, 6],
)

print(result.mean_mutual_information)
print(result.mean_conditional_mutual_information)
print(result.samples.start_frames)
```

`block_information` exposes the same pairwise and block-mean calculation for a
single frames-by-ROI array. `knn_mi` and `knn_cmi` expose the scalar numerical
kernels for advanced use.

## Batch artifact workflow

The artifact layer is independent of any atlas or clinical vocabulary. Put the
ROI names into a separate JSON file with exactly these fields:

```json
{
  "left": ["ROI_01", "ROI_02"],
  "right": ["ROI_11", "ROI_12"],
  "conditioning": ["ROI_21"]
}
```

`conditioning` may be `null` for an MI-only analysis. All named groups must be
non-empty (except `conditioning`) and disjoint. Names are resolved against the
ordered ROI axis loaded from XCP-D, so no atlas-specific indices are hidden in
this file.

The command accepts one or more lengths and runs every selected acquisition,
subject, and session:

```bash
dfc-kit fixed-information /path/to/xcp_d results/fixed-information \
  --atlas Schaefer200 \
  --space MNI152NLin2009cAsym \
  --task rest \
  --roi-selection rois.json \
  --information-groups information-groups.json \
  --length 120 --length 180 \
  --draws 20 --sample-seed 20260819 --jobs 1
```

The output is a new directory containing `manifest.json` and `arrays.npz`. The manifest
records the ordered ROI groups, estimator settings, acquisition identities,
requested length grid, actual analyzable acquisition-by-length cells, and
sampling mode. A length with no eligible retained window is omitted for that
acquisition; each retained cell still contains every draw from zero through
`draws - 1`. The numeric arrays contain pairwise MI/CMI for
every draw. `load_fixed_information` verifies that the arrays and manifest
agree before returning a result; `summarize-information` produces compact
tabular summaries when needed.

For exact replay, provide a frozen schedule with this exact header:

```text
acquisition_id\tlength\tdraw\tstart_frame\tend_frame
```

`start_frame` and `end_frame` are original, pre-censor frame indices. A schedule
may omit an entire acquisition-by-length cell. Every cell that is present must
contain all requested draws numbered from zero; a single missing draw is an
error. Unknown acquisitions and unrequested lengths are also rejected. Each
interval must be contiguous and wholly retained in one XCP-D segment. The
schedule is stable across changes in how segments are numbered; the loader
derives segment identity from the current retained frame axis.
