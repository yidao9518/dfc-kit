# Fixed-length MI and CMI

`dfckit.connectivity.FixedLengthInformation` estimates symmetric statistical
dependence between two predefined ROI blocks on equal-length retained-frame
windows. It implements scalar Kraskov mutual information (MI) and optional
Frenzel-Pompe conditional mutual information (CMI) with `scipy.spatial.cKDTree`.

MI and CMI are not directed. They do not identify information flow, effective
connectivity, or a causal pathway. CMI asks whether the statistical dependence
between two series remains after conditioning on another observed series; it
does not prove that the conditioning variable mediates either signal.

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
- `metric="chebyshev"` is fixed by the joint/marginal neighbourhood-counting
  form of the Kraskov and Frenzel-Pompe estimators.
- `jitter=1e-10` with a separately recorded seed deterministically breaks exact
  ties. Set `jitter=0` only when ties are known to be absent.
- Each ROI column is independently centered and divided by its population SD by
  default. Constant columns are rejected.
- When several conditioning ROI indices are supplied, their standardized time
  series are averaged into one scalar conditioning signal.
- Finite-sample kNN estimates may be slightly negative and are not clipped.

## Example

```python
from dfckit.connectivity import FixedLengthInformation

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
