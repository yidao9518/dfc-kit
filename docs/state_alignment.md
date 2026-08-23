# State alignment

State labels are identifiers assigned by an optimization run. A second KMeans
or HMM fit can recover the same patterns under a different label permutation.
State-specific occupancy, dwell, posterior, or transition values must therefore
use a common numbering before they are compared across seeds or resamples.

## Mapping convention

Every `StateAlignment` stores a `candidate_to_reference` permutation. If the
array is `[2, 0, 1]`, candidate state `0` becomes reference state `2`, candidate
state `1` becomes reference state `0`, and candidate state `2` becomes reference
state `1`.

The default one-to-one assignment minimizes the total standardized Euclidean
distance between state patterns using the Hungarian algorithm. Each feature is
scaled by the reference model's training scale, so absolute differences across
the whole network are retained without allowing naturally high-variance
features to dominate. The result retains the complete reference-by-candidate
cost matrix and one matched cost per reference state. A Pearson alternative is
available when only pattern shape should be compared: it minimizes `1 - r` and
therefore deliberately ignores proportional amplitude changes.

Both metrics record feature identity, source contract, and sample interval. The
Euclidean metric always uses the reference model's feature scales, which gives
the reference and candidate patterns one common coordinate system even when
they came from separate resamples.

## KMeans and CAP models

```python
from dfckit.states import (
    align_cap_centroids,
    align_kmeans_centroids,
    apply_state_alignment,
    relabel_kmeans_model,
)

alignment = align_kmeans_centroids(reference_fit.model, candidate_fit.model)
aligned_assignments = apply_state_alignment(candidate_fit.assignments, alignment)
aligned_model = relabel_kmeans_model(candidate_fit.model, alignment)
```

KMeans matching uses centroid patterns in the original feature space.
`relabel_kmeans_model` reorders both original and standardized centers, so
future predictions from the returned model already use reference numbering.
CAP fits use the same `KMeansStateModel` contract but expose a domain-specific
alignment entry point. Use `align_cap_centroids` for CAP models. It defaults to Pearson matching because
CAP centres describe relative spatial co-activation configurations after
within-segment ROI standardization:

```python
cap_alignment = align_cap_centroids(reference_cap.model, candidate_cap.model)
```

Pass `metric="euclidean"` explicitly if CAP amplitude differences are part of
the matching question. The generic `align_kmeans_centroids` function continues
to default to Euclidean distance for window-FC and other KMeans states.

## Gaussian HMM models

```python
from dfckit.states import (
    align_gaussian_hmm_emissions,
    apply_gaussian_hmm_alignment,
    relabel_gaussian_hmm_model,
)

alignment = align_gaussian_hmm_emissions(
    reference_hmm.model,
    candidate_hmm.model,
)
aligned_result = apply_gaussian_hmm_alignment(candidate_hmm.states, alignment)
aligned_model = relabel_gaussian_hmm_model(candidate_hmm.model, alignment)
```

HMM matching uses emission means reconstructed in the original feature space.
This is important when separate fits learned different PCA bases: reduced-space
coordinates are then not a common axis for matching.

Applying the alignment to a decoded result relabels the Viterbi sequence and
reorders posterior-probability columns. Relabeling the model applies the same
permutation to start probabilities, both axes of the transition matrix,
reduced and original emission means, and state covariance arrays. Decoding with
the relabeled model is numerically equivalent to decoding first and aligning
the result afterward.

For compact FeatureStore HMMs, `emission_covariances` remains `None` after
relabeling. The operation does not allocate the omitted original-feature
covariance tensor.

## Repeated fits

Choose the reference seed before inspecting clinical outcomes. Fit every seed
on the same training participants, align candidates to that reference, and
inspect the distribution of `matched_costs` together with aligned held-out
metrics. Smaller costs indicate closer whole-network state patterns. Use
`metric="pearson"` only when proportional shape similarity is the intended
question. In cross-validation, select a reference and perform alignment
inside each training fold; a full-cohort reference would cross the held-out
boundary.

All compared models must have the same number of states, feature identity,
source contract, and sample interval. Pearson matching is
undefined for a constant state pattern. A complete assignment can still have
large matched costs, so the mapping alone is not evidence that the solutions
are stable.

## Command-line alignment

The same complete relabeling is available for saved artifacts:

```bash
dfc-kit align-states reference.model candidate.model aligned.model alignment \
  --predictions candidate.labels \
  --predictions-output aligned.labels
```

Command-line metric selection defaults to `auto`: CAP source contracts use
Pearson matching, while other KMeans and HMM models use Euclidean distance.
`--alignment-metric euclidean` or `--alignment-metric pearson` overrides this
choice.

The aligned model and optional aligned predictions are written as new
artifacts; inputs are never modified. The prediction artifact must record the
candidate model's seed and feature settings, so labels generated by another fit
are only relabeled after the model family and feature space have been checked.
The separate alignment artifact
can be loaded through `dfckit.artifacts.load_state_alignment` for auditing or reuse.
