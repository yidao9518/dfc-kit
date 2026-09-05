# State modeling

State methods consume uninterrupted feature sequences. A sequence may contain
sliding-window FC samples or instantaneous CAP samples, but transitions and
dwell runs are never formed between two censor-delimited segments or between
different acquisitions.

LEiDA leading vectors can be persisted as the same feature geometry used by
state models:

```bash
dfc-kit build-store /path/to/xcp_d /path/to/features/leida.store \
  --atlas Glasser --roi-selection rois.json --method leida \
  --minimum-segment-length 20 --tr 0.75
```

The store contains ROI-wise leading phase-coherence vectors. It does not include
the separate phase-block summary measures, which should remain in their own
analysis artifact.

## Sliding-window KMeans

```python
from dfckit.states import fit_kmeans_states, window_fc_sequences

training_sequences = window_fc_sequences(training_window_results)
fit = fit_kmeans_states(
    training_sequences,
    n_states=4,
    seed=20260818,
    n_init=50,
)
```

Window FC features are pooled across the supplied training participants and
standardized feature by feature. By default, scikit-learn KMeans is fitted
directly in the original edge-feature space:

```text
window FC edges -> training-set feature scaling -> KMeans
```

For a lower-dimensional sensitivity path, set `n_pca_components`. PCA is fitted
only on the training participants, and the frozen scaler and PCA basis are
applied before KMeans:

```text
window FC edges -> training-set feature scaling -> PCA -> KMeans
```

The model stores both the clustering-space centres and their inverse-mapped
centres in the original feature space. The latter are used for state
interpretation and alignment; the former are used for prediction and scoring.
The model also stores the PCA basis and explained-variance ratios.

```python
direct = fit_kmeans_states(
    training_sequences,
    n_states=4,
    seed=20260818,
)
pca = fit_kmeans_states(
    training_sequences,
    n_states=4,
    seed=20260818,
    n_pca_components=30,
)
```

The two fits define different state geometries and should be compared as
separate analysis specifications. PCA is not fitted again on held-out
participants.

Held-out prediction rejects participant overlap by default:

```python
from dfckit.states import predict_kmeans_states

test_assignments = predict_kmeans_states(fit.model, test_sequences)
```

For large persisted feature matrices, use the out-of-core equivalent:

```python
from dfckit.states.streaming import fit_minibatch_kmeans_store, predict_kmeans_store
from dfckit.storage import FeatureStore

fit = fit_minibatch_kmeans_store(
    FeatureStore.open("/path/to/window_fc.store"),
    n_states=4,
    seed=20260818,
    subjects=("sub-001", "sub-002"),
)
test_assignments = predict_kmeans_store(fit.model, held_out_store)
```

This path uses memory-mapped chunks, streaming standardization, deterministic
uniform-row k-means++ initialization, and participant-safe prediction. It
does not materialize a `FeatureSequenceDataset` during fitting. By default,
each initialization stops after three consecutive complete-store passes with
relative center drift at or below `1e-4`, once at least two passes have run.
`max_iter` is the upper bound; use `convergence_tol=0` when a fixed number of
passes is required.

The same store boundary is available for Gaussian HMMs through
`dfckit.states.streaming_hmm`. It first learns IncrementalPCA on eligible sequences,
then passes only reduced observations and their explicit lengths to
`hmmlearn`; see `docs/hmm.md` for the full example.

The fit result already contains assignments for the training sequences, so no
overlap exception is needed to summarize the fitted cohort.

For a command-line workflow over a persisted store, run `fit-states`, then
decode a held-out cohort and persist the labels:

```bash
dfc-kit predict-states window-fc.store models/k4.model predictions/k4.labels \
  --subject sub-010 --subject sub-011
dfc-kit summarize-states predictions/k4.labels predictions/k4.metrics.json
```

The prediction artifact retains the original sequence identity and sample
indices. The summary command applies the same gap-safe
`summarize_state_assignments` implementation used by the Python API.

## CAP

```python
from dfckit.states import fit_cap_states

fit = fit_cap_states(training_dataset, n_states=5, seed=20260818)
```

CAP input consists of instantaneous ROI patterns. Each uninterrupted retained
segment is centered and scaled ROI by ROI before fitting. The CAP wrapper uses
MiniBatchKMeans and does not apply a second pooled feature standardization.
CAP centroids are co-activation patterns, not connectivity matrices.
Use `align_cap_centroids` to align repeated CAP fits. Its default is Pearson
pattern matching; generic KMeans and HMM alignment continues to default to
standardized Euclidean distance.

For a disk-backed XCP-D cohort, build the equivalent CAP FeatureStore with
`dfc-kit build-store --method cap` or `write_cap_store`. The store uses the
source contract `cap:within-segment-roi-zscore-ddof0`, one ROI feature per row,
and one sequence per uninterrupted retained segment. Its rows preserve the
original frame indices, and segments shorter than two retained frames are
omitted. To reproduce the in-memory CAP wrapper from that store, use materialized
MiniBatchKMeans with `--no-standardize-features`:

```bash
dfc-kit fit-states cap.store models/cap-k5.model \
  --method kmeans --n-states 5 --seed 20260818 \
  --n-init 20 --max-iter 300 \
  --fitting-mode materialized --algorithm minibatch \
  --no-standardize-features
```

Materialized fitting loads the selected cohort and calls one complete
`MiniBatchKMeans.fit`; it is therefore intended for moderate stores and can
reproduce historical native geometry. The default CLI path is bounded-memory
streaming `MiniBatchKMeans.partial_fit`, which is a different fitting contract.

## LEiDA states

```python
from dfckit.connectivity import LEiDA
from dfckit.states import fit_kmeans_states, leida_sequences

phase_sequences = leida_sequences(
    [LEiDA(minimum_segment_length=20).transform(run) for run in training_runs]
)
fit = fit_kmeans_states(
    phase_sequences,
    n_states=5,
    seed=20260818,
    algorithm="minibatch",
    standardize_features=False,
)
```

Each sequence contains oriented, unit-length leading phase-coherence vectors
from one uninterrupted segment. The audited definition clusters these vectors
without feature-wise rescaling. See `docs/leida.md` for phase extraction and
sign orientation.

## Gaussian HMM

```python
from dfckit.states import fit_gaussian_hmm_states, timeseries_sequences

sequences = timeseries_sequences(training_dataset, minimum_segment_length=2)
fit = fit_gaussian_hmm_states(
    sequences,
    n_states=5,
    seed=20260818,
    n_pca_components=10,
    covariance_type="diag",
)
```

The optional HMM fits pooled training scaling, PCA, and Gaussian emissions but
passes every censor-delimited sequence length separately. See `docs/hmm.md`
for initialization selection, covariance outputs, and held-out decoding.

## State metrics

```python
from dfckit.states import summarize_state_assignments

metrics = summarize_state_assignments(test_assignments)
```

For every subject/session, the result contains occupancy, mean dwell in samples
and seconds, switch rate, transition counts, and row-conditional transition
probabilities. Counts and dwell runs restart at every sequence boundary.

## State alignment

```python
from dfckit.states import (
    align_kmeans_centroids,
    apply_state_alignment,
    relabel_kmeans_model,
)

alignment = align_kmeans_centroids(reference_model, candidate_model)
aligned = apply_state_alignment(candidate_assignments, alignment)
aligned_model = relabel_kmeans_model(candidate_model, alignment)
```

States are matched one to one with Hungarian assignment. The default metric is
standardized Euclidean distance between centroid patterns in the common
original feature space; it minimizes total whole-network distance and stores
the full cost matrix plus every matched cost. Pass `metric="pearson"` to use
`1 - r` instead when proportional pattern shape, rather than amplitude, is the
intended similarity criterion. The mapping runs from candidate labels to
reference labels.

Gaussian HMM fits use `align_gaussian_hmm_emissions`,
`apply_gaussian_hmm_alignment`, and `relabel_gaussian_hmm_model`. HMM model
relabeling includes both transition-matrix axes and posterior columns, not only
the displayed emission means. See [State alignment](state_alignment.md).

For a complete repeated-seed workflow that aligns held-out predictions and
summarizes metric dispersion, see
[`Tutorial: XCP-D to dynamic states`](tutorial_xcpd_to_states.md#align-states-across-repeated-seeds).
The command-line `summarize-stability` path performs common-store decoding and
alignment internally; its exact contract is documented in
[Repeated-fit state stability](state_stability.md).

Training inertia and training HMM likelihood should not select the state count
alone. Use `score_kmeans_store`, `score_gaussian_hmm_store`, or the
`score-states` command to evaluate frozen candidate models on the same held-out
participants. KMeans uses standardized quantization error; HMM uses gap-safe
per-sequence log likelihood. See
[Held-out state-model scoring](state_scoring.md). To compare a complete
candidate set across mutually exclusive validation folds with participant- and
fold-balanced weighting and repeated complete-fit seeds, use
`compare-state-counts`; see
[Cross-fold state-count selection](state_selection.md).
