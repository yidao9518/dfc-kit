# State modeling

State methods consume uninterrupted feature sequences. A sequence may contain
sliding-window FC samples or instantaneous CAP samples, but transitions and
dwell runs are never formed between two censor-delimited segments or between
different acquisitions.

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

Window FC features are pooled across the supplied training participants,
standardized feature by feature, and fitted with scikit-learn Lloyd KMeans.
The model stores its seed, fitted participant IDs, scaler, centroids, feature
identity, source contract, and implementation version.

Held-out prediction rejects participant overlap by default:

```python
from dfckit.states import predict_kmeans_states

test_assignments = predict_kmeans_states(fit.model, test_sequences)
```

For large persisted feature matrices, use the out-of-core equivalent:

```python
from dfckit.outofcore import fit_minibatch_kmeans_store, predict_kmeans_store
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
does not materialize a `FeatureSequenceDataset` during fitting.

The same store boundary is available for Gaussian HMMs through
`dfckit.outofcore_hmm`. It first learns IncrementalPCA on eligible sequences,
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

States are matched one to one with Hungarian assignment that maximizes Pearson
correlation between centroid patterns in the common original feature space.
The mapping runs from candidate labels to reference labels and stores the full
correlation matrix plus every matched correlation.

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
