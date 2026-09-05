# Chunked feature storage

`dfckit.storage.FeatureStore` persists large samples-by-features outputs as
bounded `.npy` row chunks plus one JSON manifest. NumPy memory mapping supports
out-of-core iteration without adding HDF5 or Zarr as a required dependency.

The manifest stores feature keys and order, source contract, sampling interval,
dtype, subject/session/acquisition/segment identity, chunk boundaries, and total counts.
Appending one sequence writes its chunks first and atomically replaces the
manifest only after every part succeeds. The format is append-only and permits
one writer at a time.

FeatureStore accepts only the current manifest schema. Every sequence record
therefore has an explicit acquisition field, which may be `None` when no
acquisition label is available.

## Stream sliding-window FC

```python
from dfckit.connectivity import SlidingWindowFC
from dfckit.storage import write_window_fc_store

store = write_window_fc_store(
    "/path/to/window_fc.store",
    runs,
    SlidingWindowFC(length=60, step=10, taper="hamming"),
    chunk_size=128,
)
```

The streaming path computes at most `chunk_size` FC rows before writing them. It
is numerically identical to `SlidingWindowFC.transform` at the default
`float64` dtype and preserves every censor segment as a separate feature
sequence.

## Stream instantaneous edges

```python
from dfckit.connectivity import ETS, MTD
from dfckit.storage import write_instantaneous_edge_store

ets_store = write_instantaneous_edge_store(
    "/path/to/ets.store",
    runs,
    ETS(),
    chunk_size=128,
)

mtd_store = write_instantaneous_edge_store(
    "/path/to/mtd.store",
    runs,
    MTD(),
    chunk_size=128,
)
```

Both generators use one writer and one instantaneous edge kernel. ETS stores
frame intervals with equal endpoints; MTD stores derivative start/end frames.
Their source contracts remain distinct so one store cannot mix incompatible
sample distributions. Use `generator.rss()` when only co-fluctuation amplitude
is required; that path avoids edge materialization.

## Stream CAP

CAP stores contain instantaneous ROI patterns, not ROI-pair edges. The writer
standardizes every ROI over the complete uninterrupted retained segment with
population standard deviation (`ddof=0`) and then writes the resulting rows in
bounded chunks:

```python
from dfckit.storage import append_cap, write_cap_store

store = write_cap_store(
    "/path/to/cap.store",
    runs,
    chunk_size=256,
)

# Append another acquisition to an existing CAP store with the same contract.
append_cap(store, another_run, chunk_size=256)
```

Each retained segment becomes its own FeatureStore sequence. Segments shorter
than two retained frames are omitted, because a segment-level standardized CAP
sequence cannot provide a usable temporal sample. Every stored row retains the
original frame index in both `sample_start_indices` and `sample_end_indices`,
and censor gaps therefore cannot be joined during later state metrics. The
source contract is `cap:within-segment-roi-zscore-ddof0`, and the feature keys
are the ROI names in their input order. Standardization is performed before
chunking, so changing `chunk_size` does not change CAP values.

## Stream LEiDA

LEiDA stores contain one leading phase-coherence eigenvector per retained frame.
Hilbert phase is estimated separately within each censor-bounded segment, and
segments shorter than the configured minimum are omitted:

```python
from dfckit.connectivity import LEiDA
from dfckit.storage import append_leida, write_leida_store

store = write_leida_store(
    "/path/to/leida.store",
    runs,
    LEiDA(minimum_segment_length=20),
    chunk_size=256,
)
append_leida(store, another_run, LEiDA(minimum_segment_length=20), chunk_size=256)
```

The source contract is
`leida:hilbert=within-segment;minimum-segment-length=20;orientation=positive-vector-sum`.
The writer preserves the original frame index in both sample-index columns and
never joins two censor-delimited segments.

## Read and append

```python
from dfckit.storage import FeatureStore, append_window_fc

store = FeatureStore.open("/path/to/window_fc.store")
for chunk in store.iter_chunks(mmap=True):
    consume(chunk.values)

sequence_part = store.read_sequence(
    "sub-001", "off", 0, sample_slice=slice(100, 200)
)

append_window_fc(store, another_run, estimator, chunk_size=128)
```

`read_dataset()` reconstructs ordinary `FeatureSequenceDataset` objects for
existing in-memory state APIs. For genuinely large analyses, iterate chunks or
read selected sequences instead of reconstructing the full dataset.

`float32` stores are supported when disk footprint is more important than exact
float64 equivalence. The dtype choice is explicit at store creation and cannot
change during later appends.

## Acquisition-level feature statistics

Whole-acquisition static FC uses the same endpoint schema without an
intermediate FeatureStore:

```python
from dfckit.storage import summarize_static_fc_dataset

static_payload = summarize_static_fc_dataset(dataset)
```

Each retained XCP-D acquisition contributes one complete upper-triangular
Fisher-z FC vector; subject, session, acquisition, ROI-pair identity, and the
number of retained frames remain attached to every endpoint row.

Feature rows can be reduced to one endpoint per acquisition, feature, and
requested statistic without materializing the store:

```python
from dfckit.storage import summarize_store_statistics

payload = summarize_store_statistics(
    store,
    statistics=("mean", "variance", "standard_deviation"),
)
```

The supported statistics are `mean`, `variance`, `standard_deviation`,
`minimum`, and `maximum`. Mean and second moments are combined across chunks
with a Welford update. Variance uses all retained samples within the
acquisition (`ddof=0`). Samples from separate censor-bounded segments may
contribute to the same acquisition summary, but no temporal feature is ever
constructed across the gap.

Each statistic receives a distinct endpoint name, such as `feature_12.mean`
or `feature_12.variance`. For edge FeatureStores, one selected statistic at a
time can be reshaped into a participants-by-edges matrix for NBS. Mixing two
statistics in one NBS graph would duplicate every edge and is invalid.

## Stream MiniBatchKMeans

Feature stores can be fitted directly without calling `read_dataset()`:

```python
from dfckit.states.streaming import fit_minibatch_kmeans_store, predict_kmeans_store

store = FeatureStore.open("/path/to/window_fc.store")
fit = fit_minibatch_kmeans_store(
    store,
    subjects=("sub-001", "sub-002"),
    n_states=4,
    seed=20260818,
    n_init=10,
    max_iter=10,
    batch_size=4096,
    convergence_tol=1e-4,
    convergence_patience=3,
    minimum_passes=2,
)
held_out_assignments = predict_kmeans_store(fit.model, held_out_store)
```

The fitter performs a bounded-memory, batch-combined Welford pass for the
feature mean and population standard deviation. A deterministic uniform sample
from the global selected-row axis supplies k-means++ initial centres. Each
initialization then makes at most `max_iter` complete passes with `partial_fit`.
After each pass, the fitter measures relative center drift as
`||C_new - C_old|| / max(||C_old||, epsilon)`. It stops when drift stays below
`convergence_tol` for `convergence_patience` consecutive passes and at least
`minimum_passes` have completed. Set `convergence_tol=0` to perform exactly
`max_iter` passes. The candidate with the smallest inertia over all selected
rows is retained. The initialization sample is representative but is not
treated as a separate dataset contract.

The returned `KMeansFitResult` is the same result type as the in-memory state
API. For the streaming path, `converged`, `passes_completed`, and
`initialization_passes` report the stopping outcome. Its assignments contain
only labels and original sample indices. The feature store itself remains
memory-mapped, and prediction rejects any subject used for fitting by default.

## Materialized KMeans fitting

For moderate stores, `fit_kmeans_store_materialized` reconstructs the selected
rows as a `FeatureSequenceDataset` and delegates to the in-memory
`fit_kmeans_states` implementation. This is the appropriate path when a
historical analysis used one complete scikit-learn `.fit` call and exact native
feature geometry matters:

```python
from dfckit.states.streaming import fit_kmeans_store_materialized

fit = fit_kmeans_store_materialized(
    FeatureStore.open("/path/to/cap.store"),
    subjects=("sub-001", "sub-002"),
    n_states=5,
    seed=20260818,
    n_init=20,
    max_iter=300,
    algorithm="minibatch",
    standardize_features=False,
)
```

Materialized fitting loads the selected cohort into memory; it is not an
out-of-core operation. `algorithm="lloyd"` calls scikit-learn `KMeans.fit`,
whereas `algorithm="minibatch"` calls `MiniBatchKMeans.fit`. Both retain the
FeatureStore source contract, feature identities, and selected fit subjects in
the returned model. CAP rows have already been
standardized within segment, so CAP reproduction normally uses
`standardize_features=False`; enabling it adds a second pooled feature
standardization and changes the fitted geometry.

The two KMeans paths have distinct numerical and memory semantics:

| Mode | Algorithm | Feature rows in RAM | Estimator operation | Typical use |
| --- | --- | --- | --- | --- |
| `streaming` | `minibatch` only | bounded chunks | repeated `partial_fit` passes | large stores |
| `materialized` | `lloyd` | selected cohort | one `KMeans.fit` call | historical Lloyd fits |
| `materialized` | `minibatch` | selected cohort | one `MiniBatchKMeans.fit` call | historical CAP/MiniBatch fits |

The streaming path is not a hidden alias for `MiniBatchKMeans.fit`: its
initialization sample, chunk traversal, and repeated `partial_fit` updates are
recorded separately in the model implementation and provenance.

## Stream PCA and Gaussian HMM

For a sequence model that needs dimensionality reduction, fit IncrementalPCA
from selected store sequences:

```python
from dfckit.states.streaming import fit_incremental_pca_store, iter_pca_store_chunks

pca = fit_incremental_pca_store(
    store,
    n_components=10,
    batch_size=4096,
    subjects=("sub-001", "sub-002"),
    minimum_sequence_length=2,
)
for chunk in iter_pca_store_chunks(pca, held_out_store):
    consume_reduced(chunk.values, chunk.sample_start_indices, chunk.sequence_index)
```

Each returned chunk contains read-only reduced values and retains the source
sequence and sample-index metadata. It is not appended to the original store.
Constant features are rejected when standardization is requested.

The HMM wrapper uses the same frozen preprocessing and materializes only the
reduced observations required by `hmmlearn`:

```python
from dfckit.states.streaming_hmm import fit_gaussian_hmm_store, predict_gaussian_hmm_store

fit = fit_gaussian_hmm_store(
    store,
    n_states=5,
    n_pca_components=10,
    seed=20260818,
    pca_batch_size=4096,
)
heldout = predict_gaussian_hmm_store(fit.model, held_out_store)
```

Short sequences below `minimum_sequence_length` are omitted exactly as in the
in-memory HMM API. The complete length vector is passed to every HMM fit and
decode call, so a censor gap cannot become a transition.
