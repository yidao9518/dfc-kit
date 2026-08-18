# Chunked feature storage

`dfckit.storage.FeatureStore` persists large samples-by-features outputs as
bounded `.npy` row chunks plus one JSON manifest. NumPy memory mapping supports
out-of-core iteration without adding HDF5 or Zarr as a required dependency.

The manifest stores feature keys and order, source contract, sampling interval,
dtype, subject/session/acquisition/segment identity, chunk boundaries, and total counts.
Appending one sequence writes its chunks first and atomically replaces the
manifest only after every part succeeds. The format is append-only and permits
one writer at a time.

New stores use manifest format v2. Version 1 manifests remain readable and
represent the missing acquisition ID as `None`; appending to a v1 store upgrades
its manifest to v2 atomically.

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

## Stream complete ETS

```python
from dfckit.storage import write_ets_store

store = write_ets_store(
    "/path/to/ets.store",
    runs,
    chunk_size=128,
)
```

Segment standardization still uses the complete retained segment, but the
frame-by-edge product matrix is created and written only in bounded row chunks.
Use `ETS().rss()` instead when only co-fluctuation amplitude is required; that
algebraic path avoids edge materialization entirely.

## Stream MTD

```python
from dfckit.storage import write_mtd_store

store = write_mtd_store(
    "/path/to/mtd.store",
    runs,
    chunk_size=128,
)
```

The MTD writer uses the same run-level derivative standardization as
`MTD().transform`. It writes one FeatureStore sequence per censor-bounded
segment, with each row's original derivative start and end frame preserved.
The source contract is
`mtd:difference=within-segment;normalization=run`; it is therefore distinct
from the segment-normalized ETS contract.

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

## Stream MiniBatchKMeans

Feature stores can be fitted directly without calling `read_dataset()`:

```python
from dfckit.outofcore import fit_minibatch_kmeans_store, predict_kmeans_store

store = FeatureStore.open("/path/to/window_fc.store")
fit = fit_minibatch_kmeans_store(
    store,
    subjects=("sub-001", "sub-002"),
    n_states=4,
    seed=20260818,
    n_init=10,
    max_iter=10,
    batch_size=4096,
)
held_out_assignments = predict_kmeans_store(fit.model, held_out_store)
```

The fitter performs a bounded-memory, batch-combined Welford pass for the
feature mean and population standard deviation. A deterministic uniform sample
from the global selected-row axis supplies k-means++ initial centres. Each
initialization then makes `max_iter` complete passes with `partial_fit`; the
candidate with the smallest inertia over all selected rows is retained. The
initialization sample is representative but is not treated as a separate
dataset contract.

The returned `KMeansFitResult` is the same result type as the in-memory state
API. Its assignments contain only labels and original sample indices. The
feature store itself remains memory-mapped, and prediction rejects any subject
used for fitting by default.

## Stream PCA and Gaussian HMM

For a sequence model that needs dimensionality reduction, fit IncrementalPCA
from selected store sequences:

```python
from dfckit.outofcore import fit_incremental_pca_store, iter_pca_store_chunks

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

`TransformedFeatureChunk` is an in-memory, read-only view of one reduced chunk;
it is not appended to the original store and retains the source sequence and
sample-index metadata. Constant features are rejected when standardization is
requested.

The HMM wrapper uses the same frozen preprocessing and materializes only the
reduced observations required by `hmmlearn`:

```python
from dfckit.outofcore_hmm import fit_gaussian_hmm_store, predict_gaussian_hmm_store

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
