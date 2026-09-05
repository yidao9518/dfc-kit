# API map

`dfc-kit` exposes immutable result objects and composable numerical kernels.
Public names should be imported from the package area that owns the operation.

## Input and preprocessing

```python
from dfckit import TimeSeriesRun, TimeSeriesDataset, validate_subject_disjoint
from dfckit.io import load_xcpd_run, discover_xcpd_runs
```

`TimeSeriesRun` carries ROI time series, original frame indices, and acquisition
metadata. XCP-D loaders preserve censor gaps. ETS and CAP perform their required
within-segment standardization internally; the shared standardization helper is
an implementation detail rather than a public preprocessing API.

## Connectivity

```python
from dfckit.connectivity import (
    ETS,
    MTD,
    LEiDA,
    FixedPartitionGraph,
    LowRankCovariance,
    adjacent_window_pattern_similarity,
    all_pair_window_pattern_similarity,
    edge_vector_to_symmetric_matrix,
    partition_edge_summary,
    SlidingWindowFC,
    summarize_window_pattern_dataset,
    window_pattern_adjacency_excess,
)
```

Each estimator returns feature arrays together with feature keys, frame bounds,
and segment identifiers. `dfckit.information` contains MI/CMI rather than the
connectivity namespace:

```python
from dfckit.information import (
    InformationGroups,
    compute_fixed_information,
    load_fixed_information,
    save_fixed_information,
)
```

## FeatureStore

```python
from dfckit.storage import FeatureStore
from dfckit.storage import summarize_static_fc_dataset

store = FeatureStore.create(
    "results/window-fc.store",
    feature_keys=result.feature_keys,
    source_contract=result.source_contract,
    sample_interval_seconds=result.sample_interval_seconds,
)
store.append_dataset(dataset)
```

FeatureStore chunks are memory-mappable and retain sequence/acquisition
boundaries. They are an execution format, not a provenance database.

## State models

```python
from dfckit.states.streaming import fit_minibatch_kmeans_store
from dfckit.states.streaming_hmm import fit_gaussian_hmm_store
from dfckit.artifacts import load_fitted_model, save_fitted_model

fit = fit_minibatch_kmeans_store(store, n_states=4, seed=17)
save_fitted_model(fit.model, "models/k4.model")
```

`KMeansStateModel` uses K as the number of cluster centers. In
`GaussianHMMStateModel`, K is the number of hidden states. Model artifacts are
JSON plus NumPy arrays and validate shapes, finite values, feature keys, and
model parameters on load.

## State summaries and alignment

```python
from dfckit.states import summarize_state_assignments
from dfckit.states import align_kmeans_centroids
from dfckit.artifacts import save_state_alignment
```

Occupancy, dwell, switch, and transition metrics are computed separately for
each acquisition and censor-bounded sequence. Hungarian alignment is required
before combining repeated fits; choose Euclidean or Pearson cost explicitly.

## State-count selection

```python
from dfckit.artifacts import load_state_model_scores
from dfckit.states import compare_state_model_scores

reports = [load_state_model_scores(path) for path in score_paths]
comparison = compare_state_model_scores(reports)
print(comparison.selection.best_n_states)
print(comparison.selection.one_standard_error_n_states)
```

The function compares held-out scores across subject-disjoint folds. It keeps
the statistical essentials—held-out data, candidate K, model family, fold
membership, and subject-balanced aggregation—without requiring workflow
directories or nested artifact loaders.

## Inference

```python
from dfckit.inference import (
    hc3_confidence_interval,
    ols_hc3,
    paired_hc3,
    paired_nbs,
    paired_sign_flip,
)
```

Inference modules provide paired sign flips, bootstrap intervals, HC3 models,
declared-family FDR, independent named-endpoint group models, and paired NBS.
These are independent of state-count selection and model persistence.

## Advanced interfaces

Large-store fitting and artifact-format adapters are available for applications
that need them, but they are intentionally kept outside the compact package
surface. Use the documented command-line workflows or the advanced guides for
`dfckit.states.streaming`, `dfckit.states.streaming_hmm`, and direct artifact
serialization. Internal payload builders, file adapters, NBS component kernels,
and censor-segment preprocessing helpers are not stable public interfaces.
