# API map

The package exposes immutable result objects alongside composable numerical
kernels. Import public names from the area that owns the operation.

## Core data

```python
from dfckit import TimeSeriesRun, TimeSeriesDataset, validate_subject_disjoint
```

`TimeSeriesRun` carries a frames-by-ROI array, strictly increasing original
frame indices, ROI names, and optional subject/session/acquisition/TR metadata.

## XCP-D input

```python
from dfckit.io import (
    discover_xcpd_files,
    discover_xcpd_runs,
    load_xcpd_dataset,
    load_xcpd_run,
)
```

The loader validates atlas identity, coverage, outlier masks, and whether each
time-series table represents the full or already-censored axis.

For a shell entry point that wraps batch discovery and FeatureStore writers,
see [Command-line workflows](cli.md).

## Fitted model artifacts

```python
from dfckit.io import load_fitted_model, save_fitted_model

path = save_fitted_model(fit.model, "models/state-model")
restored_model = load_fitted_model(path)
```

The pickle-free artifact API supports `KMeansStateModel`,
`StreamingPCAModel`, and `GaussianHMMStateModel`. It preserves feature and fit
participant identity, refuses overwrite, and validates the complete schema on
load. See [Fitted model artifacts](model_artifacts.md).

`fitted_model_fingerprint(model)` computes a deterministic SHA-256 identity
over the model kind, parameters, fitted-subject metadata, algorithm settings,
and feature/source/time contract. Prediction artifacts use it to prove which
candidate model generated their labels before state alignment.

Decoded labels use a separate artifact contract:

```python
from dfckit.io import (
    StatePredictions,
    load_state_predictions,
    save_state_predictions,
    write_state_metrics,
)

prediction_path = save_state_predictions(predictions, "predictions/heldout.labels")
predictions = load_state_predictions(prediction_path)
metrics_path = write_state_metrics(predictions, "predictions/heldout.metrics.json")
```

This preserves sequence/acquisition/segment boundaries and original sample
indices; Gaussian HMM predictions additionally preserve posterior probabilities
and log likelihood. The command-line equivalents are `predict-states` and
`summarize-states`.

State alignment mappings also have a pickle-free artifact:

```python
from dfckit.io import load_state_alignment, save_state_alignment

path = save_state_alignment(alignment, "models/seed-29.alignment")
restored_alignment = load_state_alignment(path)
```

## Connectivity

```python
from dfckit.connectivity import (
    SlidingWindowFC,
    MTD,
    ETS,
    LEiDA,
    LowRankCovariance,
    FixedLengthInformation,
)
```

Lower-level functions include `weighted_correlation`, `fisher_z_edges`,
`cross_block_mtd`, `edge_rss`, `leading_phase_eigenvectors`, subspace geometry,
`knn_mi`, `knn_cmi`, and fixed-window sampling.

## Networks

```python
from dfckit.networks import (
    FixedPartitionGraph,
    positive_proportional_adjacency,
    fixed_partition_modularity,
    participation_coefficient,
    partition_edge_summary,
)
```

Network partitions are supplied by the caller; no atlas or disease-specific
partition is built into the library.

## States

```python
from dfckit.states import (
    fit_kmeans_states,
    fit_cap_states,
    fit_gaussian_hmm_states,
    summarize_state_assignments,
    align_kmeans_centroids,
    align_gaussian_hmm_emissions,
    relabel_kmeans_model,
    relabel_gaussian_hmm_model,
)
```

State feature sequences preserve acquisition and segment identity. Fit-subject overlap checks
apply to learned scalers, PCA, centroids, emissions, and state labels.
KMeans and HMM model relabeling applies one candidate-to-reference permutation
to every state-indexed parameter and decoded output.

## References, QC, and inference

```python
from dfckit import fit_feature_reference, fit_subspace_reference
from dfckit.qc import summarize_window_motion, match_within_subject
from dfckit.inference import (
    paired_sign_flip,
    paired_bootstrap_mean_ci,
    paired_hc3,
    benjamini_hochberg,
    paired_nbs,
)
```

Inference results record their participant unit, direction, tail, random seed,
and correction scope. NBS remains explicitly experimental.

## Chunked storage

```python
from dfckit.storage import (
    FeatureStore,
    append_ets,
    append_window_fc,
    write_ets_store,
    write_window_fc_store,
)
```

The store keeps feature identity, source contract, sample interval, and
subject/session/acquisition/segment boundaries alongside memory-mappable row chunks.

For state fitting without reconstructing all rows in memory:

```python
from dfckit.outofcore import (
    fit_minibatch_kmeans_store,
    predict_kmeans_store,
    score_kmeans_store,
)

fit = fit_minibatch_kmeans_store(
    FeatureStore.open("/path/to/window_fc.store"),
    subjects=("sub-001", "sub-002"),
    n_states=4,
    seed=20260818,
)
held_out = predict_kmeans_store(fit.model, held_out_store)
held_out_scores = score_kmeans_store(fit.model, held_out_store)
```

The streaming fitter keeps the training scaler, centres, and participant
identities in the same `KMeansStateModel` contract as the in-memory fitter.
Prediction emits state labels and original sample indices only; participant
overlap is rejected unless explicitly allowed.

For reduced sequence models, use the streaming PCA and HMM entry points:

```python
from dfckit.outofcore import fit_incremental_pca_store, iter_pca_store_chunks
from dfckit.outofcore_hmm import (
    fit_gaussian_hmm_store,
    predict_gaussian_hmm_store,
    score_gaussian_hmm_store,
)

pca = fit_incremental_pca_store(
    FeatureStore.open("/path/to/window_fc.store"),
    n_components=10,
    subjects=("sub-001", "sub-002"),
)
reduced_chunks = iter_pca_store_chunks(pca, FeatureStore.open("/path/to/heldout.store"))
hmm_fit = fit_gaussian_hmm_store(
    FeatureStore.open("/path/to/window_fc.store"),
    n_states=5,
    n_pca_components=10,
    seed=20260818,
)
heldout_states = predict_gaussian_hmm_store(hmm_fit.model, heldout_store)
heldout_scores = score_gaussian_hmm_store(hmm_fit.model, heldout_store)
```

The HMM path materializes only the reduced observations passed to `hmmlearn`;
each stored sequence remains a separate length in the likelihood and decoding
calls.

The scoring functions return acquisition-level standardized KMeans
quantization error or Gaussian-HMM log likelihood. The HMM path scores each
censor segment separately before run aggregation. Use
`dfckit.io.write_state_model_scores` for strict JSON provenance or the
`score-states` CLI for the complete workflow.

For cross-fold state-count selection:

```python
from dfckit.io import compare_state_model_scores, load_state_model_scores

reports = [load_state_model_scores(path) for path in score_paths]
selection = compare_state_model_scores(reports, score_artifacts=score_paths)
print(selection.selection.one_standard_error_n_states)
```

This API requires v2 score artifacts, a complete and identical
candidate-by-seed grid in every fold, identical model contracts, and mutually
exclusive validation participants. Runs are sample-weighted within participant
and seed; seeds, participants, and folds are then weighted equally in that
order. Uncertainty is estimated across folds only. See
[Cross-fold state-count selection](state_selection.md).

To construct the same balanced participant folds used by the automated CLI:

```python
from dfckit.states import make_subject_validation_folds

folds = make_subject_validation_folds(subjects, n_folds=5, seed=20260818)
for fold in folds:
    fit_on(fold.fit_subjects)
    score_on(fold.evaluation_subjects)
```

Assignment is determined by a stable SHA-256 ranking of the split seed and
subject label. Every participant appears in exactly one evaluation fold, and
all of that participant's sessions and acquisitions remain together.

Automated inner workflows can be reconstructed before a custom outer
evaluation:

```python
from dfckit.io import load_state_count_cross_validation

workflow = load_state_count_cross_validation("results/k-selection")
selected_k = workflow.selected_n_states("one-standard-error")
development_subjects = workflow.development_subjects
model_seeds = workflow.model_seeds
```

The loader reopens every model and score, rebuilds the comparison, and verifies
the manifest, split, grid, paths, and fingerprints. The
`evaluate-selected-state-count` CLI is the supported complete refit-and-score
adapter. For custom workflows,
`selected_state_count_evaluation_payload` and
`write_selected_state_count_evaluation` provide the same strict hierarchical
evaluation JSON contract after models have been fitted and scored.

Completed outer evaluations and complete nested workflows can be audited after
they have been moved:

```python
from dfckit.io import (
    load_nested_state_count_cross_validation,
    load_selected_state_count_evaluation,
)

outer = load_selected_state_count_evaluation("results/outer-evaluation")
nested = load_nested_state_count_cross_validation("results/nested-k")
print(outer.cohort_score, nested.cohort_score)
```

Both loaders reconstruct summaries from the referenced models and held-out
scores. The nested loader additionally rebuilds every inner selection and
checks that each participant enters exactly one outer test fold. The
`nested-cross-validate-state-counts --checkpoint` CLI option is the supported
resumable executor; the checkpoint manifest is an internal workflow contract,
not a result artifact. See
[Nested state-count cross-validation](nested_cross_validation.md).

For a read-only progress audit tied to the current FeatureStore:

```python
from dfckit.io import (
    inspect_nested_state_count_progress,
    nested_state_count_progress_payload,
)
from dfckit.storage import FeatureStore

store = FeatureStore.open("features/window-fc.store")
progress = inspect_nested_state_count_progress(
    store,
    "results/nested-k.checkpoint",
)
print(progress.status, progress.fit_completion_fraction)
print(progress.execution_status, progress.lock_owner_pid)
print(nested_state_count_progress_payload(progress)["folds"])
```

Validated inner models count toward fit progress even when their scores are
still pending; completed score counts are reported separately. The function
never cleans or resumes a checkpoint and rejects current-data drift, child
tampering, or symlinks before returning an immutable summary.
Execution state is independent of artifact completion: it reports whether the
checkpoint is actively owned, idle after release, stale after an unclean exit,
or backed by an invalid owner record. The inspector does not reclaim stale
ownership; the next checkpoint writer does so only after acquiring the
operating-system lock.

For a store-fitted model, reconstruct an original-space covariance one state at
a time with `dfckit.states.reconstruct_emission_covariance`; the default model
does not allocate the full high-dimensional covariance tensor.

## Repeated-fit stability

```python
from dfckit.io import write_state_stability
from dfckit.states import summarize_state_stability

runs = summarize_state_stability(reference_numbered_assignments)
write_state_stability(runs, "results/stability.json", **model_identity)
```

The low-level function accepts assignments already mapped to one reference
numbering and checks exact sequence/sample boundaries. The CLI normally
performs model decoding and alignment before calling it. See
[Repeated-fit state stability](state_stability.md).
