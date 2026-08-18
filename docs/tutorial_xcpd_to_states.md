# Tutorial: XCP-D to dynamic states

This tutorial follows the supported production path:

```text
BIDS -> fMRIPrep -> XCP-D -> dfc-kit
```

It assumes that XCP-D has already performed nuisance regression, filtering,
censoring, and atlas extraction. `dfc-kit` does not repeat those steps.

## Load runs

Choose ROI names and their order once, then use the same selection for every
participant and session:

```python
from dfckit import TimeSeriesDataset
from dfckit.io import load_xcpd_run

xcpd_root = "/path/to/xcp_d"
subjects = ("sub-001", "sub-002", "sub-003", "sub-004")
sessions = ("off", "on")
roi_names = {
    "Glasser": ("Left_V1", "Right_V1", "Left_4", "Right_4"),
    "Tian": ("PUT-DP-lh", "PUT-DP-rh"),
}

runs = []
for subject in subjects:
    for session in sessions:
        loaded = load_xcpd_run(
            xcpd_root,
            subject=subject,
            session=session,
            atlases=("Glasser", "Tian"),
            space="MNI152NLin2009cAsym",
            roi_names=roi_names,
            minimum_coverage=0.5,
            tr=0.75,
        )
        runs.append(loaded.run)

dataset = TimeSeriesDataset(runs)
```

`TimeSeriesDataset` rejects inconsistent ROI order, mixed TR metadata, and
duplicate `(subject, session, acquisition_id)` identities.

## Calculate direct dynamic representations

```python
from dfckit.connectivity import ETS, MTD, SlidingWindowFC

mtd_results = [MTD().transform(run) for run in dataset.runs]
ets_rss_results = [ETS().rss(run) for run in dataset.runs]
window_results = [
    SlidingWindowFC(length=56, step=8, taper="hamming").transform(run)
    for run in dataset.runs
]
```

MTD differences and sliding windows never cross XCP-D censor gaps. ETS uses
separate within-segment normalization and preserves each retained original
frame index.

## Split by participant

All sessions from one participant must stay on the same side of the split:

```python
from dfckit import validate_subject_disjoint

training_subjects = {"sub-001", "sub-002", "sub-003"}
training_runs = [run for run in dataset.runs if run.subject in training_subjects]
test_runs = [run for run in dataset.runs if run.subject not in training_subjects]
validate_subject_disjoint(training_runs, test_runs)

training_windows = [
    result for result in window_results if result.subject in training_subjects
]
test_windows = [
    result for result in window_results if result.subject not in training_subjects
]
```

## Fit states and score held-out participants

Install the state-model extra before this step:

```bash
python -m pip install 'dfc-kit[states]'
```

```python
from dfckit.states import (
    fit_kmeans_states,
    predict_kmeans_states,
    summarize_state_assignments,
    window_fc_sequences,
)

training_sequences = window_fc_sequences(training_windows)
test_sequences = window_fc_sequences(test_windows)

fit = fit_kmeans_states(
    training_sequences,
    n_states=4,
    seed=20260818,
    n_init=50,
)
test_assignments = predict_kmeans_states(fit.model, test_sequences)
test_metrics = summarize_state_assignments(test_assignments)
```

The model records the fitted participant IDs. Held-out prediction rejects any
overlap with those IDs. Occupancy, dwell, switches, and transitions are then
computed separately inside every censor-delimited sequence.

If $K$ is not prespecified, split the development participants into mutually
exclusive validation folds, fit and score every candidate under every
prespecified complete-fit seed in each fold, and combine the complete
fold-by-candidate-by-seed grid of v2 score artifacts with
`compare-state-counts`. Seeds are averaged within participant; uncertainty for
the one-standard-error rule is estimated across validation folds only. In nested
cross-validation this selection runs independently inside each outer training
set; the outer test fold remains untouched. See
[Cross-fold state-count selection](state_selection.md) for the complete command
and weighting contract.

The executable path can construct those inner folds and run the complete grid
without manually invoking fit and score commands:

```bash
dfc-kit cross-validate-state-counts \
  features/window-fc.store results/k-selection \
  --method kmeans \
  --n-states 2 \
  --n-states 4 \
  --n-states 6 \
  --seed 20260818 \
  --seed 20260819 \
  --n-folds 5 \
  --split-seed 20260818
```

When this is nested inside an outer split, add one `--subject` option for each
outer-training participant. The resulting portable directory retains every
inner-fold model, held-out score, split assignment, and final comparison.

After inner selection, refit the chosen $K$ on that complete outer-training
cohort and score the untouched outer-test participants exactly once:

```bash
dfc-kit evaluate-selected-state-count \
  features/window-fc.store \
  results/k-selection \
  results/outer-evaluation \
  --selection one-standard-error \
  --test-subject sub-004
```

The command reuses every inner fit setting and model seed. It also checks that
the development feature values have not changed and that the named test
participants did not enter any inner fold. Repeat the complete inner-selection
and outer-evaluation sequence independently for each outer fold.

The complete nested loop can instead be run atomically:

```bash
dfc-kit nested-cross-validate-state-counts \
  features/window-fc.store \
  results/nested-k \
  --checkpoint results/nested-k.checkpoint \
  --method kmeans \
  --n-states 2 \
  --n-states 4 \
  --n-states 6 \
  --seed 20260818 \
  --seed 20260819 \
  --outer-n-folds 5 \
  --outer-split-seed 20260818 \
  --inner-n-folds 4 \
  --inner-split-seed 20260819
```

Every outer fold receives its own independent inner workflow. The final score
averages the one held-out score from every participant rather than averaging
fold means, so unequal outer-fold sizes do not change participant weights. If
the process is interrupted, rerun the identical command to reuse validated
inner model/score cells and complete fold artifacts; any data or option drift is
rejected. See
[Nested state-count cross-validation](nested_cross_validation.md).

Check the interrupted workflow without changing it:

```bash
dfc-kit inspect-nested-state-counts \
  features/window-fc.store \
  results/nested-k.checkpoint
```

The reported completion fraction counts validated fitted models, while separate
fields report completed inner scores; neither is an estimate of remaining
wall-clock time.
`execution_status` additionally identifies a live worker, an idle checkpoint,
or a stale owner record left by a terminated worker. Rerunning the original
command safely reclaims a stale record; a concurrent live writer is rejected.

## Align states across repeated seeds

State numbers are arbitrary: state `0` from one fit is not automatically state
`0` from another. Use one prespecified seed as the reference, align every other
fit by one-to-one centroid-pattern correlation, and only then compare state
metrics across seeds:

```python
import numpy as np

from dfckit.states import (
    align_kmeans_centroids,
    apply_state_alignment,
)

seeds = tuple(range(20260818, 20260828))
repeated_fits = [
    fit_kmeans_states(
        training_sequences,
        n_states=4,
        seed=seed,
        n_init=20,
    )
    for seed in seeds
]

reference_fit = repeated_fits[0]
aligned_test_assignments = [
    predict_kmeans_states(reference_fit.model, test_sequences)
]
matched_correlations = []

for candidate_fit in repeated_fits[1:]:
    alignment = align_kmeans_centroids(
        reference_fit.model,
        candidate_fit.model,
    )
    matched_correlations.append(alignment.matched_correlations)
    candidate_test = predict_kmeans_states(candidate_fit.model, test_sequences)
    aligned_test_assignments.append(
        apply_state_alignment(candidate_test, alignment)
    )

centroid_stability = np.stack(matched_correlations)
metrics_by_seed = [
    summarize_state_assignments(assignments)
    for assignments in aligned_test_assignments
]
occupancy_by_seed = np.stack(
    [[run.occupancy for run in metrics] for metrics in metrics_by_seed]
)
occupancy_seed_sd = occupancy_by_seed.std(axis=0, ddof=0)
```

`centroid_stability` has one row per non-reference seed and one column per
reference state. Report its distribution rather than only the best seed.
`occupancy_seed_sd` describes how much each held-out run's aligned occupancy
changes across complete refits. The first seed must be selected without looking
at clinical outcomes. In cross-validation, repeat this entire fit, alignment,
and stability procedure separately inside each training fold; a reference
centroid learned from the full cohort would leak information into held-out
folds. `n_init` controls candidate initializations within one fit, while the
outer `seeds` loop measures stability across complete fitted solutions.

For persisted FeatureStore models, the command-line path performs held-out
decoding and alignment internally and writes the complete distribution rather
than requiring manual stacking:

```bash
dfc-kit summarize-stability \
  features/window-fc.store \
  models/k4-seed-20260818.model \
  results/k4-stability.json \
  --candidate-model models/k4-seed-20260819.model \
  --candidate-model models/k4-seed-20260820.model \
  --subject sub-004
```

The reference fingerprint defines state numbering. Candidate models must share
the exact training-data fingerprint and all recorded non-seed settings. See
[Repeated-fit state stability](state_stability.md) for the JSON schema and
cross-validation boundary.

## Fit large feature stores without materializing windows

When the FC edge matrix is too large for an in-memory
`FeatureSequenceDataset`, write the same window definition to separate
participant-disjoint stores. The store retains ROI/edge identity, source
contract, sampling interval, original sample indices, and censor segments:

```python
from pathlib import Path

from dfckit.connectivity import SlidingWindowFC
from dfckit.outofcore import fit_minibatch_kmeans_store, predict_kmeans_store
from dfckit.storage import write_window_fc_store

estimator = SlidingWindowFC(length=56, step=8, taper="hamming")
training_store = write_window_fc_store(
    Path("features/training-window-fc.store"),
    training_runs,
    estimator,
    chunk_size=128,
)
test_store = write_window_fc_store(
    Path("features/test-window-fc.store"),
    test_runs,
    estimator,
    chunk_size=128,
)

fit = fit_minibatch_kmeans_store(
    training_store,
    n_states=4,
    seed=20260818,
    n_init=10,
    max_iter=10,
    batch_size=4096,
)
test_assignments = predict_kmeans_store(fit.model, test_store)
test_metrics = summarize_state_assignments(test_assignments)
```

The streaming fitter estimates the pooled scaler with a bounded-memory Welford
pass, initializes candidates from a deterministic uniform sample, and selects
the best MiniBatchKMeans initialization by a full training-store inertia pass.
It emits labels and original sample indices only; the complete edge matrix is
never reconstructed. Prediction rejects any training participant accidentally
present in the test store.

## Fit an HMM from the same stores

If a sequence model is required, use an explicit PCA dimension so the original
edge matrix remains on disk:

```python
from dfckit.outofcore_hmm import fit_gaussian_hmm_store, predict_gaussian_hmm_store

hmm_fit = fit_gaussian_hmm_store(
    training_store,
    n_states=5,
    n_pca_components=10,
    seed=20260818,
    n_init=5,
    pca_batch_size=4096,
    minimum_sequence_length=2,
)
hmm_test = predict_gaussian_hmm_store(hmm_fit.model, test_store)
hmm_metrics = summarize_state_assignments(hmm_test.assignments)
```

IncrementalPCA learns only from training participants. `hmmlearn` receives the
reduced observations together with one length per eligible censor-bounded
sequence; singleton or otherwise short sequences are omitted and recorded.
Held-out decoding uses the frozen scaler, PCA components, and HMM parameters.

## Persist fitted models

Save fitted models before a later held-out run without using pickle:

```python
from dfckit.io import load_fitted_model, save_fitted_model

model_path = save_fitted_model(
    hmm_fit.model,
    Path("models/window-hmm.model"),
)
restored_hmm = load_fitted_model(model_path)
restored_test = predict_gaussian_hmm_store(restored_hmm, test_store)
```

The artifact preserves the scaler, PCA basis, HMM parameters, feature
contract, and fitted participant IDs. Prediction after loading therefore uses
the same participant-overlap guard. The save target must not already exist;
versioned model directories avoid accidental replacement of an audited fit.

## Paired inference

After converting a run-level metric into one ON-minus-OFF difference per
participant:

```python
from dfckit.inference import (
    benjamini_hochberg,
    paired_bootstrap_mean_ci,
    paired_hc3,
    paired_sign_flip,
)

permutation = paired_sign_flip(
    on_minus_off,
    subject_ids,
    n_permutations=10_000,
    seed=20260818,
)
interval = paired_bootstrap_mean_ci(
    on_minus_off,
    subject_ids,
    n_resamples=10_000,
    seed=20260819,
)
conditional = paired_hc3(
    on_minus_off,
    covariates=qc_differences,
    subject_ids=subject_ids,
    covariate_names=("delta_mean_fd", "delta_censor"),
    difference_direction="ON minus OFF",
)
adjusted = benjamini_hochberg(
    endpoint_pvalues,
    family="prespecified dynamic-state endpoints",
)
```

The library intentionally leaves clinical-table joins, ON/OFF definitions,
endpoint selection, and comparison-family design to the study application.
