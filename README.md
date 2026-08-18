# dfc-kit

`dfc-kit` is an XCP-D-first Python toolkit under active development for dynamic
functional connectivity analysis.

Its central design rule is **censor-aware by construction**: sliding windows,
time derivatives, and state transitions must remain inside contiguous retained
frame segments. Methods that learn scalers, templates, or state centers will
also expose subject-grouped interfaces to prevent leakage across sessions from
the same participant.

## Current status

The repository is independent from the Parkinson medication-intervention
research repository. The research workflows remain unchanged and serve only as
reference implementations for numerical regression tests.

Implemented through `v0.25.0`:

- discovery and validation of XCP-D parcellated derivatives;
- single- or multi-atlas ROI loading with coverage alignment;
- recovery of the original retained-frame axis from XCP-D outlier masks;
- validated ROI time-series data model;
- subject-grouped datasets with train/test participant-overlap validation;
- reconstruction of contiguous retained-frame segments;
- censor-bounded fixed-length windows;
- weighted Pearson correlation;
- Fisher-z edge-vector extraction;
- sliding-window FC transformation with explicit window metadata;
- globally standardized, censor-aware MTD with edge and block summaries;
- segment-normalized ETS, memory-efficient RSS, and subject-recorded event thresholds;
- participant-level sign-flip, bootstrap CI, BH-FDR, and HC3 inference;
- subject-recorded sliding-window KMeans and CAP state models;
- gap-safe occupancy, dwell, switch, transition, and Hungarian alignment utilities.
- low-rank standardized covariance subspaces, held-out reconstruction, and
  gap-safe adjacent-window geometry.
- positive proportional thresholding, predefined-partition modularity,
  participation, block summaries, and density AUC.
- segment-wise Hilbert phase, LEiDA leading vectors and eigenvalues, phase
  block summaries, and conversion to gap-safe state sequences.
- subject-balanced feature and subspace references, leave-one-subject-out HC
  benchmarks, overlap rejection, and balanced quantiles.
- full-axis window motion summaries and within-subject one-to-one caliper
  matching with matched endpoint differences.
- optional PCA-reduced Gaussian HMM with gap-bounded likelihood/decoding,
  multiple initializations, posterior probabilities, and held-out protection.
- fixed-length, censor-safe Kraskov MI and Frenzel-Pompe CMI with uniform-start
  sampling, deterministic tie handling, and generic ROI-block summaries.
- experimental paired NBS with direction-separated components, participant-level
  sign flips, edge-extent or intensity statistics, and fixed-threshold sensitivity.
- append-only, memory-mappable chunk stores and bounded-memory streaming writers
  for sliding-window FC and complete ETS.
- out-of-core MiniBatchKMeans fitting and prediction directly from feature stores,
  with deterministic initialization, streaming standardization, and participant
  overlap protection.
- out-of-core IncrementalPCA and Gaussian HMM fitting/decoding: only reduced
  observations are materialized, while censor-bounded sequence lengths remain
  explicit.
- pickle-free fitted-model artifacts for KMeans, streaming PCA, and Gaussian
  HMM models, with strict schemas, overwrite protection, and restored
  participant-overlap guards.
- complete KMeans and Gaussian HMM state alignment across model parameters,
  decoded labels, posterior columns, and future reference-numbered predictions.
- explicit acquisition identities across runs, connectivity results, state
  sequences, out-of-core metadata, and FeatureStore v2, with v1 compatibility;
- batch XCP-D acquisition discovery and dataset loading that preserves
  task/acq/run filename entities.
- a dependency-light `dfc-kit` CLI for XCP-D inspection and window-FC/ETS/MTD
  store construction from JSON ROI selections.
- a `fit-states` CLI that fits out-of-core MiniBatchKMeans or PCA-reduced
  Gaussian HMM models from FeatureStores and writes portable model artifacts.
- `predict-states` and `summarize-states` commands that persist held-out labels,
  HMM posteriors, gap-safe sequence metadata, and run-level state metrics.
- `align-states` for complete seed-to-reference KMeans/HMM model, label, and
  posterior relabeling with a portable alignment audit artifact.
- exact fitted-model fingerprints propagated to prediction artifacts so
  same-seed but different-cohort or different-hyperparameter results cannot be
  aligned accidentally.
- `summarize-stability` for common-store repeated-fit decoding, internal
  reference alignment, and strict JSON dispersion of gap-safe state metrics.
- fitted-model artifact v2 provenance for exact training data, KMeans
  initialization sampling, and HMM IncrementalPCA batching, while v1 artifacts
  remain readable.
- `score-states` and Python scoring APIs for acquisition-level held-out KMeans
  quantization error and censor-gap-safe Gaussian-HMM log likelihood, with
  exact model/evaluation-data fingerprints.
- `compare-state-counts` and immutable selection objects for complete candidate
  comparisons across subject-disjoint validation folds, with run, repeated-fit
  seed, participant, and fold-balanced scoring plus the one-standard-error rule.
- `cross-validate-state-counts` for deterministic participant-level folds and
  atomic end-to-end fitting, held-out scoring, and selection of a complete
  KMeans or Gaussian-HMM candidate-by-seed grid.
- strict automated-workflow loading plus `evaluate-selected-state-count` for
  full-development refitting and one-time scoring of an explicit,
  participant-disjoint outer-test cohort.
- `nested-cross-validate-state-counts` for atomic outer-fold construction,
  independent inner state-count selection, full-development refitting, and
  equal-participant aggregation across the complete outer test cohort.
- optional strict checkpoint/resume for nested state-count evaluation, with
  parameter and data-drift rejection, per-inner-cell model/score reuse,
  complete-fold reuse, and atomic finalization.
- `inspect-nested-state-counts` and Python progress objects for read-only,
  current-data-bound auditing of interrupted, promotion-ready, or complete
  nested workflows with separate completed inner-model and score counts.
- single-writer process locks for resumable nested checkpoints, with visible
  owner identity, normal-release state, and crash-recoverable stale records.

The supported production boundary is XCP-D output. `dfc-kit` does not repeat
fMRIPrep-to-XCP-D denoising, filtering, censoring, or parcellation. The array
API remains available as the internal mathematical contract and for advanced
users who already have equivalently preprocessed ROI time series.

Version `0.25.0` is prepared as a public alpha. Synthetic performance
profiling, chunked feature storage, and out-of-core state fitting are included
for high-dimensional outputs.

## Example

```python
from dfckit.connectivity import SlidingWindowFC
from dfckit.io import load_xcpd_run

loaded = load_xcpd_run(
    "/path/to/xcp_d",
    subject="sub-001",
    session="off",
    atlases=("Glasser", "Tian"),
    space="MNI152NLin2009cAsym",
    roi_names={
        "Glasser": ("Left_V1", "Right_V1", "Left_4", "Right_4"),
        "Tian": ("PUT-DP-lh", "PUT-DP-rh"),
    },
    minimum_coverage=0.5,
    tr=0.75,
)

result = SlidingWindowFC(length=48, step=24, taper="hamming").transform(loaded.run)
print(result.features.shape)
print(result.start_frames)
```

## Development

```bash
python -m pip install -e '.[all,dev,docs]'
python -m unittest discover
ruff check src tests
mkdocs build --strict
```

See `docs/xcpd_input.md`, `docs/datasets.md`, `docs/correlation.md`,
`docs/mtd.md`, `docs/ets.md`,
`docs/lowrank.md`, `docs/inference.md`, `docs/states.md`,
`docs/partition_graphs.md`, `docs/leida.md`, `docs/reference.md`,
`docs/qc_matching.md`, `docs/hmm.md`, `docs/information.md`,
`docs/nbs.md`, `docs/storage.md`, `docs/state_alignment.md`,
`docs/state_stability.md`, `docs/model_artifacts.md`,
`docs/state_scoring.md`, `docs/state_selection.md`,
`docs/nested_cross_validation.md`,
`docs/architecture.md`, and `ROADMAP.md`
for the input contract, scope, and migration rules. A complete worked path is available in
`docs/tutorial_xcpd_to_states.md`.
