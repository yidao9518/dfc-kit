# Engineering plan

## Goal

Build a reusable Python library for dynamic functional connectivity estimation,
state modeling, stability analysis, quality-control sensitivity, and paired
inference. Production inputs are XCP-D parcellated derivatives. The library
must remain independent from any disease, cohort, atlas, clinical scale,
manuscript, or server layout, and it does not reimplement fMRIPrep-to-XCP-D
denoising or parcellation.

## Repository boundary

This repository owns:

- mathematical kernels;
- validated in-memory data structures;
- general file adapters;
- method-level result objects;
- unit, numerical regression, and integration tests;
- tutorials using synthetic or openly redistributable data.
- reproducible synthetic performance benchmarks.

The Parkinson research repository owns:

- cohort discovery and inclusion rules;
- fixed study ROI definitions;
- ON/OFF and clinical variable semantics;
- manuscript-specific comparison groups and figures;
- frozen historical runners and outputs.

No research workflow will be moved or modified during library development. A
canonical algorithm may be reimplemented here after its mathematical contract
is written. Numerical equivalence will then be checked against a small frozen,
de-identified fixture exported explicitly for regression testing.

## Design principles

### Censor-aware by construction

Original frame indices are part of the core data model. Algorithms that depend
on temporal adjacency receive segment/window objects rather than arbitrary
concatenated arrays. This prevents windows, derivatives, phase transitions, and
state transitions from silently crossing removed frames.

### XCP-D-first production input

The public file-based workflow starts from XCP-D atlas time-series, coverage,
outlier-mask, and optional motion files. It supports both full-length atlas
tables and `--output-type censored` tables, but always reconstructs the original
retained-frame indices before a temporal method runs. The lower-level array API
is retained for numerical kernels, synthetic tests, and already preprocessed
data; it is not a second preprocessing pipeline.

### Subject-safe learning

Any method that learns a scaler, PCA basis, template, centroid, threshold, or
hyperparameter must record the fit subjects. Cross-validation helpers must
reject overlap between training and test participant IDs, even when session IDs
differ.

### Explicit statistical contracts

Permutation unit, tail, comparison group, random seed, and correction scope are
arguments and result metadata. The library will not infer an FDR group from a
table or automatically turn every returned p-value into one comparison family.

### Small composable objects

Numerical kernels return arrays or immutable result objects. CLI tools, TSV
writers, BIDS discovery, plotting, and manuscript formatting remain adapters or
applications rather than being embedded in estimators.

### Stable terminology

- `n_states`: number of clusters or latent states;
- `n_components`: PCA or low-rank dimensionality;
- `segment`: consecutive retained original frames;
- `window`: fixed-length portion of one segment;
- `run`: one ROI time-series acquisition, identified by `(subject, session,
  acquisition_id)` when those fields are available;
- `dataset`: multiple runs with participant/session metadata.

## Planned package areas

| Area | Scope |
|---|---|
| `data`, `segments` | Runs, original indices, segments, windows, dataset grouping |
| `connectivity.correlation` | Static/weighted FC, edge indexing, Fisher transform |
| `connectivity.mtd` | Globally standardized within-segment derivatives and MTD |
| `connectivity.ets` | Edge time series, RSS, events, event summaries |
| `connectivity.leida` | Phase extraction and leading eigenvectors |
| `connectivity.lowrank` | Effective rank, held-out reconstruction, subspace geometry |
| `connectivity.information` | Fixed-length MI/CMI estimators and sampling |
| `states` | KMeans, CAP, HMM, state summaries, and complete KMeans/HMM label alignment |
| `outofcore`, `outofcore_hmm` | memory-mapped PCA/KMeans/HMM workflows with sequence-safe decoding |
| `networks` | Block means and partition metrics |
| `inference` | Paired permutation, bootstrap, multiple testing, robust regression, experimental NBS |
| `qc` | Motion summaries and within-participant window matching |
| `io.xcpd` | Primary XCP-D derivative discovery, validation, atlas loading, coverage, censor axis |
| `io.models` | pickle-free fitted KMeans/PCA/HMM artifacts with strict schema validation |

## Development phases

### Phase 1: core topology and FC

- validated `TimeSeriesRun`;
- contiguous segments and fixed windows;
- weighted correlation and Fisher-z edges;
- sliding-window FC result object;
- XCP-D single/multi-atlas input adapter;
- subject-grouped datasets and participant-overlap validation;
- globally standardized censor-aware MTD;
- segment-normalized ETS with a non-materializing RSS path;
- participant-level sign-flip, bootstrap CI, BH-FDR, and HC3 inference;
- subject-recorded KMeans/CAP models, gap-safe state metrics, and Hungarian alignment;
- synthetic gap and mathematical tests.

Status: initial implementation complete.

### Phase 2: first useful release

- MTD and ETS;
- KMeans and CAP state estimators;
- occupancy, dwell, switch rate, and transitions;
- Hungarian label alignment and cross-seed stability;
- paired sign-flip, bootstrap CI, BH-FDR, and HC3;
- golden regression tests for selected kernels.

Target: `v0.1`.

### Phase 3: continuous organization

- low-rank covariance subspaces;
- fixed-partition graph metrics;
- LEiDA;
- healthy-reference utilities;
- motion-matched window sensitivity.

Target: `v0.2`.

### Phase 4: optional and experimental methods

- Gaussian HMM as an optional dependency;
- fixed-length MI/CMI;
- NBS under an experimental namespace;
- performance profiling and chunked feature storage.

Target: `v0.3`.

Status: HMM, fixed-length MI/CMI, experimental NBS, public documentation,
performance profiling, chunked feature storage, and out-of-core MiniBatchKMeans
are complete. The v0.5 streaming PCA/HMM extension, v0.6 portable fitted
models, and v0.7 complete KMeans/HMM state alignment are also complete.
Further bounded-memory methods must retain an explicit sequence and leakage
contract.

## Definition of done for a migrated method

1. The mathematical object and assumptions are documented.
2. Synthetic unit tests cover known values and invalid inputs.
3. Gap tests demonstrate that no temporal operation crosses censor boundaries.
4. Randomized methods reproduce under a fixed seed.
5. ROI and edge ordering travel with the result.
6. A numerical regression fixture agrees with the selected reference
   implementation within a declared tolerance.
7. The public API states what the result measures and what it cannot establish.
8. Examples do not contain private or identifiable research data.

## Immediate next work

1. Add first Git commit and configure a remote only after the repository owner
   explicitly chooses the remote; no remote is currently configured.
2. Completed in `v0.5`: add high-dimensional streaming PCA/HMM benchmark cases
   and document their reduced-observation memory boundary.
3. Completed in `v0.5`: extend the tutorial with state alignment and
   repeated-seed stability examples.
4. Completed in `v0.6`: add pickle-free fitted-model persistence with strict
   restoration and participant-leakage validation.
5. Completed in `v0.7`: align repeated HMM fits in original-feature emission
   space and apply one permutation to every state-indexed model/result axis.
6. Completed in `v0.8`: propagate acquisition identity through all result and
   state objects, upgrade FeatureStore manifests to v2 with v1 compatibility,
   and add batch XCP-D discovery/loading for repeated acquisitions.
7. Completed in `v0.9`: add dependency-light CLI acquisition inspection and
   bounded-memory window-FC/ETS FeatureStore construction from XCP-D outputs.
8. Completed in `v0.10`: add CLI MiniBatchKMeans and PCA/Gaussian HMM fitting
   from FeatureStores with explicit training subjects and portable artifacts.
9. Completed in `v0.11`: add participant-safe CLI decoding, strict state
   prediction persistence, HMM posterior retention, and JSON state metrics.
10. Completed in `v0.12`: expose complete KMeans/HMM artifact alignment with
    seed-identified predictions and auditable mapping persistence.
11. Completed in `v0.13`: replace seed-only prediction provenance with exact
    fitted-model fingerprints while retaining legacy summary compatibility.
12. Completed in `v0.14`: decode repeated fits on one cohort, align every
    candidate to a fingerprint-identified reference, and emit gap-safe metric
    dispersion with exact fit and sequence provenance.
13. Completed in `v0.15`: score frozen KMeans and Gaussian HMM models on an
    explicit held-out cohort, aggregate by acquisition without crossing censor
    gaps, and persist exact model/evaluation identities for state-count
    comparison.
14. Completed in `v0.16`: compare complete candidate state counts across
    subject-disjoint validation folds, aggregate runs without participant or
    fold-size weighting, apply the one-standard-error rule, and preserve exact
    model, feature, training, evaluation, and candidate identities.
15. Completed in `v0.17`: repeat each candidate fit across a complete seed grid,
    average seeds inside each participant, retain folds as the only uncertainty
    unit, and preserve every seed-level score and fitted-model identity.
16. Completed in `v0.18`: construct deterministic participant-level validation
    folds and run the complete KMeans/HMM candidate-by-seed fitting, scoring,
    persistence, and state-count comparison workflow atomically from one
    FeatureStore, with an explicit outer-training cohort boundary.
17. Completed in `v0.19`: reconstruct and verify the complete inner-selection
    workflow, refit its selected state count on the full development cohort for
    every prespecified seed, and atomically evaluate only an explicit,
    participant-disjoint outer-test cohort with hierarchical score aggregation.
18. Completed in `v0.20`: construct deterministic outer participant folds,
    repeat independent inner state-count selection inside every outer training
    cohort, refit and score each untouched outer fold once, and strictly rebuild
    the complete nested evidence tree with equal-participant aggregation.
19. Completed in `v0.21`: preserve complete nested-workflow subartifacts in an
    explicit checkpoint, strictly reject command or data drift on resume, clean
    interrupted child temporaries, and atomically consume the verified checkpoint
    into the final portable result.
20. Completed in `v0.22`: inspect an interrupted or complete nested workflow
    without mutation, rebuild every complete child against the current
    FeatureStore, and report fold state, temporary residue, and audited fit-step
    completion through Python and JSON CLI objects.
21. Completed in `v0.23`: serialize writers to each resumable nested checkpoint
    with an OS advisory lock, retain auditable process ownership across normal
    release or hard exit, and expose live, idle, stale, or invalid execution
    state through the read-only progress interface.
22. Completed in `v0.24`: persist each inner fold-by-state-count-by-seed model
    and held-out score independently, reuse validated model-only or complete
    cells after interruption, and report both fit and score progress without
    changing the final nested-result schema.
