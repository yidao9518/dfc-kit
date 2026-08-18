# Roadmap

## v0.1: reliable core

- [x] Repository and package skeleton
- [x] Validated time-series data model
- [x] Censor-bounded segments and windows
- [x] Weighted FC and Fisher-z edges
- [x] XCP-D-first adapter with explicit full-length/censored handling
- [x] Subject-grouped datasets and participant-overlap validation
- [x] MTD
- [x] ETS
- [x] KMeans/CAP state models
- [x] Occupancy, dwell, transition, and switch metrics
- [x] Hungarian state alignment
- [x] Paired sign-flip, bootstrap CI, BH-FDR, and HC3
- [x] Golden regression fixtures against frozen audited outputs
- [x] End-to-end tutorial

## v0.2: continuous dynamic organization

- [x] Low-rank covariance subspaces
- [x] LEiDA
- [x] Fixed-partition graph metrics
- [x] Healthy-reference utilities
- [x] Within-subject motion matching

## v0.3: optional and experimental methods

- [x] Gaussian HMM optional dependency
- [x] Fixed-length MI/CMI
- [x] NBS experimental module
- [x] Documentation site and public release preparation

## v0.4: scalable feature workflows

- [x] Reproducible synthetic time and memory benchmarks
- [x] Append-only memory-mapped feature store
- [x] Bounded-memory sliding-window FC writer
- [x] Bounded-memory complete ETS writer
- [x] Streaming MiniBatchKMeans state-model fitting from feature stores

## v0.5: streaming sequence models

- [x] IncrementalPCA scaler and transform over selected feature-store sequences
- [x] Feature-store Gaussian HMM fitting with explicit sequence lengths
- [x] Feature-store held-out HMM decoding with frozen preprocessing
- [x] Chunk-boundary, short-sequence, constant-feature, and overlap tests
- [x] High-dimensional PCA/HMM benchmark and repeated-seed alignment tutorial

## v0.6: portable fitted models

- [x] Pickle-free KMeans, streaming PCA, and Gaussian HMM artifacts
- [x] Strict version, metadata, array, shape, and probability validation
- [x] Overwrite protection and read-only restored model parameters
- [x] Roundtrip prediction, corruption, and object-array rejection tests
- [x] Persistence API and XCP-D-to-state tutorial documentation

## v0.7: complete state alignment

- [x] Validated candidate-to-reference state permutations
- [x] KMeans center/model relabeling for future predictions
- [x] HMM matching in the common original-feature emission space
- [x] HMM start, transition, emission, covariance, label, and posterior relabeling
- [x] Decode-before/after-alignment equivalence and compact-HMM tests
- [x] Dedicated state-alignment API documentation

## v0.8: explicit acquisition identity

- [x] Acquisition-aware run, connectivity, state, and out-of-core metadata
- [x] FeatureStore v2 identity with v1 read/append compatibility
- [x] Batch XCP-D acquisition discovery and dataset loading
- [x] Repeated same-session run and boundary-isolation tests

## v0.9: executable XCP-D workflows

- [x] Dependency-light `inspect-xcpd` command with JSON acquisition inventory
- [x] `build-store` command for sliding-window FC and ETS
- [x] JSON ROI-selection validation and machine-readable build summaries
- [x] CLI unit and end-to-end tests

## v0.10: executable state fitting

- [x] `fit-states` command for FeatureStore MiniBatchKMeans fitting
- [x] `fit-states` command for FeatureStore PCA/Gaussian HMM fitting
- [x] Explicit fitted-subject selection and absent-subject rejection
- [x] Pickle-free artifact output with machine-readable fit summaries
- [x] CLI artifact roundtrip and failure-boundary tests

## v0.11: executable state prediction and metrics

- [x] Strict state-prediction artifact with sequence/acquisition/index metadata
- [x] `predict-states` held-out KMeans/HMM command with overlap protection
- [x] HMM posterior and log-likelihood persistence
- [x] `summarize-states` gap-safe JSON metrics command
- [x] Persistence, CLI, posterior, and metrics tests

## v0.12: executable state alignment

- [x] Prediction artifact v2 model-seed identity with v1 read compatibility
- [x] Pickle-free state-alignment audit artifacts
- [x] `align-states` KMeans/HMM model relabeling
- [x] Optional synchronized label and HMM posterior relabeling
- [x] Seed, family, path, overwrite, and roundtrip boundary tests

## v0.13: exact model identity

- [x] Deterministic fitted-model parameter and contract fingerprints
- [x] Prediction artifact v3 exact source-model identity
- [x] v1/v2 prediction read and summarize compatibility
- [x] Same-seed different-model alignment rejection
- [x] Fingerprint determinism and artifact roundtrip tests

## v0.14: repeated-fit stability

- [x] Reference-fingerprint state-numbering identity
- [x] Common-store held-out decoding and internal Hungarian alignment
- [x] Per-run cross-fit occupancy, dwell, switch, and transition dispersion
- [x] Strict JSON output with mappings, correlation matrices, and null handling
- [x] Fitted-model v2 training-data and initialization/PCA provenance
- [x] Family, training-contract, exact-boundary, and overwrite rejection tests

## v0.15: held-out state-model scoring

- [x] Run-level held-out KMeans standardized quantization error
- [x] Run-level Gaussian-HMM total and per-sample log likelihood
- [x] Independent HMM scoring at every censor-bounded sequence
- [x] `score-states` CLI with model and evaluation-data fingerprints
- [x] Short-sequence accounting and participant-overlap protection
- [x] Direct-formula, chunk-invariance, gap, JSON, and overwrite tests

## v0.16: cross-fold state-count selection

- [x] Complete candidate-set comparison across subject-disjoint validation folds
- [x] Participant-balanced within-fold and equal-weight cross-fold aggregation
- [x] Best-score ranking and the one-standard-error rule
- [x] Exact feature, training, evaluation, model, and candidate provenance checks
- [x] `compare-state-counts` CLI and strict JSON comparison artifact
- [x] Object-invariant, CLI, schema, fold-boundary, and overwrite tests
- [x] Nested-cross-validation usage and final-test separation documentation

## v0.17: repeated-fit state-count selection

- [x] Complete fold-by-candidate-by-seed scoring grids
- [x] Equal-seed aggregation inside participants before fold selection
- [x] Fold-only uncertainty for the one-standard-error rule
- [x] Per-seed scores, model identities, specifications, and artifact paths
- [x] Exact seed-grid and complete validation-cohort enforcement
- [x] Multi-seed immutable-object and CLI roundtrip tests

## v0.18: automated state-count cross-validation

- [x] Deterministic balanced participant-level fold construction
- [x] Atomic KMeans and Gaussian-HMM fit-score-compare workflow
- [x] Complete fold-by-candidate-by-seed model and score persistence
- [x] Relative-path workflow manifest and portable output directory
- [x] Explicit outer-training participant restriction
- [x] KMeans reproducibility, HMM, leakage-boundary, and overwrite tests

## v0.19: selected state-count outer evaluation

- [x] Strict reconstruction of automated inner-selection workflows
- [x] Best or one-standard-error decision loading without outer-test access
- [x] Full-development refitting for every prespecified model seed
- [x] Explicit participant-disjoint outer-test scoring
- [x] Portable strict evaluation artifact with hierarchical score aggregation
- [x] KMeans/HMM, leakage, drift, tampering, and atomic-failure tests

## v0.20: complete nested state-count evaluation

- [x] Deterministic participant-level outer-fold construction
- [x] Independent inner state-count selection inside every outer development cohort
- [x] Full-development refitting and one-time outer-test scoring per participant
- [x] Equal-participant aggregation across unequal outer-fold sizes
- [x] Strict reconstruction of every inner workflow, outer model, score, and summary
- [x] KMeans/HMM, portability, tampering, path, leakage, and atomic-failure tests

## v0.21: resumable nested state-count evaluation

- [x] Explicit sibling checkpoint directory for long nested workflows
- [x] Complete inner-selection and outer-evaluation artifact reuse
- [x] Command, FeatureStore, split, seed, model, and fold-data drift rejection
- [x] Atomic final-manifest creation and completed-checkpoint promotion
- [x] Stale-temporary cleanup and symlink-free checkpoint-tree enforcement
- [x] Interrupted-run, zero-refit resume, tampering, and path-boundary tests

## v0.22: nested workflow progress audit

- [x] Immutable workflow and per-outer-fold progress objects
- [x] Read-only checkpoint and completed-result inspection CLI
- [x] Complete-artifact-only fit counts and completion fraction
- [x] Temporary-path reporting without mutation or partial-fit credit
- [x] Current FeatureStore, split, model, child-artifact, and path validation
- [x] KMeans/HMM, drift, tampering, ready-for-promotion, and symlink tests

## v0.23: concurrent checkpoint ownership

- [x] Cross-process advisory lock for each resumable nested checkpoint
- [x] Persistent PID, hostname, process-start, acquisition, and release record
- [x] Active, idle, stale, and invalid execution-state inspection
- [x] Automatic kernel release and safe stale-record reclamation after crashes
- [x] Concurrent writer, hard-exit, malformed-record, and unsafe-path tests

## v0.24: per-cell inner-grid recovery

- [x] Persistent inner fold-by-state-count-by-seed model and score cells
- [x] Independent pending-model, pending-score, and complete cell states
- [x] Exact fold, configuration, sampling, model, and current-data validation
- [x] Inner model/score completion and reuse reporting
- [x] KMeans/HMM interruption, partial-score, and tampering tests
- [x] Internal checkpoint removal with unchanged published-result schema

## Release boundary

Parkinson-specific ROI definitions, clinical variables, neuromaps analyses,
figures, and manuscript workflows remain outside this repository.
