# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.25.0] - 2026-08-19

### Added

- A bounded-memory `build-store --method mtd` writer that persists run-level,
  censor-gap-safe MTD edge products in FeatureStore v2 while retaining original
  derivative start/end frames and the MTD source contract.

### Verified

- Agent-driven replay of 116 ROI32 ON/OFF acquisitions reproduced all historical
  MTD session block means within `2.22e-16` and the paired visual--sensorimotor/6d
  result (`Delta=-0.0618114702`, `p=.00859914`, `q=.02579742`) exactly.

## [0.24.1] - 2026-08-19

### Fixed

- Repeated XCP-D `--subject` filters now select all requested subjects rather
  than silently retaining only the last command-line value.
- `inspect-xcpd` inventories retain atlas time-series and coverage paths for
  downstream provenance checks.

## [0.24.0] - 2026-08-18

### Added

- Per-cell persistence for interrupted inner state-count grids in resumable
  nested KMeans and Gaussian-HMM workflows.
- Independent `pending-model`, `pending-score`, and `complete` cell states, so
  a fitted model can be reused even when its held-out score was interrupted.
- Inner model/score completion counts in read-only nested progress reports and
  model/score reuse counts in completed command summaries.
- KMeans and Gaussian-HMM interruption tests covering model-only, scored, and
  pending cells within the same inner grid.

### Changed

- Resume validation now binds every reusable inner cell to its exact fold,
  state count, seed, model family, fit settings, sampling contract, and current
  FeatureStore fingerprints before any new fit begins.
- Completed selections recovered during the final manifest transition are
  revalidated against every current inner training and evaluation fold before
  their internal checkpoint manifest is removed.
- Read-only progress audits now validate a lingering internal cell manifest
  against the completed selection during that same crash window without
  deleting or rewriting it.
- Published nested results retain the existing portable schema; per-cell
  checkpoint manifests remain internal and are removed at finalization.
- Fixed-partition graph metrics remain part of the core NumPy-only package;
  the unused NetworkX installation extra has been removed.
- Release metadata now links to the public repository and issue tracker; CI
  covers Python 3.10, 3.12, and 3.13 and smoke-tests the built wheel.
- CI uses the current Node 24 GitHub actions and invokes Ruff through the
  selected Python matrix interpreter.

## [0.23.0] - 2026-08-18

### Added

- Cross-process advisory ownership for resumable nested checkpoints, with one
  persistent sibling lock record containing PID, hostname, process-start token,
  and acquisition/release timestamps.
- `active`, `idle`, `stale`, and `invalid` execution states in nested progress
  audits, including owner metadata for diagnosing live and crashed workers.
- Process-contention, normal release, hard-exit recovery, stale reclamation,
  malformed record, symlink, non-regular path, and active-progress tests.

### Changed

- `nested-cross-validate-state-counts --checkpoint` now refuses a concurrent
  writer before changing the checkpoint or final output.
- Kernel-released locks from terminated workers are reclaimed safely while the
  unreleased owner record remains available as crash evidence.

## [0.22.0] - 2026-08-18

### Added

- `inspect_nested_state_count_progress` and immutable fold/workflow progress
  objects for read-only inspection of checkpoints and completed nested results.
- `inspect-nested-state-counts` to report per-outer-fold state, complete inner
  fits, complete outer models, temporary paths, split seeds, completion fraction,
  and available final workflow/cohort identities as finite JSON.
- Distinct `in-progress`, `ready-for-promotion`, and `complete` states, counting
  only child artifacts that pass complete model/score/summary reconstruction.
- Current-FeatureStore, KMeans/HMM, manifest and child tampering, symlink,
  temporary-residue, and pre-promotion interruption coverage.

### Changed

- Nested resume and read-only inspection now share one current-data validation
  implementation for inner selections and outer evaluations.

## [0.21.0] - 2026-08-18

### Added

- Optional `--checkpoint` persistence for
  `nested-cross-validate-state-counts`, retaining only complete inner-selection
  and outer-evaluation artifacts across interrupted runs.
- Strict resume contracts covering command settings, FeatureStore feature and
  content fingerprints, outer and inner participant splits, model seeds,
  fitted-model specifications, and every reused child artifact.
- Atomic final-manifest creation, stale temporary cleanup, complete-checkpoint
  promotion without refitting, and symlink-free checkpoint-tree enforcement.
- Failure/resume, parameter and data drift, completed-result promotion,
  internal-symlink, schema, duplicate-field, and non-finite JSON tests.

## [0.20.0] - 2026-08-18

### Added

- `nested-cross-validate-state-counts` to construct deterministic outer
  participant folds and repeat complete inner state-count selection independently
  inside every outer development cohort.
- Atomic full-development refitting and one-time outer-test scoring for each
  participant, with participant-balanced aggregation across unequal outer folds.
- Strict loaders for selected-state-count evaluations and complete nested
  workflows that rebuild summaries from every referenced model and score.
- Portable per-fold inner/outer artifact trees plus KMeans/HMM, path-escape,
  tampering, move, leakage, and mid-workflow failure coverage.

## [0.19.0] - 2026-08-18

### Added

- Strict loading and reconstruction of complete automated state-count
  cross-validation workflows, including split, grid, model, score, comparison,
  relative-path, and fingerprint validation.
- `evaluate-selected-state-count` to apply the best or one-standard-error
  inner decision, refit every prespecified seed on the complete development
  cohort, and score only explicit participant-disjoint outer-test subjects.
- Atomic portable evaluation directories containing full-development fitted
  models, v2 outer-test score artifacts, and a strict `evaluation.json` with
  inner-workflow, development-data, test-data, model, and feature identities.
- Run-sample-weighted, equal-seed-within-participant, and equal-participant
  outer-test aggregation with formula-level tests.
- KMeans and Gaussian-HMM outer-test command tests plus leakage, data-drift,
  tampering, path-escape, overwrite, and failed-fit cleanup coverage.

## [0.18.0] - 2026-08-18

### Added

- `make_subject_validation_folds` with input-order-independent SHA-256
  assignment, balanced fold sizes, and complete participant coverage.
- `cross-validate-state-counts` to fit, persist, score, and compare a complete
  subject-disjoint fold-by-candidate-by-seed KMeans or Gaussian-HMM grid from
  one FeatureStore.
- An atomic portable workflow directory containing fitted models, v2 held-out
  scores, the v2 comparison, and a relative-path manifest with the development
  cohort, split algorithm, fit settings, and artifact identities.
- Explicit `--subject` restriction for running state-count selection inside an
  outer training cohort without exposing the outer test participants.
- Deterministic KMeans repeat-run, HMM, invalid-grid, overwrite, and outer-
  training-boundary command-line tests.

## [0.17.0] - 2026-08-18

### Added

- Complete `fold x candidate K x complete-fit seed` comparison grids for
  KMeans and Gaussian-HMM state-count selection.
- Equal-seed aggregation inside each participant before equal-participant and
  equal-fold aggregation; uncertainty remains estimated only across validation
  folds, so repeated fits are not treated as independent observations.
- Per-seed participant scores, fold scores, model fingerprints,
  specifications, and source score artifacts in comparison JSON format v2.
- Validation that every candidate and fold uses one exact seed set and that
  every development participant appears in exactly one validation fold.
- Multi-seed object-invariant and CLI roundtrip tests plus updated selection
  formulas, examples, and provenance documentation.

## [0.16.0] - 2026-08-18

### Added

- `compare-state-counts` for complete KMeans or Gaussian-HMM candidate
  comparison across subject-disjoint validation folds.
- Participant-balanced within-fold scoring, equal fold weighting, mean-score
  ranking, and the one-standard-error selection rule with smaller-K preference.
- Strict validation of candidate sets, training/evaluation identities, run
  boundaries, model family and specifications, feature/source/time contracts,
  model fingerprints, and cross-fold participant exclusivity.
- Immutable selection and fold-report objects that recompute and validate
  derived means, uncertainty, rankings, candidate alignment, and decisions.
- v2 score artifacts with complete fitted-model specifications and FeatureStore
  feature-contract fingerprints while retaining v1 read compatibility.
- Dedicated state-count selection documentation, CLI/API examples, nested-CV
  boundary guidance, and end-to-end CLI contract tests.

## [0.15.0] - 2026-08-18

### Added

- `score-states` for participant-safe acquisition-level evaluation of frozen
  KMeans and Gaussian HMM artifacts on explicit held-out FeatureStore cohorts.
- Standardized KMeans total/mean quantization error and Gaussian-HMM total/per-
  sample log likelihood, with sample-weighted pooled summaries.
- Independent HMM likelihood evaluation at every censor-bounded sequence,
  minimum-length accounting, and rejection when a requested participant has no
  eligible sequence.
- Strict JSON model/evaluation-data fingerprints, fit/evaluation identities,
  scoring direction, short-sequence counts, and overwrite protection.
- Direct-formula, chunk-invariance, gap-restart, leakage, CLI, and persistence
  tests plus state-count selection documentation.

## [0.14.0] - 2026-08-18

### Added

- `summarize-stability` for common-store decoding, reference-model Hungarian
  alignment, and repeated-fit occupancy/dwell/switch/transition dispersion.
- Strict state-stability JSON with exact model fingerprints, mappings, complete
  correlation matrices, valid-fit counts, and JSON-null missing values.
- Fitted-model artifact v2 provenance for exact training data, effective KMeans
  initialization sample size, and HMM IncrementalPCA batch size, with v1 read
  compatibility.
- Exact sequence, segment, and original sample-index equality checks before
  metrics from separate fits can be stacked.
- KMeans/HMM end-to-end tests plus family, training-contract, duplicate-model,
  legacy-provenance, output-overwrite, and zero-residual-DF rejection tests.

## [0.13.0] - 2026-08-18

### Added

- Deterministic fitted-model fingerprints over model kind, parameters,
  training metadata, algorithm settings, and sampling contract.
- Prediction artifact v3 exact model identity, while v1/v2 artifacts remain
  readable for summarization.
- Alignment refusal for equal-seed predictions generated by a different model,
  training cohort, or hyperparameter configuration.
- Fingerprint determinism, save/load stability, format-version selection, and
  same-seed wrong-model rejection tests.

## [0.12.0] - 2026-08-18

### Added

- `align-states` for KMeans-centroid or Gaussian-HMM emission alignment from
  saved model artifacts, with synchronized model and optional prediction
  relabeling.
- Pickle-free alignment artifacts preserving the permutation, complete and
  matched correlation arrays, seeds, and sampling contract.
- Prediction artifact v2 model-seed identity, with backward reading of v1
  artifacts for summarization and explicit refusal to align unknown-seed data.
- CLI and persistence tests for KMeans/HMM parameters, Viterbi labels,
  posterior columns, path pairing, overwrite boundaries, and v1 compatibility.

## [0.11.0] - 2026-08-18

### Added

- `predict-states` for participant-safe KMeans/HMM FeatureStore decoding, with
  an explicit `--allow-fit-subjects` override.
- Strict, pickle-free state-prediction artifacts preserving acquisition,
  segment, and original sample-index metadata; HMM artifacts also retain
  posterior probabilities and log likelihood.
- `summarize-states` for JSON occupancy, dwell, switching, and transition
  metrics with censor-gap boundaries preserved.
- State-result persistence, CLI leakage-boundary, posterior-validation, and
  metric serialization tests.

## [0.10.0] - 2026-08-18

### Added

- `dfc-kit fit-states` for out-of-core MiniBatchKMeans and PCA-reduced Gaussian
  HMM fitting from existing FeatureStores.
- Explicit training-subject selection, required random seeds, method-specific
  hyperparameters, machine-readable fit diagnostics, and absent-subject checks.
- Direct creation of pickle-free fitted-model artifacts with overwrite
  protection and CLI roundtrip tests for both state-model families.

## [0.9.0] - 2026-08-18

### Added

- Dependency-light `dfc-kit` command with `inspect-xcpd` and `build-store`
  subcommands.
- JSON ROI selection validation and machine-readable acquisition/store summaries.
- CLI tests covering repeated acquisitions, window-FC output, and invalid input.

## [0.8.0] - 2026-08-18

### Added

- Explicit optional acquisition IDs on runs, connectivity results, state
  sequences, state metrics, and out-of-core chunks.
- FeatureStore manifest v2 with acquisition-aware identities and backward
  reading plus atomic v1-to-v2 upgrade on append.
- Batch XCP-D discovery and dataset loading across repeated task/acq/run
  acquisitions.
- Tests for repeated same-session acquisitions, state-metric isolation, legacy
  store compatibility, and batch XCP-D loading.

## [0.7.0] - 2026-08-18

### Added

- Original-feature emission-mean alignment for repeated Gaussian HMM fits.
- Complete HMM relabeling across start probabilities, transition rows/columns,
  reduced/original emissions, optional covariances, decoded labels, and
  posterior columns.
- KMeans model relabeling for reference-numbered future predictions.
- Validated immutable state-alignment objects and explicit sampling-contract
  checks across candidate/reference models.
- Permutation-equivalence, compact-HMM, posterior-validation, and
  decode-before/after-alignment tests plus dedicated state-alignment docs.

## [0.6.0] - 2026-08-18

### Added

- Pickle-free fitted-model artifacts for KMeans state models, streaming PCA,
  and Gaussian HMM state models.
- Exact manifest/array schema validation, overwrite protection, read-only
  restored parameters, and preservation of fit-participant leakage guards.
- Roundtrip and frozen-prediction tests for compact and dense HMM covariance
  representations, malformed artifacts, and object-array rejection.
- Model-artifact API documentation and XCP-D-to-state persistence tutorial.

## [0.5.0] - 2026-08-18

### Added

- `StreamingPCAModel`, IncrementalPCA fitting, and metadata-preserving reduced
  feature chunks over selected FeatureStore sequences.
- Out-of-core Gaussian HMM fitting and held-out decoding with frozen scaler/PCA
  parameters, explicit censor-bounded sequence lengths, and participant-overlap
  protection.
- Validation for constant features, minimum sequence length, chunk-boundary
  invariance, reduced-observation likelihoods, and HMM posterior outputs.
- Streaming PCA/HMM API documentation, a high-dimensional synthetic benchmark,
  and XCP-D-to-states examples for repeated-seed alignment and stability.

## [0.4.0] - 2026-08-18

### Added

- Append-only, memory-mappable feature stores with bounded-memory FC and ETS
  writers.
- Streaming MiniBatchKMeans fitting from feature stores using Welford
  standardization, deterministic uniform initialization samples, k-means++
  centres, multiple initializations, and full-store inertia selection.
- Feature-store prediction that materializes labels and original sample indices
  without reconstructing the feature matrix, with participant-overlap checks.
- Storage and out-of-core state-model tests, documentation, and synthetic
  profiling notes.

## [0.3.0] - 2026-08-18

### Added

- XCP-D-first loading with atlas coverage and censor-axis validation.
- Immutable run, dataset, segment, and window data structures.
- Weighted FC, MTD, ETS, LEiDA, and low-rank covariance estimators.
- Fixed-partition graph metrics and healthy-reference utilities.
- KMeans, CAP, and optional Gaussian HMM state models with participant-safe fits.
- Gap-safe state occupancy, dwell, transition, switch, and alignment metrics.
- Participant-level sign-flip, bootstrap, HC3, BH-FDR, and experimental NBS.
- Fixed-length Kraskov MI and Frenzel-Pompe CMI.
- Within-participant window matching and motion summaries.
- Memory-mappable chunked feature storage and bounded-memory FC/ETS writers.
- Reproducible synthetic runtime and memory benchmarks.
- Synthetic, integration, and frozen numerical-regression tests.
