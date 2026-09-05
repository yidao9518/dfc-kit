# Changelog

## 1.0.1 (unreleased)

- Fixed-length MI/CMI now shares one process pool across acquisition, window
  length, and draw combinations. Bounded task submission avoids materializing
  the complete window grid, and ordered results preserve the serial samples,
  jitter, pair estimates, and artifact rows. `jobs=1` remains serial.
- Reused full-window and half-window PCA fits across requested ranks, avoiding
  repeated decompositions while preserving covariance endpoint definitions.
- Shared paired-test, HMM decoding, and feature-building internals; removed
  redundant storage forwarders and derived-score constructor arguments.
- Added `static-fc-endpoints` to compute one whole-acquisition Fisher-z edge
  vector from each XCP-D run's retained frames.
- Added `infer-paired-nbs`, a machine-readable endpoint adapter for paired NBS
  with component-level FWE, explicit thresholds, tails, component statistics,
  and optional participant-level nuisance variables.
- Added `lowrank-endpoints` for direct XCP-D-to-JSON summaries of effective
  rank, eigenvalue concentration, held-out reconstruction, and split/adjacent
  subspace similarity.
- Added segment-weighted all-pair FC-pattern and low-rank subspace similarity,
  together with adjacency excess, so configuration homogeneity can be separated
  from order-specific similarity.
- Added `window-pattern-endpoints` for direct XCP-D-to-JSON summaries of
  all-pair, adjacent, and adjacency-excess whole-edge pattern similarity.
- Added deterministic pass-level early stopping to out-of-core MiniBatchKMeans.
- Reported the selected initialization's completed passes and all initialization
  pass counts without changing the fitted-model artifact format.
- Clarified that streaming `max_iter` is a maximum number of complete store
  passes; `convergence_tol=0` retains fixed-pass fitting.
- Added an explicit equal-run mean for paired inference with repeated
  acquisitions. Runs remain separate through feature and state summarization,
  and JSON/TSV results report the acquisition-count range in each condition.
- Paired named-endpoint inference can now select an explicit endpoint subset,
  allowing one source artifact to be tested as several scientifically declared
  FDR families without changing each endpoint's permutation or bootstrap draw.
- Added `infer-independent-endpoints` for two independent named-endpoint
  cohorts, with optional numeric covariates, HC3 group effects, Hedges' g,
  full and group-incremental R2, and declared-family FDR.
- XCP-D discovery now enters session directories linked into an otherwise
  ordinary BIDS subject hierarchy without recursively following arbitrary
  directory symlinks.

## 1.0.0 - 2026-08-24

- Moved MI/CMI estimators and fixed-window orchestration into
  `dfckit.information`.
- Unified random sampling and frozen schedules through one fixed-window
  estimation path.
- Simplified state-count selection to subject-disjoint held-out score
  comparison with compact JSON output.
- Removed cryptographic data/model identities, workflow-directory loaders,
  nested checkpointing, and legacy compatibility layers from the development
  codebase.
- Kept model, prediction, alignment, and information artifacts pickle-free and
  validated by explicit shapes, parameters, feature keys, and participant
  metadata.
- ETS and MTD share the same segment-safe instantaneous-edge framework while
  retaining their distinct sample generators.
- Paired NBS, HC3 inference, information estimates, and all censor-gap-safe
  connectivity methods retain their numerical APIs.
- FeatureStore summaries support streaming acquisition-level mean, population
  variance, standard deviation, minimum, and maximum endpoints.
