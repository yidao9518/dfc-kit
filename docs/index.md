# dfc-kit

`dfc-kit` is a Python toolkit for censor-aware dynamic functional connectivity
analysis of XCP-D parcellated derivatives.

```text
BIDS -> fMRIPrep -> XCP-D -> dfc-kit
```

The package reconstructs the original retained-frame axis before any temporal
operation. Sliding windows, derivatives, Hilbert phase, low-rank comparisons,
and state transitions therefore remain inside uninterrupted segments rather
than silently crossing censored frames.

## What is included

- weighted sliding-window FC, MTD, ETS, LEiDA, and low-rank covariance geometry;
- fixed-partition graph metrics and subject-balanced healthy references;
- KMeans, CAP, and optional Gaussian HMM state models;
- complete KMeans/HMM state relabeling across seeds or resamples;
- append-only feature stores and out-of-core MiniBatchKMeans fitting;
- streaming PCA and Gaussian HMM fitting with explicit sequence lengths;
- pickle-free fitted-model artifacts with participant-safe restoration;
- held-out state prediction artifacts and JSON occupancy/dwell/transition
  metrics with sequence-boundary preservation;
- repeated-fit state decoding, reference alignment, and metric-dispersion JSON;
- repeated-fit, subject-disjoint state-count selection with fold-only uncertainty;
- deterministic end-to-end state-count cross-validation from one FeatureStore;
- full-development refitting and explicit outer-test evaluation of an
  inner-selected state count;
- complete nested state-count cross-validation with one outer-test score per
  participant and strict child-workflow reconstruction;
- strict checkpoint/resume for interrupted nested evaluation, with per-inner-cell
  model/score reuse, complete-fold reuse, and drift rejection;
- read-only nested progress audits that bind separate inner fit/score completion
  to the current FeatureStore;
- cross-process checkpoint ownership with active/stale execution visibility and
  crash-safe reclamation;
- fixed-length MI/CMI and experimental paired NBS;
- paired permutation, bootstrap, HC3, FDR, and within-subject motion matching;
- explicit participant-level train/test separation for learned models.
- explicit acquisition identity across repeated task/acq/run recordings and
  state boundaries.

## Scope

The package starts after XCP-D. It does not reimplement denoising, filtering,
censoring, interpolation, or parcellation. Cohort inclusion, disease-specific
ROI definitions, clinical variables, plots, and manuscript logic belong in the
downstream study repository.

Continue with [Getting started](getting_started.md) or the complete
[XCP-D-to-state tutorial](tutorial_xcpd_to_states.md).
