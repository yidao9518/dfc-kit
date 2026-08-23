# dfc-kit

`dfc-kit` is an open-source Python toolkit for censor-aware dynamic functional
connectivity analysis of XCP-D parcellated derivatives.

```text
BIDS -> fMRIPrep -> XCP-D -> dfc-kit
```

The package reconstructs the original retained-frame axis before any temporal
operation. Sliding windows, derivatives, Hilbert phase, low-rank comparisons,
and state transitions therefore remain inside uninterrupted segments rather
than silently crossing censored frames. Learned transformations carry explicit
participant identities and reject accidental overlap between fitting and
evaluation cohorts.

## What is included

The toolkit is organized around four public layers:

- **Data and topology:** XCP-D discovery, ROI loading, retained-frame
  reconstruction, acquisition identity, and censor-bounded sequences.
- **Connectivity:** sliding-window FC, one instantaneous-edge
  pipeline with ETS and MTD sample generators, LEiDA, low-rank geometry,
  MI/CMI, partition metrics, and healthy-reference utilities.
- **State models:** CAP, KMeans, Gaussian HMMs, state alignment, held-out
  scoring, repeated-fit stability, and participant-disjoint state-count
  selection.
- **Inference and persistence:** paired inference, covariate matching,
  paired NBS, memory-mapped FeatureStores, and portable model artifacts.

## Scope

The package starts after XCP-D. It does not reimplement denoising, filtering,
censoring, interpolation, or parcellation. Cohort inclusion, disease-specific
ROI definitions, clinical variables, plots, and manuscript logic belong in the
downstream study repository.

Start with [Getting started](getting_started.md), inspect the [XCP-D input
contract](xcpd_input.md), or follow the complete [XCP-D-to-state
tutorial](tutorial_xcpd_to_states.md). The [method inventory](method_inventory.md)
provides the full method-to-API map.
