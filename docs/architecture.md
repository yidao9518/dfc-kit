# Architecture

## Separation of concerns

The library has five layers:

1. **XCP-D input** discovers parcellated derivatives, validates atlas coverage,
   and reconstructs the original retained-frame axis from outlier masks.
2. **Data topology** describes retained samples, their original frame indices,
   ROI identity, and censor-bounded segments.
3. **Estimators** transform time series or windows into FC, phase, subspace,
   or state representations. ETS and MTD construct different time-aligned
   rows, then return the same instantaneous-edge result for summaries, state
   conversion, and storage.
4. **Metrics** summarize representations at run or subject level.
5. **Inference** compares metrics while making the permutation unit and
   multiple-comparison group explicit.

Persisted feature stores sit between estimators and state models: they retain
feature identity and censor-segment metadata while allowing large FC and
instantaneous-edge outputs, MiniBatchKMeans fits, and IncrementalPCA/HMM
preprocessing to be traversed as memory-mapped chunks. HMM fitting materializes
only the reduced observations and preserves one length per uninterrupted
sequence.

Fitted-model artifacts sit after learning. They store JSON metadata and numeric
NumPy arrays without pickle, preserve fitted participant IDs, and restore the
same feature/source contract for held-out prediction. They do not replace
FeatureStore or serialize study-specific outputs.

XCP-D derivative discovery belongs to the library. Disease-specific ROI
selection, clinical-table parsing, fMRIPrep denoising, and manuscript output are
application concerns and do not belong in the library.

## Internal package layout

The command line is a thin dispatch layer rather than an implementation
module. `dfckit.cli` owns parser dispatch, while the implementations are
organized under `dfckit.commands`:

- `source` handles XCP-D discovery, ROI loading, FeatureStore writers, and
  fixed-information artifacts.
- `states` handles state-model fitting, prediction, scoring, and metrics.
- `reporting` handles endpoint summaries, state descriptions, and inference.
- `states` contains state-count comparison and subject-disjoint selection.
- `stability` handles state alignment and repeated-fit summaries.

The public package boundaries follow ownership rather than file format:

- `dfckit.io` contains only XCP-D discovery and loading.
- `dfckit.artifacts` persists fitted models, decoded states, scores, alignments,
  and stability summaries.
- `dfckit.states` compares state counts and reconstructs fold-level summaries.
- `dfckit.information` owns fixed-length MI/CMI estimators, censor-safe window
  sampling, acquisition-level orchestration, and information artifacts.

This layout keeps numerical kernels and artifact schemas independent from
argument parsing. The executable adapts the parsed namespace to the focused
implementation without putting command-specific logic into numerical modules.

## Core invariants

- `values[row]` corresponds exactly to `original_indices[row]`.
- Original indices are strictly increasing.
- A segment contains only consecutive original indices.
- A temporal window belongs to one segment.
- Derivatives and transitions never cross segment boundaries.
- ROI names and edge indices are part of every transferable feature contract.
- Acquisition identity is carried with every run-level result and state
  sequence; no state transition or run summary crosses an acquisition boundary.
- Learned transformations expose the participant grouping used for fitting.
- Randomized methods record their random seed and complete configuration.

## Output model

Initial transforms return immutable dataclasses carrying arrays and metadata.
The XCP-D adapter converts BIDS-derivative files into the same immutable core
objects used by synthetic tests. Numerical kernels therefore remain independent
from file naming and local directory layouts even though XCP-D is the supported
production input.

## Extension and validation

New estimators should be defined from a documented mathematical contract,
tested on synthetic data, and accompanied by explicit metadata for frame
indices, segment identity, feature order, and participant grouping. When a
reference implementation is available, numerical regression tests can be kept
alongside the estimator without making that application-specific workflow a
runtime dependency of the library.
