# Architecture

## Separation of concerns

The library has five layers:

1. **XCP-D input** discovers parcellated derivatives, validates atlas coverage,
   and reconstructs the original retained-frame axis from outlier masks.
2. **Data topology** describes retained samples, their original frame indices,
   ROI identity, and censor-bounded segments.
3. **Estimators** transform time series or windows into FC, events, phases,
   subspaces, or state representations.
4. **Metrics** summarize representations at run or subject level.
5. **Inference** compares metrics while making the permutation unit and
   multiple-comparison group explicit.

Persisted feature stores sit between estimators and state models: they retain
feature identity and censor-segment metadata while allowing large FC/ETS
outputs, MiniBatchKMeans fits, and IncrementalPCA/HMM preprocessing to be
traversed as memory-mapped chunks. HMM fitting materializes only the reduced
observations and preserves one length per uninterrupted sequence.

Fitted-model artifacts sit after learning. They store JSON metadata and numeric
NumPy arrays without pickle, preserve fitted participant IDs, and restore the
same feature/source contract for held-out prediction. They do not replace
FeatureStore or serialize study-specific outputs.

XCP-D derivative discovery belongs to the library. Disease-specific ROI
selection, clinical-table parsing, fMRIPrep denoising, and manuscript output are
application concerns and do not belong in the library.

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

## Migration rule

Research scripts are not moved into this repository. A method is reimplemented
from a selected canonical definition, tested mathematically on synthetic data,
and then compared against a frozen research output. Only after numerical
equivalence is demonstrated may a research runner optionally adopt the released
library in a separate change.
