# Roadmap

## Current development target

- Keep the public API small and composable.
- Treat XCP-D derivatives and ROI definitions as the input boundary.
- Keep all temporal operations inside contiguous retained-frame segments.
- Keep KMeans/HMM fitting, decoding, held-out scoring, and stability summaries
  independent from manuscript-specific workflows.
- Select K with subject-disjoint held-out scores and a compact in-memory
  comparison.
- Maintain pickle-free JSON/NumPy artifacts with explicit numerical metadata.
- Expand numerical regression tests for MI/CMI, ETS/MTD, low-rank methods,
  state alignment, and inference.

## Deliberately out of scope

- Resumable workflow directories and nested state-count orchestration.
- Cryptographic identities for arrays, models, or result files.
- Parkinson-specific ROI definitions, clinical variables, figures, and
  manuscript automation.

## Release boundary

The independent `dfc-agent` project may orchestrate this toolkit, but remains
in a separate repository and package.
