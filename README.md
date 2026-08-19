# dfc-kit

`dfc-kit` is an open-source Python toolkit for dynamic functional connectivity
analysis of XCP-D parcellated derivatives. It provides composable estimators,
state models, network summaries, statistical inference, and command-line
workflows for reproducible neuroimaging analysis.

```text
BIDS -> fMRIPrep -> XCP-D -> dfc-kit
```

## Why dfc-kit

Temporal analysis is **censor-aware by construction**. The toolkit restores the
original frame axis before temporal operations and keeps windows, derivatives,
phase features, and state transitions inside contiguous retained-frame
segments. Learned scalers, templates, and state centers record their fitting
participants and reject accidental overlap between fitting and evaluation
cohorts.

## Features

- **Input and topology:** XCP-D discovery and validation, multi-atlas ROI
  loading, retained-frame reconstruction, acquisition identity, and
  censor-bounded sequences.
- **Connectivity:** weighted sliding-window FC, MTD, ETS, LEiDA, low-rank
  covariance geometry, and fixed-length MI/CMI.
- **Network and state analysis:** partition-based graph metrics, CAP, KMeans,
  Gaussian HMMs, state alignment, occupancy/dwell/transition summaries, and
  subject-disjoint state-count validation.
- **Inference and QC:** paired sign-flips, bootstrap intervals, HC3 models,
  FDR, experimental paired NBS, and within-subject motion matching.
- **Scalable and portable outputs:** append-only memory-mapped FeatureStores,
  bounded-memory fitting, pickle-free model artifacts, held-out prediction
  records, and provenance fingerprints.

The [method inventory](docs/method_inventory.md) maps each method family to its
public API and guide. NBS is explicitly marked experimental; the remaining
public data, connectivity, state, reference, QC, and inference objects are
covered by the package test suite and documented contracts.

## Scope

The supported input boundary is XCP-D output. `dfc-kit` does not reimplement
fMRIPrep-to-XCP-D denoising, filtering, censoring, interpolation, or
parcellation. Callers provide ROI definitions, cohort labels, clinical
variables, and manuscript-specific analyses around the library's numerical
interfaces. The array API is also available for equivalently preprocessed ROI
time series that are not stored as XCP-D derivatives.

## Installation

```bash
python -m pip install dfc-kit
```

Install only the optional method families required by an analysis:

```bash
python -m pip install 'dfc-kit[phase,states,hmm,information,qc]'
```

Python 3.10 or newer is required. See [Getting started](docs/getting_started.md)
for development installation and dependency details.

## Quick start

```python
from dfckit.connectivity import SlidingWindowFC
from dfckit.io import load_xcpd_run

loaded = load_xcpd_run(
    "/path/to/xcp_d",
    subject="sub-001",
    session="01",
    task="rest",
    atlases=("Schaefer200",),
    space="MNI152NLin2009cAsym",
    minimum_coverage=0.5,
    tr=0.8,
)

result = SlidingWindowFC(length=60, step=10, taper="hamming").transform(loaded.run)
print(result.features.shape)
print(result.start_frames, result.end_frames, result.segment_ids)
```

The result contains Fisher-z upper-triangle edges and the original-frame bounds
of every valid window. For a complete path from XCP-D discovery through state
fitting, see the [XCP-D-to-state tutorial](docs/tutorial_xcpd_to_states.md).

## Command line

The `dfc-kit` command exposes XCP-D inspection, FeatureStore construction,
state fitting, held-out prediction, scoring, alignment, and state-count
validation. Start with:

```bash
dfc-kit --help
dfc-kit inspect-xcpd --help
dfc-kit build-store --help
```

See [Command-line workflows](docs/cli.md) for complete examples and arguments.

## Documentation

- [Documentation home](docs/index.md)
- [XCP-D input contract](docs/xcpd_input.md)
- [Connectivity methods](docs/correlation.md)
- [State models and validation](docs/states.md)
- [API map](docs/api.md)
- [Release process](docs/release.md)

## Development

```bash
python -m pip install -e '.[all,dev,docs]'
python -m unittest discover
ruff check src tests
mkdocs build --strict
```

`dfc-kit` is distributed under the BSD-3-Clause license. See `LICENSE` and
`CITATION.cff` for licensing and citation information.
