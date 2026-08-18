# Getting started

## Installation

Install the core package:

```bash
python -m pip install dfc-kit
```

Install method-specific extras as needed:

```bash
python -m pip install 'dfc-kit[phase,states,hmm,information,qc]'
```

Fixed-partition graph metrics are included in the core NumPy-only package.

For a development checkout:

```bash
python -m pip install -e '.[all,dev,docs]'
```

Python 3.10 or newer is required.

## Load one XCP-D run

```python
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
run = loaded.run
print(run.acquisition_id)
```

Exact atlas and ROI selection depends on the derivatives supplied by XCP-D.
See [XCP-D input](xcpd_input.md) for the file and censor-axis contract.

## Compute sliding-window FC

```python
from dfckit.connectivity import SlidingWindowFC

result = SlidingWindowFC(length=60, step=10, taper="hamming").transform(run)
print(result.features.shape)
print(result.start_frames, result.end_frames, result.segment_ids)
```

The result contains Fisher-z upper-triangle edges and the exact original-frame
bounds of every valid window.

## Choose a state model

Convert one or more window-FC results into segment-bounded feature sequences,
then fit only on explicitly identified participants:

```python
from dfckit.states import (
    fit_kmeans_states,
    window_fc_sequences,
)

sequences = window_fc_sequences([result])
fitted = fit_kmeans_states(sequences, n_states=4, seed=17)
assignments = fitted.assignments
```

Use participant-disjoint datasets for validation. The model rejects prediction
on participants recorded during fitting unless the API explicitly represents a
training-set operation.
