# Subject-safe datasets

`dfc-kit` loads each acquisition from XCP-D into a `TimeSeriesRun`. A
`TimeSeriesDataset` then combines runs only when they share the same ROI names,
ROI order, and TR. The dataset preserves session labels but assigns them no
clinical meaning. Each run also carries an optional `acquisition_id`; XCP-D
loads derive it from the acquisition filename stem, preserving `task`, `acq`,
`run`, and other BIDS distinctions.

```python
from dfckit import TimeSeriesDataset, validate_subject_disjoint
from dfckit.io import load_xcpd_run

off = load_xcpd_run(
    "/path/to/xcp_d",
    subject="sub-001",
    session="off",
    atlases="Glasser",
    tr=0.8,
).run
on = load_xcpd_run(
    "/path/to/xcp_d",
    subject="sub-001",
    session="on",
    atlases="Glasser",
    tr=0.8,
).run

dataset = TimeSeriesDataset([off, on])
print(dataset.runs_by_subject()["sub-001"])
```

For a derivative tree containing multiple runs, discover and load the complete
set in one call:

```python
from dfckit.io import load_xcpd_dataset

dataset = load_xcpd_dataset(
    "/path/to/xcp_d",
    subject="sub-001",
    atlases="Glasser",
    task="rest",
    space="MNI152NLin2009cAsym",
    tr=0.8,
)
for run in dataset.runs:
    print(run.acquisition_id)
```

Any estimator that learns a scaler, projection, template, state center, or
hyperparameter must require subject identifiers. Before fitting, validate the
actual run collections used by a split:

```python
validate_subject_disjoint(train_runs, test_runs)
```

The validator rejects the split when, for example, `sub-001/ses-off` is used
for training and `sub-001/ses-on` is used for testing. Session separation is
not participant separation.

Two runs are duplicates only when their complete identity
`(subject, session, acquisition_id)` is equal. An omitted `acquisition_id`
retains the conservative legacy behavior and cannot be used twice for one
subject/session. State sequences extend this identity with `segment_id`, so
state transitions and dwell summaries never cross either censor gaps or
acquisition boundaries.
