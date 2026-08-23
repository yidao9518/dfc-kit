# XCP-D input contract

## Scope

`dfc-kit` starts from XCP-D parcellated derivatives. It does not repeat
confound regression, temporal filtering, censoring, interpolation, or atlas
extraction from fMRIPrep BOLD images.

The expected analysis chain is:

```text
BIDS -> fMRIPrep -> XCP-D -> dfc-kit
```

## Required files

For every requested atlas, the adapter requires:

```text
sub-*_task-*_space-*_atlas-<atlas>_stat-mean_timeseries.tsv
sub-*_task-*_space-*_atlas-<atlas>_stat-coverage_bold.tsv
```

The acquisition must also have one original-axis outlier mask:

```text
sub-*_task-*_outliers.tsv
```

`*_motion.tsv` is discovered when present and supports covariate matching. It
is not currently required for FC estimation.

## Time-axis handling

XCP-D derivatives occur in two valid forms:

1. The atlas table has one row per original frame. The adapter removes rows
   marked in `outliers.tsv`.
2. With `--output-type censored`, the atlas table already contains only
   retained rows. The adapter uses `outliers.tsv` to recover their original
   frame indices.

Any other row count is rejected. Temporal estimators receive these recovered
indices and cannot construct a window, derivative, or transition across a
censor gap.

## Atlas and coverage handling

One run may combine multiple XCP-D atlases. ROI columns are selected by exact
name and returned in the explicitly requested order. Duplicate names across
atlases, missing columns, missing coverage nodes, and ambiguous spaces are
errors.

`minimum_coverage` checks the selected ROIs against each atlas's `Node` and
`coverage` table. It does not silently remove parcels because session-specific
parcel removal would destabilize ROI and edge order across participants.

## Public API

```python
from dfckit.io import load_xcpd_run

loaded = load_xcpd_run(
    "/path/to/xcp_d",
    subject="sub-001",
    session="off",
    atlases=("Glasser", "Tian"),
    space="MNI152NLin2009cAsym",
    roi_names={
        "Glasser": ("Left_V1", "Right_V1"),
        "Tian": ("PUT-DP-lh", "PUT-DP-rh"),
    },
    minimum_coverage=0.5,
    tr=0.75,
)

run = loaded.run
```

`run` is the file-independent `TimeSeriesRun` consumed by all connectivity and
state estimators. `loaded.coverage`, `loaded.files`, and `loaded.source_axes`
retain the XCP-D input provenance.

## Multiple acquisitions

`discover_xcpd_runs()` scans the derivative tree and returns one file bundle per
acquisition stem. Optional `subject`, `session`, `task`, and `space` filters are
applied to BIDS entities. `load_xcpd_dataset()` loads those bundles into a
`TimeSeriesDataset` and sets `run.acquisition_id` to the stem before the
`_space-...`/`_atlas-...` derivative suffix, for example
`sub-001_ses-off_task-rest_run-2`. This makes repeated runs in one session
coexist without merging their state sequences.
