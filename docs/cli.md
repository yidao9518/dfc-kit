# Command-line workflows

The command line is a thin wrapper around the Python API. It covers XCP-D
discovery, FeatureStore construction, information estimates, state fitting,
decoding, scoring, alignment, repeated-fit stability, and paired inference.

## Inspect and build a FeatureStore

```bash
dfc-kit inspect-xcpd /path/to/xcpd \
  --atlas Schaefer200 \
  --space MNI152NLin2009cAsym

dfc-kit build-store /path/to/xcpd /path/to/features.store \
  --atlas Schaefer200 \
  --space MNI152NLin2009cAsym \
  --method window-fc \
  --window-length 60 \
  --window-step 10
```

All temporal methods preserve original frame indices and never cross a censor
gap. `--method` accepts `window-fc`, `cap`, `ets`, `mtd`, or `leida`.

## Fit, decode, and score states

```bash
dfc-kit fit-states /path/to/features.store /path/to/models/k4.model \
  --method kmeans --n-states 4 --seed 17

dfc-kit predict-states /path/to/features.store /path/to/models/k4.model \
  /path/to/results/k4.labels --subject sub-010

dfc-kit score-states /path/to/features.store /path/to/models/k4.model \
  /path/to/results/k4-scores.json --subject sub-010 --subject sub-011
```

The scorer uses only the frozen model parameters. KMeans reports mean squared
distance to the nearest center; Gaussian HMM reports mean log likelihood per
sample. Training subjects are rejected by default during prediction and
scoring. Use `--allow-fit-subjects` only for a labelled training-set
diagnostic.

## Compare candidate state counts

Generate score JSON files for each subject-disjoint fold, candidate `K`, and
prespecified seed. Then compare them:

```bash
dfc-kit compare-state-counts results/k-selection.json \
  --score results/fold-1-k2-seed-17.json \
  --score results/fold-1-k4-seed-17.json \
  --score results/fold-2-k2-seed-17.json \
  --score results/fold-2-k4-seed-17.json
```

The comparison checks the model family, candidate/seed grid, held-out run
identities, and subject-disjoint folds. It reports the best K and the
one-standard-error K. The command does not create a workflow directory; a
final refit can be scripted as a separate call to the model-fitting API.

## Align and summarize repeated fits

```bash
dfc-kit align-states \
  models/k4-seed-17.model models/k4-seed-29.model \
  models/k4-seed-29-aligned.model results/k4-alignment \
  --alignment-metric euclidean

dfc-kit summarize-stability \
  /path/to/features.store models/k4-seed-17.model \
  results/k4-stability.json \
  --candidate-model models/k4-seed-29.model
```

State alignment uses Hungarian matching. Euclidean matching preserves
amplitude differences; Pearson matching compares pattern shape and is useful
for CAP-style states. Stability summaries are calculated only after alignment.

## Fixed-length information

```bash
dfc-kit fixed-information /path/to/xcpd results/fixed-information \
  --atlas Schaefer200 --space MNI152NLin2009cAsym \
  --information-groups groups.json \
  --length 120 --length 180 \
  --draws 20 --sample-seed 20260819
```

The fixed-information command samples only within contiguous retained frame
segments and supports an optional frozen window schedule.

## Inference and summaries

Summarize every named FeatureStore feature within each acquisition. Repeat
`--statistic` to request multiple summaries; with no option, only the mean is
written.

```bash
dfc-kit summarize-store features/ets.store results/ets-summary.json \
  --statistic mean \
  --statistic variance \
  --statistic standard_deviation
```

Available statistics are `mean`, `variance`, `standard_deviation`, `minimum`,
and `maximum`. Variance and standard deviation use all retained samples with
`ddof=0`. Censor-delimited segments remain distinct during feature generation,
then their valid samples are combined by sample count within the acquisition.

Use `summarize-states`, `summarize-information`, `describe-states`,
`infer-state-metrics`, and `infer-paired-endpoints` for reporting and
inference. Every command writes finite JSON and refuses accidental overwrite
of an existing output path. Paired NBS is available through the
`dfckit.inference.paired_nbs` Python API.
