# Command-line workflows

The command line is a thin wrapper around the Python API. It covers XCP-D
discovery, FeatureStore construction, whole-acquisition static FC,
whole-edge window-pattern and low-rank covariance endpoints, information
estimates, state fitting,
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
  --method kmeans --n-states 4 --seed 17 \
  --max-iter 100 --streaming-tol 1e-4 \
  --streaming-patience 3 --streaming-min-passes 2

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

For streaming KMeans, `--max-iter` is the maximum number of complete
FeatureStore passes per initialization. The command stops earlier when relative
center drift remains within `--streaming-tol` for the requested consecutive
passes. Its JSON output reports `converged`, `passes_completed`, and
`initialization_passes`. Set `--streaming-tol 0` to force every initialization
to run exactly `--max-iter` passes. These options do not alter materialized
KMeans fitting.

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

## Low-rank covariance endpoints

```bash
dfc-kit lowrank-endpoints /path/to/xcpd results/lowrank.json \
  --atlas Schaefer200 --space MNI152NLin2009cAsym \
  --roi-selection rois.json --tr 0.75 \
  --window-length 96 --window-step 48 \
  --rank 1 --rank 2 --rank 4
```

The command standardizes each ROI inside each censor-bounded window and writes
acquisition-level means for effective rank, eigenvalue concentration,
cross-half reconstruction, split-subspace similarity, adjacent-window
subspace similarity, all-pair subspace similarity, and adjacency excess. Its
JSON can be passed directly to
`infer-paired-endpoints`.

## Whole-edge window-pattern endpoints

```bash
dfc-kit window-pattern-endpoints /path/to/xcpd results/patterns-100tr.json \
  --atlas Glasser --atlas Tian \
  --space MNI152NLin2009cAsym \
  --roi-selection rois.json --tr 0.75 \
  --window-length 100 --window-step 100 --taper hamming
```

This command retains every Fisher-z edge in each window and writes three
acquisition-level endpoints: order-insensitive all-pair pattern similarity,
observed adjacent-window similarity, and their difference (adjacency excess).
All comparisons remain inside censor-bounded retained-frame segments. Run the
command separately for each window length when defining a length-sensitivity
family.

## Whole-acquisition static FC and paired NBS

```bash
dfc-kit static-fc-endpoints /path/to/xcpd results/static-fc.json \
  --atlas Glasser --atlas Tian \
  --space MNI152NLin2009cAsym \
  --roi-selection rois.json --tr 0.75

dfc-kit infer-paired-nbs results/static-fc.json results/static-nbs.json \
  --condition-a on --condition-b off \
  --threshold 2.0 --threshold 2.5 --threshold 3.0 \
  --permutations 10000 --seed 20260725 \
  --alternative two-sided \
  --component-statistic edge_extent \
  --component-sign-mode separate
```

`static-fc-endpoints` correlates all retained frames within each acquisition
once and writes the complete upper-triangle Fisher-z edge vector. It does not
average sliding windows. `infer-paired-nbs` can also consume an edge
FeatureStore summary, so a sliding-window edge statistic remains available as
a separate analysis with a different scientific definition.

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
`infer-state-metrics`, `infer-paired-endpoints`, and `infer-paired-nbs` for
reporting and inference. Every command writes finite JSON and refuses
accidental overwrite of an existing output path. Paired NBS is also available
through the `dfckit.inference.paired_nbs` Python API.

When one endpoint artifact contains more than one scientific comparison
family, repeat `--endpoint` to state exactly which endpoints enter the current
paired inference and BH correction:

```bash
dfc-kit infer-paired-endpoints lowrank.json results/reconstruction.json \
  --condition-a on --condition-b off \
  --fdr-family "low-rank reconstruction across declared ranks" \
  --endpoint heldout_r2.rank_4.mean \
  --endpoint heldout_r2.rank_5.mean \
  --endpoint heldout_r2.rank_6.mean \
  --permutations 10000 --bootstrap 10000 --seed 20260905
```

With no `--endpoint`, the command retains its all-endpoint behavior. Empty,
duplicate, and unknown explicit names are rejected. The output artifact records
the requested and selected names so the FDR family can be audited directly.

Paired inference rejects more than one acquisition per participant and
condition by default. For studies with repeated runs, opt in to an equal-run
mean before the participant-level contrast:

```bash
dfc-kit infer-state-metrics state-metrics.json results/state-inference \
  --condition-a postop --condition-b preop \
  --metric occupancy --fdr-family "state occupancy" --seed 20260831 \
  --within-condition-aggregation mean
```

Each acquisition is summarized independently. The option only averages those
acquisition-level endpoints within participant and condition; it never joins
windows, dwell sequences, or transitions across run boundaries. Acquisition
IDs must remain unique, so duplicate rows are still rejected.

For two cohorts whose participants are independent, use the separate group
endpoint command:

```bash
dfc-kit infer-independent-endpoints \
  pd-off-endpoints.json hc-endpoints.json results/pd-off-vs-hc.json \
  --group-a PD_OFF --group-b HC \
  --fdr-family "independent endpoint comparison" \
  --covariates group-covariates.tsv \
  --covariate age --covariate sex
```

The reported group coefficient is `group A - group B` after adjustment for the
declared covariates. See [Independent group inference](group_inference.md) for
input rows, covariate scaling, repeated-acquisition handling, and effect-field
semantics.
