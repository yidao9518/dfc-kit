# Repeated-fit state stability

State labels are arbitrary across optimization runs. Stability summaries are
therefore valid only after every candidate fit has been mapped to one declared
reference numbering. `dfc-kit` can perform decoding, alignment, and metric
aggregation as one operation so unaligned labels cannot be stacked by mistake.

## Command-line workflow

Fit the same model specification on the same training participants under
several seeds, then decode one common held-out cohort:

```bash
dfc-kit summarize-stability \
  features/window-fc.store \
  models/k4-seed-17.model \
  results/k4-stability.json \
  --candidate-model models/k4-seed-29.model \
  --candidate-model models/k4-seed-41.model \
  --subject sub-010 \
  --subject sub-011
```

The first model is the reference. Its fitted-model fingerprint, not merely its
seed, defines the state numbering. Each candidate is decoded from the same
`FeatureStore`, matched to the reference with Hungarian assignment, and
relabelled before any occupancy, dwell, switch, or transition metric is
calculated.

The command requires:

- one model family and state count;
- identical feature, source, and sample-interval contracts;
- identical training participants, exact training-data fingerprints, fitted
  sample counts, and model hyperparameters apart from the seed;
- distinct fitted-model fingerprints;
- the same held-out subjects and exact sequence/sample boundaries for every
  decoded fit.

Training participants are rejected by default. `--allow-fit-subjects` is
available for explicitly labelled training-set diagnostics, but it does not
turn them into held-out results. An existing output path is never overwritten.

## Output contract

The strict JSON report records:

- reference and candidate fingerprints and seeds in fit order;
- the exact training-data fingerprint shared by all fits;
- each candidate-to-reference permutation, full correlation matrix, matched
  correlations, and alignment method;
- subject/session/acquisition identity and retained sequence/sample counts;
- per-fit, mean, population-standard-deviation, and valid-fit-count values for
  occupancy, dwell samples, dwell seconds, switch rate, and transition
  probabilities.

Unavailable dwell or transition values are JSON `null`, never non-standard
`NaN`. A standard deviation is also `null` when fewer than two fits provide a
finite value. Otherwise it is the population standard deviation across the
included fits (`ddof=0`). Counts describe the common decoded cohort once; they
are not multiplied by the number of fits.

Low or negative matched correlations remain visible in the report. A complete
Hungarian permutation only establishes one-to-one numbering; it does not by
itself establish that two state solutions are similar.

## Python API

For assignments already aligned by the caller:

```python
from dfckit.states import summarize_state_stability

runs = summarize_state_stability(
    (reference_assignments, candidate_1_aligned, candidate_2_aligned)
)
```

The function verifies exact sequence identity, censor segment, sample start/end
indices, source contract, and sample interval before stacking metrics. Use
`dfckit.io.write_state_stability` to write the same strict JSON schema when
model identities and `StateAlignment` objects are available.

## Cross-validation boundary

Choose and fit the reference inside each training fold. A full-cohort reference
would expose held-out participants to the state patterns used for numbering.
The complete sequence of fit, decode, alignment, and stability summarization
must therefore be repeated independently in every outer fold.

Fitted-model artifact v2 records the training-data fingerprint, effective
KMeans initialization-sample size, and HMM IncrementalPCA batch size. Version 1
artifacts remain readable for prediction. A legacy model lacking this evidence
is not accepted for a claim that repeated fits changed only the seed.
