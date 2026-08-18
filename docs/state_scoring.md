# Held-out state-model scoring

Training inertia and training likelihood reward fit to observations already
seen by the estimator. They are not evidence that a selected state count will
generalize. `dfc-kit` therefore scores a frozen KMeans or Gaussian HMM on an
explicit held-out participant cohort.

## Command-line workflow

```bash
dfc-kit score-states \
  features/window-fc.store \
  models/k4.model \
  results/k4-heldout-scores.json \
  --subject sub-010 \
  --subject sub-011
```

Fit and score every candidate state count inside each subject-disjoint
validation split. The validation participant list and FeatureStore must be
identical across candidate values of `n_states`. Participants used to fit the
model are rejected by default. `--allow-fit-subjects` exists only for explicitly
labelled training-set diagnostics and such artifacts cannot enter formal
state-count comparison.

The report records the exact fitted-model fingerprint and a content
fingerprint of the eligible evaluation sequences. Before comparing two score
files, require equal evaluation-data fingerprints, model families, feature
contracts, and preprocessing specifications. Different validation folds produce
different evaluation fingerprints and must be aggregated as fold-level scores,
not compared as though they were the same observations.

## KMeans score

Each held-out row is transformed with the scaler fitted on the training
participants. Its score is the squared Euclidean distance to the nearest frozen
centroid in that standardized feature space:

$$
d_t^2 = \min_k \lVert z_t - c_k \rVert_2^2.
$$

For each acquisition the report contains `total_squared_distance` and
`mean_squared_distance`. The latter is

$$
\frac{1}{T}\sum_{t=1}^{T} d_t^2,
$$

so it divides by samples, not by samples times features. Lower is better when
the held-out cohort is identical. It is a standardized quantization error, not
a likelihood and not a direct measure of biological state validity.

## Gaussian HMM score

The frozen training scaler and PCA basis transform every eligible held-out
sequence. Each censor-bounded sequence is scored separately:

$$
\ell_r = \sum_{s \in r} \log p(Z_s \mid \hat\theta).
$$

Separate calls restart from the fitted initial-state distribution at every
censor boundary. No HMM transition is introduced between retained segments.
The report includes total `log_likelihood` and
`log_likelihood_per_sample`; higher is better on the identical held-out cohort.

Sequences shorter than the model's fitted `minimum_sequence_length` are
excluded under the same rule used during fitting. The report records both this
threshold and `omitted_short_sequence_count`. If any requested participant has
no eligible sequence, scoring fails instead of silently dropping that person.

## Choosing a state count

Held-out score answers only one part of state-count selection. A defensible
choice also checks repeated-seed stability, state occupancy and interpretability,
and whether conclusions repeat across validation folds. Use
`compare-state-counts` to enforce identical candidate sets, exact feature and
model contracts, a complete fold-by-candidate-by-seed grid, run/sample
aggregation within participants, equal-seed and equal-participant aggregation,
and the one-standard-error rule with uncertainty estimated only across folds. See
[Cross-fold state-count selection](state_selection.md).

Do not choose `n_states` from the final test cohort and then report that cohort
as an untouched generalization result. In nested cross-validation, run the
state-count selection folds only inside each outer training set.

KMeans and Gaussian-HMM scores are on different mathematical scales and cannot
rank one model family against the other. Compare KMeans state counts with the
KMeans metric, and HMM state counts with the HMM metric.

## Python API

```python
from dfckit.outofcore import score_kmeans_store
from dfckit.outofcore_hmm import score_gaussian_hmm_store

kmeans_runs = score_kmeans_store(kmeans_model, heldout_store)
hmm_runs = score_gaussian_hmm_store(hmm_model, heldout_store)
```

Both functions return immutable run-level records grouped by
`(subject, session, acquisition_id)`. `dfckit.io.write_state_model_scores`
writes the same strict JSON contract used by the CLI.
