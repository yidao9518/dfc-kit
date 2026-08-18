# Cross-fold state-count selection

`compare-state-counts` combines held-out scores from subject-disjoint
validation folds and selects a candidate number of states, $K$. It supports
KMeans quantization error and Gaussian-HMM log likelihood without treating
training fit as evidence of generalization.

## Validation boundary

State-count selection belongs inside the development data. For each validation
fold:

1. fit every candidate $K$ using only that fold's training participants;
2. score every frozen candidate on exactly the same validation participants;
3. retain every acquisition and censor-bounded sequence under the same feature
   contract; and
4. repeat the complete fit for every prespecified model seed;
5. write one v2 `score-states` artifact per candidate and complete-fit seed.

All sessions and runs from one participant stay on the same side of a fold.
Validation participants must be mutually exclusive across the score files
combined in one comparison.

In nested cross-validation, repeat this procedure independently inside each
outer training set. Choose $K$, refit on that complete outer training set,
and evaluate once on its untouched outer test set. Do not pass final test-fold
scores to `compare-state-counts`.

## Automated workflow

`cross-validate-state-counts` performs the complete inner validation workflow
from one FeatureStore:

```bash
dfc-kit cross-validate-state-counts \
  features.store results/k-cross-validation \
  --method kmeans \
  --n-states 2 \
  --n-states 4 \
  --n-states 6 \
  --seed 17 \
  --seed 29 \
  --n-folds 5 \
  --split-seed 20260818
```

The splitter hashes each `(split_seed, subject)` identity with SHA-256, ranks
the hashes, and partitions that order into balanced groups. Input row order and
the number of sessions or runs per participant cannot change fold membership.
The resulting assignments are recorded in `workflow.json`.

For every fold, candidate, and complete-fit seed, the command saves the fitted
model and its v2 held-out score. It then writes the same v2 comparison described
below. All paths in `workflow.json` and `comparison.json` are relative to the
workflow root. The requested output directory is published atomically only
after the complete grid succeeds.

Use repeated `--subject` options to restrict the development cohort when this
runs inside an outer cross-validation fold. The outer test participants must
not be passed to this command. For a Gaussian HMM, add
`--n-pca-components` and any other prespecified HMM settings.

## Outer-test evaluation

After the inner workflow is complete, apply its decision without reopening the
candidate comparison:

```bash
dfc-kit evaluate-selected-state-count \
  features.store results/k-cross-validation results/outer-evaluation \
  --selection one-standard-error \
  --test-subject sub-010 \
  --test-subject sub-011
```

The command strictly reloads every inner model and score, rebuilds the
comparison, recomputes its participant folds, and verifies the current
FeatureStore development-data fingerprint. It then refits the selected $K$ on
all development participants independently for every model seed recorded by
the inner workflow. The named outer-test participants must exist, be unique,
and be completely disjoint from that development cohort.

`--selection one-standard-error` is the default. Use `--selection best` only
when that decision rule was prespecified. Both rules are read from the inner
comparison; outer-test scores cannot alter $K$.

The new output directory contains one full-development model and one v2 test
score artifact per seed, plus `evaluation.json`. Its relative paths remain
valid if the directory is moved. Publication is atomic: a failed fit, score,
or validation leaves no partial output directory.

Outer-test acquisitions are sample-weighted within participant and seed. Seeds
are then averaged equally within each participant, followed by an
equal-participant cohort mean. Test participants and seeds are not treated as
additional inner-validation folds, and no outer-test uncertainty enters the
one-standard-error decision.

To execute every outer fold under one atomic, auditable command, use
`nested-cross-validate-state-counts`. It constructs a separate inner workflow
inside each outer development cohort, evaluates each participant once, and
aggregates the resulting participant scores without fold-size weighting. See
[Nested state-count cross-validation](nested_cross_validation.md).

## Command-line workflow

The lower-level `compare-state-counts` command accepts already generated score
artifacts. The following compact example compares $K=2,4$ across two validation folds
and two complete-fit seeds, 17 and 29:

```bash
dfc-kit compare-state-counts results/k-selection.json \
  --score results/fold-1-k2-seed-17.json \
  --score results/fold-1-k2-seed-29.json \
  --score results/fold-1-k4-seed-17.json \
  --score results/fold-1-k4-seed-29.json \
  --score results/fold-2-k2-seed-17.json \
  --score results/fold-2-k2-seed-29.json \
  --score results/fold-2-k4-seed-17.json \
  --score results/fold-2-k4-seed-29.json
```

Input order is irrelevant. The command groups files by their evaluation-data
fingerprint and orders candidates by `n_states` and `model_seed`.

`model_seed` identifies a complete model fit. It is distinct from an
estimator's internal `n_init`: several initializations used to produce one
selected fit do not replace repeated complete fits in this comparison.

## Score aggregation

Run-level scores are first combined within each participant. If participant
$i$ contributes runs $r$, the participant score is

$$
s_{ifkm} =
\frac{\sum_r S_{irfkm}}{\sum_r n_{irfm}},
$$

where $S$ is total squared distance for KMeans or total log likelihood for
the HMM, $n$ is the retained sample count, and $m$ indexes complete-fit seeds.
A participant with several or longer runs therefore does not receive extra
weight at the fold level.

Complete-fit seeds are then averaged with equal weight inside each participant:

$$
s_{ifk}=\frac{1}{M}\sum_m s_{ifkm}.
$$

The fold score is the equal-participant mean

$$
s_{fk}=\frac{1}{N_f}\sum_i s_{ifk},
$$

and candidate summaries give every validation fold equal weight:

$$
\bar{s}_k=\frac{1}{F}\sum_f s_{fk},
\qquad
SE_k=\frac{SD(s_{1k},\ldots,s_{Fk})}{\sqrt{F}}.
$$

The SD and SE are computed across subject-disjoint validation folds only.
Seeds are repeated measurements of fitting variability, not independent
validation samples, and therefore do not increase the uncertainty sample size.

Lower KMeans squared distance is better. Higher Gaussian-HMM log likelihood
per sample is better. Scores from the two model families are not comparable.

## Selection rule

The best candidate optimizes the mean fold score; exact score ties go to the
smaller $K$. The one-standard-error rule then chooses the smallest candidate
within one fold-level standard error of the best score.

For KMeans, eligible candidates satisfy

$$
\bar{s}_k \leq \bar{s}_{k^*} + SE_{k^*}.
$$

For the Gaussian HMM, eligible candidates satisfy

$$
\bar{s}_k \geq \bar{s}_{k^*} - SE_{k^*}.
$$

The output records both `best_n_states` and
`one_standard_error_n_states`; the latter is the complexity-favoring choice.

## Comparison contract

The command rejects a comparison unless:

- every validation fold contains the same set of at least two candidate state
  counts;
- every fold and candidate contains the same strictly increasing set of one or
  more non-negative complete-fit seeds;
- candidate models within a fold use the same training data, fit participants,
  validation participants, run boundaries, and short-sequence rule;
- validation participants do not overlap across folds;
- every fold partitions the same complete development participant cohort, and
  every development participant appears in exactly one validation fold;
- model family, exact feature contract, source contract, sample interval, and
  model settings other than the state count agree across every artifact;
- fitted-model fingerprints are unique; and
- every input uses the v2 scoring format with explicit training-data and
  feature-contract fingerprints.

The effective KMeans initialization pool may be capped by the available
training samples or by $K$. Its realized value remains attached to each
candidate, while the shared optimizer settings must match.

## Output

The strict JSON output contains:

- the candidate ranking, fold means, fold SD/SE, and one-standard-error mask;
- participant scores and their fold-level means for every candidate after
  equal-seed aggregation;
- per-seed participant and fold scores, fitted-model fingerprints,
  specifications, and source score paths;
- fitted-model, training-data, evaluation-data, and feature-contract
  fingerprints;
- fit and validation participant identities;
- retained run, sequence, sample, and omitted-sequence counts; and
- the common comparison specification plus each candidate's realized model
  specification and source score path.

The output path must be new. Existing files and symbolic links are never
overwritten.

## Python API

```python
from dfckit.io import (
    compare_state_model_scores,
    load_state_model_scores,
    write_state_count_comparison,
)

reports = [load_state_model_scores(path) for path in score_paths]
comparison = compare_state_model_scores(reports, score_artifacts=score_paths)
write_state_count_comparison(comparison, "results/k-selection.json")
```

For deterministic fold construction in a custom Python workflow:

```python
from dfckit.states import make_subject_validation_folds

folds = make_subject_validation_folds(
    development_subjects,
    n_folds=5,
    seed=20260818,
)
```

`CandidateStateModelScores`, `FoldStateModelScores`, `StateCountSelection`, and
`StateCountComparison` are immutable validated result objects. Seed,
participant, and fold summaries are recomputed from their underlying scores;
derived uncertainty, ranking, candidate order, and selected values are also
checked during construction.
