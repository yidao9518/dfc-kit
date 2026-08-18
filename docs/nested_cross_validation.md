# Nested state-count cross-validation

`nested-cross-validate-state-counts` estimates held-out state-model performance
while selecting the number of states independently inside every outer training
cohort. It is the complete executable composition of
`cross-validate-state-counts` and `evaluate-selected-state-count`.

## Statistical boundary

For outer fold $o$, the workflow:

1. keeps all acquisitions from each participant on one side of the outer split;
2. runs a complete inner fold-by-$K$-by-seed comparison using only the outer
   development participants;
3. chooses $K_o$ by the prespecified best or one-standard-error rule;
4. refits $K_o$ on the complete outer development cohort for every model seed;
5. scores only that fold's untouched outer-test participants; and
6. repeats the process until every cohort participant has been tested once.

The selected state count may differ across outer folds. This is expected: the
selection procedure, rather than one globally fixed post hoc decision, is being
evaluated.

## Command

```bash
dfc-kit nested-cross-validate-state-counts \
  /path/to/features.store \
  /path/to/results/nested-k \
  --checkpoint /path/to/results/nested-k.checkpoint \
  --method kmeans \
  --n-states 2 \
  --n-states 4 \
  --n-states 6 \
  --seed 17 \
  --seed 29 \
  --outer-n-folds 5 \
  --outer-split-seed 20260818 \
  --inner-n-folds 4 \
  --inner-split-seed 20260819 \
  --selection one-standard-error \
  --n-init 20
```

Use `--method hmm` with `--n-pca-components` for Gaussian HMM selection. All
other KMeans and HMM fitting options match `cross-validate-state-counts`.
Repeated `--subject` options restrict the complete cohort before either split
is created.

`--checkpoint` is optional. It must name a new or existing real directory next
to the final output directory. Without it, any failure removes the complete
temporary workflow. With it, a failure preserves every complete inner
model or score cell, complete inner selection, and outer evaluation so the same
command can resume instead of repeating valid work. If fitting completed but
held-out scoring did not, the fitted cell is reused and only its score is
recomputed.

Only one process may execute against a given checkpoint at a time. A sibling
file named `.<checkpoint-name>.dfckit.lock` is locked by the operating system
before the workflow reads or writes the checkpoint. A second writer exits
without changing the checkpoint or final output. The persistent record stores
the owner's PID, hostname, process-start token, and acquisition time. Normal
exit adds a release time; after a hard exit the kernel releases ownership while
the unreleased record remains available for diagnosis and safe reclamation.

## Aggregation

For model seed $m$, acquisition-level totals are first divided by their retained
sample counts after pooling acquisitions within participant $i$. Complete-fit
seeds are then averaged equally:

$$
s_i=\frac{1}{M}\sum_m
\frac{\sum_r S_{irm}}{\sum_r n_{irm}}.
$$

Every participant appears in exactly one outer test fold. The final nested
score is therefore

$$
\bar{s}_{\mathrm{nested}}=\frac{1}{N}\sum_i s_i.
$$

It is not the unweighted mean of outer-fold means. Participant weighting remains
equal when balanced splitting produces folds of different sizes.

Lower KMeans squared distance is better. Higher Gaussian-HMM log likelihood per
sample is better. Scores from the two model families are not comparable.

## Output and audit

The output directory is portable and contains:

```text
nested-k/
  nested_evaluation.json
  outer-folds/
    fold-001/
      selection/
      evaluation/
    fold-002/
      selection/
      evaluation/
    ...
```

Each `selection/` directory contains the complete inner models, score artifacts,
comparison, and workflow manifest. Each `evaluation/` directory contains the
full-development refits and outer-test scores. `nested_evaluation.json` records
the deterministic outer split, child fingerprints, selected $K$ per fold,
participant scores, complete inner/outer model counts, and the
participant-balanced cohort score.

```python
from dfckit.io import load_nested_state_count_cross_validation

nested = load_nested_state_count_cross_validation("results/nested-k")
print(nested.selected_state_counts)
print(nested.cohort_score)
```

The loader reopens every inner model, inner score, outer refit, and outer score;
rebuilds both child summaries; verifies both split levels and participant
boundaries; and then recomputes the final participant-level aggregation. It
rejects incomplete grids, altered summaries, model or score drift, duplicate
JSON fields, non-finite values, and paths that leave the workflow directory.

The top-level directory is published only after every outer fold passes this
audit. In checkpoint mode, the final manifest is written atomically and the
checkpoint directory is renamed to the requested output; `checkpoint.json` is
removed, so there is only one final result tree.

## Resume contract

A checkpoint is reusable only by the same nested workflow. Before any reuse,
the command verifies:

- method, candidate state counts, complete-fit seeds, selection rule, split
  counts and seeds, and every non-$K$ fit setting;
- source, sampling, feature-contract, complete-cohort data, per-inner-fold data,
  and outer-test data fingerprints from the current FeatureStore;
- complete inner model/score/comparison grids and complete outer refit/score
  bundles through their strict loaders; and
- canonical fold paths and a symlink-free checkpoint tree.

A changed option, participant cohort, feature value, sequence boundary, model
artifact, score artifact, or manifest is rejected rather than silently reused.
Directories left by an interrupted child write use the command's temporary
naming convention and are removed before that child is retried. Each available
inner model is validated before reuse; each score is additionally bound to that
exact model and held-out fold. A complete child workflow must pass its full
reconstruction audit or the resume fails. A checkpoint that already contains a
complete valid `nested_evaluation.json` can be promoted without any model refit.

## Inspect progress

Inspect a checkpoint or a completed result against the current FeatureStore:

```bash
dfc-kit inspect-nested-state-counts \
  /path/to/features.store \
  /path/to/results/nested-k.checkpoint
```

The command is read-only. It does not clean temporary directories, resume
fitting, or alter a manifest. For every outer fold it reports one of:

- `pending-selection`: no complete inner workflow exists;
- `selection-in-progress`: at least one validated inner model or score cell
  exists, but the complete inner workflow is not yet finalized;
- `pending-evaluation`: the inner workflow is complete but the selected-$K$
  outer evaluation is not;
- `complete`: both child artifacts pass strict reconstruction.

The workflow status is `in-progress`, `ready-for-promotion` when the final
manifest already exists inside a checkpoint, or `complete` for a published
result. Temporary paths are reported separately and receive no completion
credit.

`execution_status` independently reports checkpoint ownership:

- `active`: a process currently holds the operating-system lock;
- `idle`: no process holds it and no unreleased owner record remains;
- `stale`: no process holds it, but the last owner did not record release; or
- `invalid`: the persistent lock record is malformed.

The JSON payload includes the lock path, owner PID and hostname, process-start
token, and acquisition/release timestamps when available. Inspection is
read-only: it briefly probes kernel ownership but never rewrites or reclaims the
record. Resuming the workflow reclaims a stale record only after acquiring the
kernel lock.

The fit-step denominator is the planned model count

$$
OICM + OM.
$$

`fit_completion_fraction` is the fraction of those model fits represented by
validated fitted models, including a cell whose score is still pending. Separate
`completed_inner_scores` and `total_inner_scores` fields expose scoring progress.
These are not elapsed-time estimates: candidates, folds, seeds, KMeans and HMM
fits can have different run times. The inspector also rechecks the current
FeatureStore, split seeds, fit settings, child models, scores, summaries, and
symlink boundary before reporting progress.

## Compute cost

With $O$ outer folds, $I$ inner folds, $C$ candidate state counts, and $M$
complete-fit seeds, the workflow performs

$$
OICM + OM
$$

model fits. The first term is inner selection; the second is full-development
refitting. Choose candidate grids, seeds, and fold counts before looking at the
outer-test results.
