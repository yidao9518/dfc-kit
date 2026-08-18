# Command-line workflows

The `dfc-kit` command connects the supported XCP-D input boundary to the
bounded-memory FeatureStore writers. It does not perform fMRIPrep or XCP-D
preprocessing.

## Inspect acquisitions

List matching task/run entities without loading ROI values:

```bash
dfc-kit inspect-xcpd /path/to/xcp_d \
  --atlas Glasser \
  --atlas Tian \
  --space MNI152NLin2009cAsym \
  --task rest
```

The command prints one JSON object containing `n_acquisitions` and an ordered
`acquisitions` array. Each entry includes its derived `acquisition_id`, atlas
set, outlier file, and optional motion file. Use this step to catch ambiguous
spaces or missing atlas counterparts before writing a store.

## Build a FeatureStore

ROI selection is an ordinary JSON object whose arrays define the exact column
order sent to the estimators:

```json
{
  "Glasser": ["Left_V1", "Right_V1"],
  "Tian": ["PUT-DP-lh", "PUT-DP-rh"]
}
```

Build a censor-bounded sliding-window FC store:

```bash
dfc-kit build-store /path/to/xcp_d /path/to/features/window-fc.store \
  --atlas Glasser \
  --atlas Tian \
  --space MNI152NLin2009cAsym \
  --task rest \
  --roi-selection rois.json \
  --method window-fc \
  --window-length 56 \
  --window-step 8 \
  --chunk-size 128 \
  --tr 0.75
```

Build a complete ETS store with the same input contract:

```bash
dfc-kit build-store /path/to/xcp_d /path/to/features/ets.store \
  --atlas Glasser \
  --space MNI152NLin2009cAsym \
  --roi-selection rois.json \
  --method ets \
  --chunk-size 256 \
  --tr 0.75
```

Build a run-standardized MTD store. MTD differences are formed only within
contiguous retained segments, and the output keeps the original derivative
start/end frames:

```bash
dfc-kit build-store /path/to/xcp_d /path/to/features/mtd.store \
  --atlas Glasser \
  --atlas Tian \
  --space MNI152NLin2009cAsym \
  --task rest \
  --roi-selection rois.json \
  --method mtd \
  --chunk-size 256 \
  --tr 0.75
```

The output JSON reports run count, subject IDs, acquisition IDs, sequence and
feature counts, and the FeatureStore format version. Output directories must
be new; an existing store is never silently overwritten.

Build an instantaneous CAP pattern store. CAP uses one feature per selected ROI
and standardizes each ROI independently within every uninterrupted retained
segment before writing rows:

```bash
dfc-kit build-store /path/to/xcp_d /path/to/features/cap.store \
  --atlas Glasser \
  --atlas Tian \
  --space MNI152NLin2009cAsym \
  --task rest \
  --roi-selection rois.json \
  --method cap \
  --chunk-size 256 \
  --tr 0.75
```

The store source contract is
`cap:within-segment-roi-zscore-ddof0`. A censor-bounded segment shorter than
two retained frames is omitted; all other rows retain the original frame index
and segment ID. CAP stores contain ROI patterns, not FC edges.

## Fit a state model

By default, `fit-states --method kmeans` uses `--fitting-mode streaming` with
`--algorithm minibatch`. It fits directly from memory-mapped FeatureStore
chunks, performs bounded-memory standardization and deterministic initialization,
and updates scikit-learn `MiniBatchKMeans` with repeated `partial_fit` passes:

```bash
dfc-kit fit-states /path/to/features/window-fc.store /path/to/models/k4.model \
  --method kmeans \
  --n-states 4 \
  --seed 20260818 \
  --n-init 10 \
  --max-iter 10 \
  --batch-size 4096
```

Streaming KMeans accepts only `--algorithm minibatch`; `--algorithm lloyd` is
rejected in this mode. `--init-sample-size` controls the bounded initialization
sample and is also streaming-only.

To reproduce a historical in-memory fit, select materialized mode. This reads
the selected subjects into a complete `FeatureSequenceDataset` and calls one
scikit-learn `.fit` operation. Lloyd and MiniBatch are distinct estimators:

```bash
# Complete KMeans.fit using Lloyd updates.
dfc-kit fit-states /path/to/features/cap.store /path/to/models/cap-lloyd.model \
  --method kmeans \
  --n-states 5 \
  --seed 20260818 \
  --n-init 20 \
  --max-iter 300 \
  --fitting-mode materialized \
  --algorithm lloyd \
  --no-standardize-features

# Complete MiniBatchKMeans.fit, matching the in-memory CAP wrapper.
dfc-kit fit-states /path/to/features/cap.store /path/to/models/cap-k5.model \
  --method kmeans \
  --n-states 5 \
  --seed 20260818 \
  --n-init 20 \
  --max-iter 300 \
  --fitting-mode materialized \
  --algorithm minibatch \
  --batch-size 4096 \
  --no-standardize-features
```

Materialized mode has no bounded-memory guarantee. The CAP writer has already
performed within-segment ROI z-scoring under the
`cap:within-segment-roi-zscore-ddof0` contract, so CAP reproduction should use
`--no-standardize-features`. Leaving the default
`--standardize-features` enabled performs a second pooled scaling over CAP rows
and therefore defines a different model. For a large store, use streaming mode
instead and treat its `partial_fit` semantics as distinct from historical
`MiniBatchKMeans.fit`.

The `--fitting-mode` and `--algorithm` options apply to KMeans. The HMM path
continues to use its streaming PCA/HMM implementation and explicit
censor-bounded sequence lengths:

Fit a Gaussian HMM after streaming PCA reduction:

```bash
dfc-kit fit-states /path/to/features/window-fc.store /path/to/models/hmm-k5.model \
  --method hmm \
  --n-states 5 \
  --n-pca-components 10 \
  --seed 20260818 \
  --n-init 3 \
  --n-iter 200 \
  --minimum-sequence-length 2
```

`--subject` may be repeated to declare an explicit training cohort. Every
requested subject must exist in the store; misspelled IDs are rejected rather
than silently omitted. If no subjects are listed, all store subjects are used.
The command records fitted subject IDs in the model, and later held-out
prediction APIs reject overlap by default.

The output directory contains a strict JSON manifest and numeric NPZ arrays,
never pickle. It must not already exist. The command prints a JSON summary with
the model kind, output path, fitted subjects, sample/sequence counts, state
count, seed, fitting mode, and method-specific fit diagnostics. KMeans reports inertia;
HMM reports convergence, log likelihood, PCA dimension, and omitted short
sequences. Training assignments are not written implicitly.

KMeans requires the `states` extra; HMM requires the `hmm` extra:

```bash
python -m pip install 'dfc-kit[states]'
python -m pip install 'dfc-kit[hmm]'
```

## Decode held-out subjects

Use a saved model to decode a FeatureStore. Subjects used during fitting are
rejected by default, so a held-out cohort must be named explicitly when the
store also contains training subjects:

```bash
dfc-kit predict-states /path/to/features/window-fc.store \
  /path/to/models/k4.model \
  /path/to/predictions/k4-heldout.labels \
  --subject sub-010 \
  --subject sub-011
```

`--allow-fit-subjects` is an explicit escape hatch for exporting training
assignments. It is never implied by the model or by an omitted subject filter.
The prediction artifact preserves sequence identities, acquisition IDs,
segment boundaries, original sample start/end indices, and (for HMMs) posterior
probabilities and log likelihood. It is also a strict JSON manifest plus
numeric NPZ arrays and cannot overwrite an existing path.

## Summarize state metrics

Convert a prediction artifact into one JSON document of run-level metrics:

```bash
dfc-kit summarize-states \
  /path/to/predictions/k4-heldout.labels \
  /path/to/predictions/k4-heldout.metrics.json
```

The metrics document contains occupancy, mean dwell in samples and seconds,
switch count/rate, transition counts, and row-conditional transition
probabilities for each subject/session/acquisition. Dwell and transitions are
restarted at every stored sequence boundary, so censor gaps and repeated
acquisitions cannot be joined accidentally.

## Score a model on held-out subjects

Evaluate a frozen model without using training inertia or training likelihood:

```bash
dfc-kit score-states \
  /path/to/features/window-fc.store \
  /path/to/models/k4.model \
  /path/to/results/k4-heldout-scores.json \
  --subject sub-010 \
  --subject sub-011
```

KMeans reports sample-mean squared distance to the nearest centroid in the
frozen training-standardized space; lower is better. Gaussian HMM reports
sample-mean log likelihood under the frozen training scaler, PCA, and HMM;
higher is better. HMM likelihood is computed separately for every
censor-bounded sequence and summed within each acquisition, so no transition
crosses a gap.

The strict JSON contains run and pooled scores, the model fingerprint, the
eligible evaluation-data fingerprint, retained sample/sequence counts, and the
HMM short-sequence rule. It also contains the FeatureStore feature-contract
fingerprint and complete fitted-model specification required for cross-fold
comparison. Compare candidate state counts only on an identical held-out cohort
and within one model family. See
[Held-out state-model scoring](state_scoring.md).

## Compare candidate state counts

Run the complete participant-level workflow directly from one FeatureStore:

```bash
dfc-kit cross-validate-state-counts \
  /path/to/features.store \
  /path/to/results/k-cross-validation \
  --method kmeans \
  --n-states 2 \
  --n-states 4 \
  --seed 17 \
  --seed 29 \
  --n-folds 5 \
  --split-seed 20260818 \
  --n-init 20
```

The command assigns whole participants to balanced folds by a stable SHA-256
ranking of the split seed and subject label. It then fits and saves every
fold-by-candidate-by-seed model, scores the frozen models on that fold's held-out
participants, and runs the same `compare-state-counts` contract described
below. KMeans and HMM use the same fitting options as `fit-states`; HMM also
requires `--n-pca-components`.

The output is a new workflow directory containing `models/`, `scores/`,
`comparison.json`, and `workflow.json`. Artifact references are relative to the
workflow root, so the complete directory is portable. The directory appears at
the requested path only after every fit, score, and comparison succeeds.

In nested cross-validation, repeat `--subject` for every participant in the
current outer training set. Participants absent from that explicit development
cohort cannot enter an inner fit or validation fold. Do not include outer test
participants.

### Refit the selected state count and score an outer test set

Once the inner workflow has selected $K$, refit it on the complete development
cohort and evaluate an explicit untouched test cohort:

```bash
dfc-kit evaluate-selected-state-count \
  /path/to/features.store \
  /path/to/results/k-cross-validation \
  /path/to/results/outer-evaluation \
  --selection one-standard-error \
  --test-subject sub-010 \
  --test-subject sub-011
```

The command reloads and validates the complete inner workflow rather than
trusting its summary fields. The current FeatureStore must contain the exact
development data used during selection, and every test subject must be absent
from the development cohort. All fit settings and complete-fit seeds come from
the inner workflow; they cannot be changed on the evaluation command line.

The output directory contains `models/`, `scores/`, and `evaluation.json`.
The latter records the inner-workflow fingerprint, selected rule and $K$,
development/test identities and fingerprints, relative artifact paths, and
scores at seed, participant, and cohort levels. Runs are sample-weighted within
participant and seed; seeds are equal-weighted within participant; participants
are equal-weighted in the final cohort score. The directory is published only
after every seed completes successfully.

### Run the complete nested evaluation

Automate both levels for the complete cohort:

```bash
dfc-kit nested-cross-validate-state-counts \
  /path/to/features.store \
  /path/to/results/nested-k \
  --checkpoint /path/to/results/nested-k.checkpoint \
  --method kmeans \
  --n-states 2 \
  --n-states 4 \
  --seed 17 \
  --seed 29 \
  --outer-n-folds 5 \
  --outer-split-seed 20260818 \
  --inner-n-folds 4 \
  --inner-split-seed 20260819 \
  --selection one-standard-error
```

Each outer development cohort receives a new inner split, candidate comparison,
and selected-$K$ refit. Every participant is scored in one outer test fold and
never enters that fold's inner selection. The final cohort score weights these
participant scores equally, including when outer folds differ in size.

The top-level output contains `nested_evaluation.json` and one portable
`selection/` plus `evaluation/` pair per outer fold. The command reconstructs
all child artifacts before publishing the directory. With `--checkpoint`,
complete fold artifacts survive interruption and are reused only when command,
FeatureStore, split, model, and child-artifact identities still agree. An
interrupted inner grid also reuses each validated model and score cell; a model
whose score is missing is only rescored. Without `--checkpoint`, failure removes
the whole temporary tree. See
[Nested state-count cross-validation](nested_cross_validation.md).

### Inspect nested progress

Audit an interrupted checkpoint or completed nested workflow without modifying
it:

```bash
dfc-kit inspect-nested-state-counts \
  /path/to/features.store \
  /path/to/results/nested-k.checkpoint
```

The JSON output reports each outer fold as `pending-selection`,
`selection-in-progress`, `pending-evaluation`, or `complete`, plus
complete/total inner models and scores, outer models, temporary paths, and an
audited fit-step completion fraction. It also
distinguishes an ordinary checkpoint, a complete checkpoint waiting for
promotion, and a published result. The fraction counts model fits represented
by complete artifacts; it is not a wall-clock estimate. Current FeatureStore
content and every complete child artifact are revalidated before output.
The same JSON also reports `execution_status` as `active`, `idle`, `stale`, or
`invalid`, together with available owner and timing fields. Checkpoint writers
are serialized by an operating-system lock, so a concurrent invocation using
the same checkpoint fails before altering workflow artifacts.

### Compare existing score artifacts

Combine a complete candidate set across subject-disjoint validation folds:

```bash
dfc-kit compare-state-counts /path/to/results/k-selection.json \
  --score /path/to/results/fold-1-k2-seed-17.json \
  --score /path/to/results/fold-1-k2-seed-29.json \
  --score /path/to/results/fold-1-k4-seed-17.json \
  --score /path/to/results/fold-1-k4-seed-29.json \
  --score /path/to/results/fold-2-k2-seed-17.json \
  --score /path/to/results/fold-2-k2-seed-29.json \
  --score /path/to/results/fold-2-k4-seed-17.json \
  --score /path/to/results/fold-2-k4-seed-29.json
```

The command first aggregates acquisitions by retained sample count within each
participant and complete-fit seed, then weights seeds equally within each
participant, participants equally within a fold, and folds equally across the
comparison. It reports the best mean score and the smallest $K$ within one
fold-level standard error of that best score. Seeds do not count as independent
folds when estimating that standard error.

Every fold must contain the same candidate values. Training participants and
validation participants must be disjoint within a fold, and validation
participants must be mutually exclusive across folds. Model family, feature
contract, sampling contract, complete model-seed set, and all model settings
other than the state count must agree. Every fold must partition the same
development participant cohort, and every participant must be evaluated once.
In nested cross-validation this command belongs
inside each outer training set; outer test scores must not enter selection. See
[Cross-fold state-count selection](state_selection.md).

## Align repeated state fits

Align a candidate model to a prespecified reference model and optionally apply
the same permutation to a candidate prediction artifact:

```bash
dfc-kit align-states \
  /path/to/models/reference.model \
  /path/to/models/candidate.model \
  /path/to/models/candidate-aligned.model \
  /path/to/models/candidate.alignment \
  --predictions /path/to/predictions/candidate.labels \
  --predictions-output /path/to/predictions/candidate-aligned.labels
```

KMeans models are matched by original-feature centroid correlation. Gaussian
HMMs are matched by original-feature emission-mean correlation, which remains
valid when two fits learned different PCA bases. Hungarian assignment produces
one complete candidate-to-reference permutation. The relabeled HMM applies it
to start probabilities, both transition axes, emission means, and covariance
axes; the prediction artifact applies the same mapping to labels and posterior
columns.

The alignment artifact stores the mapping, all pairwise correlations, matched
correlations, both seeds, feature identity, source contract, and sample
interval. Prediction format v3 also stores a deterministic fingerprint of the
complete fitted model parameters and training contract. Alignment requires an
exact candidate fingerprint match: an equal seed alone is insufficient because
the training cohort or hyperparameters may differ. Older v1/v2 predictions
remain readable and summarizable but cannot be automatically aligned. Choose
the reference seed before inspecting downstream outcomes.

## Summarize repeated-fit stability

Decode one common cohort with a reference model and every repeated-seed
candidate, align candidates internally, and write metric dispersion:

```bash
dfc-kit summarize-stability \
  /path/to/features/window-fc.store \
  /path/to/models/k4-seed-17.model \
  /path/to/results/k4-stability.json \
  --candidate-model /path/to/models/k4-seed-29.model \
  --candidate-model /path/to/models/k4-seed-41.model \
  --subject sub-010 \
  --subject sub-011
```

The command accepts KMeans or Gaussian HMM models. All fits must share the
training cohort, fit sample count, model family, feature/source/time contract,
state count, exact training-data fingerprint, and non-seed hyperparameters.
Candidate labels are never assumed
to be aligned: matching and relabeling happen before the report is assembled.
The output preserves fingerprints, mappings, all centroid/emission
correlations, matched correlations, and per-run values, means, population
standard deviations, and valid-fit counts. See
[Repeated-fit state stability](state_stability.md).
