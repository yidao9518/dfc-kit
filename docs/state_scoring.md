# Held-out state scoring

`score-states` evaluates a frozen KMeans or Gaussian HMM on acquisitions that
were not used for fitting. KMeans reports squared distance to the nearest
center; HMM reports log likelihood per sample. The scorer keeps acquisitions
and censor-bounded sequences separate and rejects overlap with the fit cohort
unless explicitly requested with `--allow-fit-subjects`.

```bash
dfc-kit score-states \
  /path/to/features.store \
  /path/to/models/k4.model \
  /path/to/results/k4-heldout.json \
  --subject sub-010 \
  --subject sub-011
```

The JSON contains the model family, seed, state count, fit subjects, run-level
scores, and the common model settings. It can be passed directly to
`dfckit.states.compare_state_model_scores` together with scores from other
folds and candidate state counts.
