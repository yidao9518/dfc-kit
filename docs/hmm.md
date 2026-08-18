# Gaussian HMM

`dfckit.states.fit_gaussian_hmm_states` fits an optional Gaussian hidden
Markov model to gap-bounded feature sequences. Install its dependencies with:

```bash
python -m pip install -e '.[hmm]'
```

The implementation uses `hmmlearn.hmm.GaussianHMM` and scikit-learn PCA. It is
appropriate for instantaneous ROI signals or another declared
`FeatureSequenceDataset`; clinical labels are never inputs to the state fit.

## Input sequences

Raw XCP-D ROI runs can be converted without joining censor gaps:

```python
from dfckit.states import timeseries_sequences

sequences = timeseries_sequences(dataset, minimum_segment_length=2)
```

Every retained segment becomes one sequence. During fitting and decoding, the
pooled observation matrix is always accompanied by the complete sequence
length vector, so the likelihood, Viterbi path, posterior probabilities,
dwell runs, and transition counts do not bridge a gap.

Sequences shorter than the HMM `minimum_sequence_length` are omitted and the
model records their count. The default is two samples because a singleton
cannot inform a transition.

## Preprocessing and fitting

```python
from dfckit.states import fit_gaussian_hmm_states

fit = fit_gaussian_hmm_states(
    sequences,
    n_states=5,
    seed=20260818,
    n_pca_components=10,
    covariance_type="diag",
    n_init=5,
    n_iter=200,
)
```

Features are centered and divided by their pooled population standard
deviation using training observations only. Constant features are rejected.
If `n_pca_components` is set, PCA is fitted on those standardized training
observations; `None` uses the standardized feature space without reduction.

Each initialization uses seed `seed + initialization_index`. The model with
the greatest training log likelihood is retained. All candidate likelihoods,
the selected initialization, convergence status, iteration count, tolerance,
and package versions are stored.

Supported emission covariance types are `diag` and `full`. The result includes
means and full covariance matrices in both reduced and original feature spaces,
plus starting probabilities and the transition matrix.

For a FeatureStore fit, `emission_covariances` is intentionally `None` rather
than a potentially enormous original-space tensor. The reduced covariance,
scaler, and PCA components retain the same information; call
`dfckit.states.reconstruct_emission_covariance(model, state)` only when one
original-space state covariance is explicitly needed.

## Output and held-out prediction

The fit result contains one `StateLabelSequence` and one posterior-probability
array for every eligible input sequence. Existing state metrics then summarize
occupancy, dwell, switching, and transitions:

```python
from dfckit.states import summarize_state_assignments

metrics = summarize_state_assignments(fit.states.assignments)
```

Held-out decoding uses the frozen training scaler, PCA, and HMM parameters:

```python
from dfckit.states import predict_gaussian_hmm_states

heldout = predict_gaussian_hmm_states(fit.model, heldout_sequences)
```

Prediction rejects any participant present in the HMM fit by default. Feature
identity, construction contract, and sampling interval must also match.

## FeatureStore fitting

For high-dimensional window FC or ETS outputs, the store-specific wrapper keeps
the original feature matrix on disk:

```python
from dfckit.outofcore_hmm import fit_gaussian_hmm_store, predict_gaussian_hmm_store

fit = fit_gaussian_hmm_store(
    training_store,
    n_states=5,
    n_pca_components=10,
    seed=20260818,
    n_init=5,
    pca_batch_size=4096,
    minimum_sequence_length=2,
)
heldout = predict_gaussian_hmm_store(fit.model, test_store)
```

This API requires an explicit PCA dimension. IncrementalPCA learns the pooled
training scaler and components from memory-mapped chunks; only the resulting
low-dimensional observations are concatenated for `hmmlearn`. The source
feature matrix is never reconstructed. Each censor-bounded sequence remains a
separate entry in the `lengths` vector, including during held-out decoding.
Short sequences are omitted and counted in `omitted_short_sequence_count`.

HMM state numbers are arbitrary. Multiple seeds or resamples must be aligned
before state-specific stability or group inference; convergence and high
training likelihood alone do not establish state stability or biological
validity.

## Align repeated fits

```python
from dfckit.states import (
    align_gaussian_hmm_emissions,
    apply_gaussian_hmm_alignment,
    relabel_gaussian_hmm_model,
)

alignment = align_gaussian_hmm_emissions(reference.model, candidate.model)
aligned_training_result = apply_gaussian_hmm_alignment(candidate.states, alignment)
aligned_candidate_model = relabel_gaussian_hmm_model(candidate.model, alignment)
```

Matching uses original-feature emission means, which remain comparable when
the two fits learned different PCA coordinates. The model operation reorders
start probabilities, transition rows and columns, reduced/original emission
parameters, and optional state covariances. The result operation reorders both
decoded labels and posterior columns. See [State alignment](state_alignment.md)
for the mapping convention and repeated-seed workflow.
