# LEiDA

Leading Eigenvector Dynamics Analysis (LEiDA) represents every retained frame
by the leading eigenvector of its instantaneous phase-coherence matrix.
`dfckit.connectivity.LEiDA` extracts this representation; existing state-model
utilities can then cluster the vectors without introducing a second state
implementation.

Install the Hilbert-transform dependency with:

```bash
python -m pip install -e '.[phase]'
```

## Segment-wise phase

For every censor-bounded segment meeting `minimum_segment_length`, each ROI is
centered over that segment and transformed with `scipy.signal.hilbert`. Phase
is the angle of the analytic signal. Hilbert transforms are never evaluated
across a removed frame. The default minimum is 20 frames, matching the audited
reference method; shorter segments are omitted because their phase estimates
would be dominated by boundaries.

The transform currently retains every frame of an eligible segment. It does
not trim a second boundary margin after the Hilbert transform. Constant ROIs
are rejected because their analytic phase is undefined as a signal measure.

## Leading vectors

At frame `t`, instantaneous phase coherence is

```text
C_ij(t) = cos(phase_i(t) - phase_j(t))
```

Writing `c_i = cos(phase_i)` and `s_i = sin(phase_i)` gives
`C = c c.T + s s.T`, so the matrix has rank at most two. The implementation
solves the equivalent 2-by-2 eigensystem rather than materializing one ROI-by-
ROI matrix per frame. It returns both the normalized leading vector and its
leading eigenvalue.

Eigenvector sign is arbitrary. Each vector is deterministically multiplied by
`-1` when its element sum is negative, producing the
`positive-vector-sum` orientation used by the reference analysis.

The returned phase array can also support predefined within-network and
between-network coherence summaries without reconstructing every full
ROI-by-ROI matrix. Network membership must be supplied by the analysis; LEiDA
does not infer network blocks.

## State modeling

```python
from dfckit.connectivity import LEiDA
from dfckit.states import fit_kmeans_states, leida_sequences

results = [LEiDA(minimum_segment_length=20).transform(run) for run in training_runs]
sequences = leida_sequences(results)
fit = fit_kmeans_states(
    sequences,
    n_states=5,
    seed=20260818,
    n_init=20,
    algorithm="minibatch",
    standardize_features=False,
)
```

`leida_sequences` creates one `FeatureSequence` per retained segment. The
sample start and end are the same original frame, the sampling interval is one
TR, and dwell/transition calculations restart at every segment boundary.

The audited LEiDA state definition clusters the oriented unit vectors in their
native geometry, so `standardize_features=False` is explicit. As with other
learned state models, fitted participant IDs are recorded and held-out
prediction rejects participant overlap by default.
