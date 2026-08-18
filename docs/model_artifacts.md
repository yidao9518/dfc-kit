# Fitted model artifacts

`dfc-kit` can persist the fitted parameters needed to transform or decode new
participants. The supported model classes are:

- `KMeansStateModel`, including models fitted from a `FeatureStore`;
- `StreamingPCAModel`;
- `GaussianHMMStateModel`, including compact store-fitted models.

## Save and load

```python
from dfckit.io import load_fitted_model, save_fitted_model

artifact = save_fitted_model(
    fit.model,
    "models/window-kmeans.model",
)
restored_model = load_fitted_model(artifact)
held_out = predict_kmeans_store(restored_model, held_out_store)
```

The target is a new directory containing:

```text
window-kmeans.model/
  manifest.json
  arrays.npz
```

`manifest.json` records the model kind, schema version, feature identity,
source contract, sample interval, algorithm settings, fitted participants, and
fit counts. Artifact v2 also records the exact training-data fingerprint,
effective out-of-core KMeans initialization-sample size, and HMM IncrementalPCA
batch size so repeated-fit equality can be verified. `arrays.npz` contains only numeric
arrays. Loading uses
`allow_pickle=False`; object arrays are rejected rather than deserialized.

## Validation and overwrite behavior

Saving refuses to overwrite an existing file, directory, or symbolic link.
Loading accepts fitted-model artifact v1 and v2, then requires the version's
exact metadata field set and array set. Version 1 remains usable for prediction;
its unrecorded training-data and initialization/PCA provenance is represented as unknown.
It also checks model-specific shapes, finite values, positive feature scales, HMM
probability rows, covariance structure, and PCA explained-variance bounds.
Loaded arrays are read-only.

These checks detect malformed or incompatible artifacts; they are not a
cryptographic signature. Copy the complete artifact directory as one unit and
use filesystem or object-store integrity controls when stronger provenance is
required.

`fitted_model_fingerprint` preserves the historical identity of v1 artifacts
when newly introduced provenance is unknown. For v2 out-of-core fits, recorded
training data and fit parameters participate in the fingerprint. The
`summarize-stability` command rejects legacy artifacts whose missing provenance
prevents verification that all repeated fits changed only the seed.

## Leakage contract after loading

The fitted participant IDs are part of the artifact. Restored KMeans and HMM
models therefore continue to reject prediction on a participant used for the
fit unless the caller explicitly overrides that guard. Saving a model does not
turn a training assignment into an independent result.

An aligned KMeans or HMM returned by `relabel_kmeans_model` or
`relabel_gaussian_hmm_model` can be saved through the same artifact API. Its
state-indexed parameters retain the reference numbering after restoration.

Compact store-fitted HMMs preserve `emission_covariances=None` and do not
materialize a dense original-feature covariance tensor during persistence. Use
`reconstruct_emission_covariance(model, state)` when one original-space state
covariance is needed.

Feature stores use their own append-only directory format. Time-series inputs,
state assignments, clinical tables, figures, and inference outputs are not
fitted models and are deliberately outside this artifact API.
