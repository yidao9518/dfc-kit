# Fitted model artifacts

Fitted KMeans, Gaussian HMM, and streaming PCA models are stored as a JSON
manifest plus NumPy arrays. The pickle-free format preserves model parameters,
feature keys, source contract, fit subjects, and numerical settings needed to
decode new data. Loading validates array names, shapes, finite values, and
model-specific parameters.

```python
from dfckit.artifacts import load_fitted_model, save_fitted_model

path = save_fitted_model(fit.model, "models/state-model")
model = load_fitted_model(path)
```

Existing output paths are not overwritten.
