# Synthetic performance benchmarks

Run each method in a separate process so `ru_maxrss` has one clear scope:

```bash
PYTHONPATH=src python benchmarks/profile_core.py \
  --method window-fc --frames 600 --rois 100 --length 60 --step 5

PYTHONPATH=src python benchmarks/profile_core.py \
  --method ets-full --frames 1000 --rois 200

PYTHONPATH=src python benchmarks/profile_core.py \
  --method ets-rss --frames 1000 --rois 200

PYTHONPATH=src python benchmarks/profile_core.py \
  --method ets-store --frames 1000 --rois 200

PYTHONPATH=src python benchmarks/profile_core.py \
  --method window-kmeans-memory --frames 1200 --rois 200 \
  --length 60 --step 5 --states 4 --n-init 3 --max-iter 3 --batch-size 64

PYTHONPATH=src python benchmarks/profile_core.py \
  --method window-kmeans-store --frames 1200 --rois 200 \
  --length 60 --step 5 --states 4 --n-init 3 --max-iter 3 --batch-size 64

PYTHONPATH=src python benchmarks/profile_core.py \
  --method window-hmm-store --frames 1200 --rois 200 \
  --length 60 --step 5 --states 4 --n-init 1 --max-iter 20 \
  --batch-size 64 --pca-components 10
```

The input is deterministic synthetic Gaussian data with a small shared latent
component and one explicit censor gap. The script reports elapsed wall time,
current RSS before/after the transform on Linux, process-lifetime peak RSS,
transform-scoped `tracemalloc` peak, and the size of arrays retained in the
result object. It is a developer benchmark, not a scientific result or a CI
pass/fail threshold.

`ETS().rss()` is expected to use substantially less memory than materializing
the complete frame-by-edge ETS matrix. High-dimensional sliding-window FC and
full ETS are the main targets for chunked feature output.

The two `window-kmeans-*` cases precompute identical sliding-window FC before
starting the measured region. They therefore compare only state fitting: the
in-memory path pools a `FeatureSequenceDataset`, while the store path estimates
the scaler, updates MiniBatchKMeans, scores all initializations, and emits labels
by traversing memory-mapped chunks. Feature construction time is intentionally
excluded from both measurements.

Reference run (`1200` frames, `200` ROIs, `60`-frame windows, step `5`, `4`
states, `3` initializations, `3` passes, batch size `64`) produced:

| State-fitting path | Peak RSS | Traced peak | Time |
|---|---:|---:|---:|
| In-memory `FeatureSequenceDataset` | 497.28 MiB | 279.89 MiB | 8.11 s |
| Memory-mapped `FeatureStore` | 294.03 MiB | 135.57 MiB | 8.44 s |

These are synthetic engineering measurements, not scientific results. The
stored path's files occupied `34.02 MiB`; its advantage is bounded memory, not
an assumption that disk I/O is free.

The `window-hmm-store` command above (`1` initialization, `20` HMM iterations,
and `10` PCA components) produced `218` windows with `19,900` original FC
features. The measured fit took `1.94 s`, reached `292.15 MiB` process-lifetime
peak RSS and `127.22 MiB` transform-scoped traced peak, and retained `2.60 MiB`
of result arrays. The memory-mapped feature files occupied `34.02 MiB`. The
compact fitted model does not retain a dense
`states x original_features x original_features` covariance tensor; an
original-space covariance is reconstructed explicitly for one requested state.
