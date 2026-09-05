# Synthetic performance benchmarks

## Fixed-length MI/CMI

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python benchmarks/profile_information.py --jobs 1 2 4 8 --runs 8 --draws 25
```

This benchmark generates 36 synthetic ROI time series with a censor gap and
estimates all 12-by-4 MI/CMI pairs at five window lengths. It reports end-to-end
time, including process startup and array transfer, and requires exact agreement
with serial output for every numeric array and metadata field. Worker count is
an execution setting, so changing it must not change the scientific result.
For short batches, process startup can outweigh the gain from parallel work.

Cloud reference run on 2026-09-05 (Python 3.10.14, NumPy 1.26.4,
SciPy 1.13.0, 8 runs, 25 draws per length, 1,000 total windows):

| Processes | Elapsed time | Speedup over serial | Exact output match |
|---:|---:|---:|---|
| 1 | 184.122 s | 1.00 | Yes |
| 4 | 47.186 s | 3.90 | Yes |
| 16 | 12.207 s | 15.08 | Yes |
| 40 | 5.230 s | 35.20 | Yes |

These timings include worker startup. All arrays and metadata matched the serial
reference; the speedup describes this synthetic workload, not a guaranteed rate
for arbitrary ROI counts, sample lengths, or machine allocations.

## Feature and state methods

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
