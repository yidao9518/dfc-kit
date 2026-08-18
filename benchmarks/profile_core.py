#!/usr/bin/env python3
"""Profile one synthetic dfc-kit transform in an isolated process."""

from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import sys
import time
import tracemalloc
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit import TimeSeriesRun
from dfckit.connectivity import ETS, LowRankCovariance, SlidingWindowFC
from dfckit.outofcore import fit_minibatch_kmeans_store
from dfckit.outofcore_hmm import fit_gaussian_hmm_store
from dfckit.states import fit_kmeans_states, window_fc_sequences
from dfckit.storage import write_ets_store, write_window_fc_store


def peak_rss_mib() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return float(raw / divisor)


def current_rss_mib() -> float | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="ascii").splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return None


def array_bytes(value: object) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, tuple):
        return sum(array_bytes(item) for item in value)
    if dataclasses.is_dataclass(value):
        return sum(array_bytes(getattr(value, field.name)) for field in dataclasses.fields(value))
    return 0


def result_bytes(result: object) -> int:
    return sum(array_bytes(value) for value in vars(result).values())


def synthetic_run(frames: int, rois: int, seed: int) -> TimeSeriesRun:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(frames, min(8, rois)))
    loadings = rng.normal(scale=0.3, size=(latent.shape[1], rois))
    values = latent @ loadings + rng.normal(size=(frames, rois))
    split = frames // 2
    original = np.r_[np.arange(split), np.arange(split + 5, frames + 5)]
    return TimeSeriesRun(
        values=values,
        original_indices=original,
        roi_names=tuple(f"roi-{index:04d}" for index in range(rois)),
        subject="synthetic-001",
        session="benchmark",
        tr=0.8,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "window-fc",
            "window-fc-store",
            "ets-full",
            "ets-store",
            "ets-rss",
            "lowrank",
            "window-kmeans-memory",
            "window-kmeans-store",
            "window-hmm-store",
        ),
        required=True,
    )
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--rois", type=int, default=100)
    parser.add_argument("--length", type=int, default=60)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 8 or args.rois < 2:
        raise ValueError("frames must be at least 8 and rois at least 2")
    run = synthetic_run(args.frames, args.rois, args.seed)
    temporary: TemporaryDirectory[str] | None = None
    prepared = None
    if args.method == "window-kmeans-memory":
        prepared = window_fc_sequences(
            (SlidingWindowFC(args.length, args.step).transform(run),)
        )
    elif args.method in {"window-kmeans-store", "window-hmm-store"}:
        temporary = TemporaryDirectory()
        prepared = write_window_fc_store(
            Path(temporary.name) / "features",
            (run,),
            SlidingWindowFC(args.length, args.step),
            chunk_size=32,
        )
    baseline_rss = peak_rss_mib()
    current_rss_before = current_rss_mib()
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    storage_mib = 0.0
    if args.method == "window-fc":
        result = SlidingWindowFC(args.length, args.step).transform(run)
        n_samples = len(result.features)
        n_features = result.features.shape[1]
    elif args.method == "window-fc-store":
        temporary = TemporaryDirectory()
        result = write_window_fc_store(
            Path(temporary.name) / "features",
            (run,),
            SlidingWindowFC(args.length, args.step),
            chunk_size=32,
        )
        n_samples = result.n_samples
        n_features = result.n_features
    elif args.method == "ets-full":
        result = ETS().transform(run)
        n_samples = len(result.features)
        n_features = result.features.shape[1]
    elif args.method == "ets-store":
        temporary = TemporaryDirectory()
        result = write_ets_store(
            Path(temporary.name) / "features",
            (run,),
            chunk_size=32,
        )
        n_samples = result.n_samples
        n_features = result.n_features
    elif args.method == "ets-rss":
        result = ETS().rss(run)
        n_samples = len(result.rss)
        n_features = 1
    elif args.method == "lowrank":
        result = LowRankCovariance(
            args.length,
            args.step,
            ranks=(1, 2, 4),
        ).transform(run)
        n_samples = len(result.effective_rank)
        n_features = args.rois * sum(result.ranks)
    elif args.method == "window-kmeans-memory":
        assert prepared is not None
        result = fit_kmeans_states(
            prepared,
            n_states=args.states,
            seed=args.seed,
            n_init=args.n_init,
            max_iter=args.max_iter,
            algorithm="minibatch",
            batch_size=args.batch_size,
        )
        n_samples = prepared.n_samples
        n_features = len(prepared.feature_keys)
    elif args.method == "window-kmeans-store":
        assert prepared is not None
        result = fit_minibatch_kmeans_store(
            prepared,
            n_states=args.states,
            seed=args.seed,
            n_init=args.n_init,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
        )
        n_samples = prepared.n_samples
        n_features = prepared.n_features
    elif args.method == "window-hmm-store":
        assert prepared is not None
        result = fit_gaussian_hmm_store(
            prepared,
            n_states=args.states,
            seed=args.seed,
            n_pca_components=args.pca_components,
            n_init=args.n_init,
            n_iter=args.max_iter,
            pca_batch_size=args.batch_size,
        )
        n_samples = prepared.n_samples
        n_features = prepared.n_features
    else:
        raise RuntimeError(f"unhandled benchmark method: {args.method}")
    elapsed = time.perf_counter() - started
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if temporary is not None:
        storage_mib = sum(
            path.stat().st_size for path in Path(temporary.name).rglob("*") if path.is_file()
        ) / 1024.0**2
    payload = {
        "method": args.method,
        "frames": args.frames,
        "rois": args.rois,
        "length": args.length,
        "step": args.step,
        "seed": args.seed,
        "n_states": args.states,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "batch_size": args.batch_size,
        "pca_components": args.pca_components,
        "n_output_samples": n_samples,
        "n_output_features": n_features,
        "input_mib": run.values.nbytes / 1024.0**2,
        "materialized_result_mib": result_bytes(result) / 1024.0**2,
        "stored_files_mib": storage_mib,
        "baseline_rss_mib": baseline_rss,
        "current_rss_before_mib": current_rss_before,
        "current_rss_after_mib": current_rss_mib(),
        "peak_rss_mib": peak_rss_mib(),
        "transform_traced_current_mib": traced_current / 1024.0**2,
        "transform_traced_peak_mib": traced_peak / 1024.0**2,
        "elapsed_seconds": elapsed,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
