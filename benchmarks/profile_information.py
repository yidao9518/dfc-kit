"""Compare serial and parallel MI/CMI on synthetic ROI36-sized input.

Run from the project root, for example::

    python benchmarks/profile_information.py --jobs 1 2 4 8

The timings include process startup and data transfer. Every parallel result is
checked against a serial reference, including window indices and all pair values.
This script never loads participant data or writes analysis artifacts.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import time

import numpy as np

from dfckit import TimeSeriesDataset, TimeSeriesRun, __version__
from dfckit.information import (
    FixedInformationArtifact,
    InformationGroups,
    compute_fixed_information,
    knn_cmi,
)


def synthetic_dataset(runs: int, frames: int, seed: int) -> TimeSeriesDataset:
    rng = np.random.default_rng(seed)
    roi_names = tuple(f"roi-{index:02d}" for index in range(36))
    acquisitions = []
    for index in range(runs):
        values = rng.normal(size=(frames, 36))
        values[:, 12:16] += 0.4 * values[:, :4]
        split = frames // 2
        original_indices = np.r_[np.arange(split), np.arange(split + 5, frames + 5)]
        acquisitions.append(
            TimeSeriesRun(
                values=values,
                original_indices=original_indices,
                roi_names=roi_names,
                subject=f"synthetic-{index:03d}",
                session="benchmark",
                acquisition_id=f"synthetic-{index:03d}_benchmark",
            )
        )
    return TimeSeriesDataset(tuple(acquisitions))


def assert_exact(actual: FixedInformationArtifact, expected: FixedInformationArtifact) -> None:
    for field in dataclasses.fields(expected):
        value = getattr(expected, field.name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(actual, field.name), value, err_msg=field.name)
        else:
            if getattr(actual, field.name) != value:
                raise AssertionError(f"artifact metadata differs: {field.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--frames", type=int, default=800)
    parser.add_argument("--lengths", type=int, nargs="+", default=[120, 180, 240, 300, 360])
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if min(args.jobs) < 1 or args.runs < 1 or args.frames < 12 or args.draws < 1:
        parser.error("jobs, runs, and draws must be positive; frames must be at least 12")
    if not args.lengths or min(args.lengths) <= 5 or max(args.lengths) > args.frames // 2:
        parser.error("every length must be between 6 and the retained half-run length")

    dataset = synthetic_dataset(args.runs, args.frames, args.seed)
    names = dataset.roi_names
    groups = InformationGroups(names[:12], names[12:16], names[16:26])
    # Import SciPy before timing the reference, while retaining process startup
    # and child imports in the end-to-end parallel timings.
    knn_cmi(*(dataset.runs[0].values[:16, index] for index in range(3)))
    job_counts = tuple(dict.fromkeys((1, *args.jobs)))
    reference = None
    reference_seconds = None
    timings = []
    for jobs in job_counts:
        start = time.perf_counter()
        result = compute_fixed_information(
            dataset,
            groups,
            lengths=args.lengths,
            draws=args.draws,
            sample_seed=args.seed,
            jobs=jobs,
        )
        elapsed = time.perf_counter() - start
        if reference is None:
            reference = result
            reference_seconds = elapsed
        assert_exact(result, reference)
        timings.append(
            {
                "jobs": jobs,
                "elapsed_seconds": elapsed,
                "speedup_over_serial": reference_seconds / elapsed,
                "all_arrays_and_metadata_exact": True,
                "analyzed_windows": result.n_draws,
            }
        )
        print(json.dumps(timings[-1]), flush=True)
    print(
        json.dumps(
            {
                "dfc_kit_version": __version__,
                "python_version": platform.python_version(),
                "input": vars(args),
                "left_rois": 12,
                "right_rois": 4,
                "conditioning_rois": 10,
                "timing_includes_process_startup": True,
                "timings": timings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
