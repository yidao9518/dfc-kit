"""Numerical and process-boundary checks for fixed-window MI/CMI execution."""

import dataclasses
import importlib.util
import multiprocessing
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from dfckit import TimeSeriesDataset, TimeSeriesRun
from dfckit.information import (
    FixedLengthInformation,
    FrozenWindow,
    InformationGroups,
    compute_fixed_information,
)
from dfckit.information import estimators as information_estimators

HAS_INFORMATION_EXTRA = importlib.util.find_spec("scipy") is not None


def _worker_pid_at_barrier(window, left, right, **kwargs):
    """Require two independent workers to reach the same task boundary."""
    left[0].wait(timeout=30)
    return os.getpid()


def _dataset() -> TimeSeriesDataset:
    rng = np.random.default_rng(1069)
    runs = []
    # Keep a deliberately nonlexical input order and two different eligible grids.
    for identifier, segment_length in (("synthetic-z", 36), ("synthetic-a", 28)):
        values = rng.normal(size=(2 * segment_length, 5))
        values[:, 2] += 0.6 * values[:, 0]
        values[:, 3] += 0.4 * values[:, 1]
        runs.append(
            TimeSeriesRun(
                values=values,
                original_indices=np.r_[
                    np.arange(segment_length),
                    np.arange(segment_length + 5, 2 * segment_length + 5),
                ],
                roi_names=("left-a", "left-b", "right-a", "right-b", "condition"),
                subject=identifier,
                session="baseline",
                acquisition_id=f"{identifier}_baseline",
            )
        )
    return TimeSeriesDataset(tuple(runs))


@unittest.skipUnless(HAS_INFORMATION_EXTRA, "requires dfc-kit[information]")
class InformationProcessExecutionTests(unittest.TestCase):
    def setUp(self):
        self.dataset = _dataset()
        self.groups = InformationGroups(
            ("left-a", "left-b"), ("right-a", "right-b"), ("condition",)
        )

    def assert_artifact_arrays_equal(self, observed, expected):
        for field in dataclasses.fields(expected):
            value = getattr(expected, field.name)
            if isinstance(value, np.ndarray):
                with self.subTest(array=field.name):
                    np.testing.assert_array_equal(getattr(observed, field.name), value)
                    self.assertFalse(getattr(observed, field.name).flags.writeable)
        self.assertEqual(observed.acquisitions, expected.acquisitions)
        self.assertEqual(observed.cells, expected.cells)
        self.assertEqual(observed.implementation, expected.implementation)

    def test_single_draw_cells_preserve_acquisition_and_requested_length_order(self):
        parameters = {"lengths": (32, 16, 24), "draws": 1, "sample_seed": 92}
        serial = compute_fixed_information(self.dataset, self.groups, jobs=1, **parameters)
        with patch.object(
            information_estimators,
            "ProcessPoolExecutor",
            wraps=information_estimators.ProcessPoolExecutor,
        ) as pool:
            parallel = compute_fixed_information(self.dataset, self.groups, jobs=3, **parameters)
        self.assertEqual(pool.call_count, 1)
        self.assertEqual(pool.call_args.kwargs["mp_context"].get_start_method(), "spawn")
        self.assert_artifact_arrays_equal(parallel, serial)
        self.assertEqual(parallel.acquisition_index.tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(parallel.length.tolist(), [32, 16, 24, 16, 24])
        self.assertEqual(parallel.draw.tolist(), [0] * 5)

    def test_sampled_cmi_replays_exactly_from_shuffled_frozen_schedule(self):
        serial = compute_fixed_information(
            self.dataset, self.groups, lengths=(24, 16), draws=4, sample_seed=63
        )
        schedule = tuple(
            FrozenWindow(
                acquisition_id=serial.acquisitions[int(serial.acquisition_index[row])].acquisition_id,
                length=int(serial.length[row]),
                draw=int(serial.draw[row]),
                start_frame=int(serial.start_frame[row]),
                end_frame=int(serial.end_frame[row]),
            )
            for row in reversed(range(serial.n_draws))
        )
        parallel = compute_fixed_information(
            self.dataset,
            self.groups,
            lengths=(24, 16),
            draws=4,
            sample_seed=999,
            schedule=schedule,
            jobs=2,
        )
        self.assert_artifact_arrays_equal(parallel, serial)
        self.assertEqual(parallel.schedule_mode, "frozen")

    def test_mi_only_parallel_execution_keeps_conditional_outputs_absent(self):
        groups = InformationGroups(self.groups.left, self.groups.right, None)
        parameters = {"lengths": (24,), "draws": 3, "sample_seed": 17, "standardize": False}
        serial = compute_fixed_information(self.dataset, groups, jobs=1, **parameters)
        parallel = compute_fixed_information(self.dataset, groups, jobs=2, **parameters)
        self.assert_artifact_arrays_equal(parallel, serial)
        self.assertIsNone(parallel.conditional_mutual_information)
        self.assertIsNone(parallel.mean_conditional_mutual_information)
        self.assertIsNone(parallel.conditioning_indices)

    def test_single_run_transform_preserves_seeded_jitter_and_generator_groups(self):
        parameters = {
            "length": 24, "draws": 5, "sample_seed": 38, "jitter": 1e-3, "jitter_seed": 9
        }
        run = self.dataset.runs[0]
        serial = FixedLengthInformation(jobs=1, **parameters).transform(
            run, (0, 1), (2, 3), conditioning=(4,)
        )
        parallel = FixedLengthInformation(jobs=2, **parameters).transform(
            run, iter((0, 1)), iter((2, 3)), conditioning=iter((4,))
        )
        for name in (
            "mutual_information",
            "conditional_mutual_information",
            "mean_mutual_information",
            "mean_conditional_mutual_information",
            "left_indices",
            "right_indices",
            "conditioning_indices",
        ):
            np.testing.assert_array_equal(getattr(parallel, name), getattr(serial, name))
            self.assertFalse(getattr(parallel, name).flags.writeable)
        np.testing.assert_array_equal(parallel.samples.start_frames, serial.samples.start_frames)

    def test_worker_value_error_propagates_and_next_call_still_succeeds(self):
        original = self.dataset.runs[0]
        values = original.values.copy()
        values[:, 2] = 1.0
        invalid = dataclasses.replace(original, values=values)
        estimator = FixedLengthInformation(length=24, draws=3, sample_seed=18, jobs=2)
        with self.assertRaisesRegex(ValueError, "constant"):
            estimator.transform(invalid, (0, 1), (2, 3), conditioning=(4,))
        result = estimator.transform(original, (0, 1), (2, 3), conditioning=(4,))
        self.assertTrue(np.isfinite(result.conditional_mutual_information).all())

    def test_parallel_api_runs_in_fresh_spawn_main(self):
        # Importable worker functions must work without inherited fork-only state.
        program = """
import multiprocessing
import numpy as np
from dfckit import TimeSeriesDataset, TimeSeriesRun
from dfckit.information import InformationGroups, compute_fixed_information

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    rng = np.random.default_rng(141)
    run = TimeSeriesRun(
        values=rng.normal(size=(32, 3)), original_indices=np.arange(32),
        roi_names=('left', 'right', 'condition'), subject='synthetic',
        acquisition_id='synthetic_baseline')
    dataset = TimeSeriesDataset((run,))
    groups = InformationGroups(('left',), ('right',), ('condition',))
    kwargs = dict(lengths=(16, 24), draws=2, sample_seed=27)
    serial = compute_fixed_information(dataset, groups, jobs=1, **kwargs)
    parallel = compute_fixed_information(dataset, groups, jobs=2, **kwargs)
    np.testing.assert_array_equal(serial.mutual_information, parallel.mutual_information)
    np.testing.assert_array_equal(
        serial.conditional_mutual_information, parallel.conditional_mutual_information)
    print('spawn-exact')
"""
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "spawn-exact")

    def test_window_iterator_prefetch_is_bounded_and_close_cancels_pending_work(self):
        produced = 0
        submitted = 0
        completed = 0
        peak_pending = 0
        cancelled = []
        window = self.dataset.runs[0].values[:16]

        def windows():
            nonlocal produced
            for _ in range(1000):
                produced += 1
                yield window

        class ImmediateFuture:
            def __init__(self, function, value):
                self.function = function
                self.value = value

            def result(self):
                nonlocal completed
                completed += 1
                return self.function(self.value)

            def cancel(self):
                cancelled.append(self)
                return True

        class ImmediatePool:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, function, value):
                nonlocal submitted, peak_pending
                submitted += 1
                peak_pending = max(peak_pending, submitted - completed)
                return ImmediateFuture(function, value)

        with patch.object(information_estimators, "ProcessPoolExecutor", ImmediatePool):
            results = information_estimators._estimate_information_blocks(
                windows(), (0, 1), (2, 3), conditioning=(4,), jobs=2
            )
            next(results)
            self.assertLessEqual(produced, 5)
            next(results)
            self.assertLessEqual(produced, 6)
            results.close()
        self.assertEqual(peak_pending, 4)
        self.assertEqual(len(cancelled), submitted - completed)

    def test_two_window_tasks_reach_independent_processes_concurrently(self):
        # The barrier checks concurrency directly, without a timing speedup gate.
        # A manager proxy can cross a spawn queue without inherited global state.
        with multiprocessing.get_context("spawn").Manager() as manager:
            barrier = manager.Barrier(2)
            with patch.object(
                information_estimators, "block_information", _worker_pid_at_barrier
            ):
                pids = tuple(
                    information_estimators._estimate_information_blocks(
                        (self.dataset.runs[0].values[:16],) * 2,
                        (barrier,),
                        (),
                        jobs=2,
                    )
                )
        self.assertEqual(len(set(pids)), 2)
        self.assertNotIn(os.getpid(), pids)


if __name__ == "__main__":
    unittest.main()
