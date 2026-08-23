import csv
import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit import TimeSeriesDataset, TimeSeriesRun
from dfckit.information import (
    FrozenWindow,
    InformationGroups,
    compute_fixed_information,
    load_fixed_information,
    load_fixed_window_schedule,
    load_information_groups,
    save_fixed_information,
)


class InformationGroupAndScheduleTests(unittest.TestCase):
    def test_group_schema_is_named_strict_and_disjoint(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "groups.json"
            path.write_text(
                json.dumps(
                    {
                        "left": ["left-a", "left-b"],
                        "right": ["right"],
                        "conditioning": ["condition"],
                    }
                ),
                encoding="utf-8",
            )
            groups = load_information_groups(path)
            self.assertEqual(groups.left, ("left-a", "left-b"))
            self.assertEqual(groups.conditioning, ("condition",))

            path.write_text(
                '{"left":["a"],"left":["b"],"right":["c"],"conditioning":null}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate field"):
                load_information_groups(path)

            with self.assertRaisesRegex(ValueError, "disjoint"):
                InformationGroups(("same",), ("same",), None)

    def test_schedule_uses_original_frame_bounds(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "schedule.tsv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(
                    ["acquisition_id", "length", "draw", "start_frame", "end_frame"]
                )
                writer.writerow(["sub-001_task-rest", 5, 0, 11, 15])
            schedule = load_fixed_window_schedule(path)
            self.assertEqual(schedule, (FrozenWindow("sub-001_task-rest", 5, 0, 11, 15),))

            path.write_text(
                "acquisition_id\tlength\tdraw\tstart_frame\nsub-001\t5\t0\t11\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "header"):
                load_fixed_window_schedule(path)


class FixedInformationArtifactTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(804)
        runs = []
        for subject, offset in (("sub-001", 0), ("sub-002", 100)):
            driver = rng.normal(size=80)
            values = np.column_stack(
                (
                    driver + rng.normal(scale=0.2, size=80),
                    driver + rng.normal(scale=0.3, size=80),
                    driver + rng.normal(scale=0.2, size=80),
                    driver + rng.normal(scale=0.3, size=80),
                    driver,
                )
            )
            runs.append(
                TimeSeriesRun(
                    values=values,
                    original_indices=np.r_[np.arange(40), np.arange(45, 85)] + offset,
                    roi_names=("left-a", "left-b", "right-a", "right-b", "condition"),
                    subject=subject,
                    session="baseline",
                    acquisition_id=f"{subject}_ses-baseline_task-rest",
                )
            )
        self.dataset = TimeSeriesDataset(tuple(runs))
        self.groups = InformationGroups(
            ("left-a", "left-b"),
            ("right-a", "right-b"),
            ("condition",),
        )

    def _compute(self):
        return compute_fixed_information(
            self.dataset,
            self.groups,
            lengths=(20, 24),
            draws=3,
            sample_seed=91,
            k=3,
        )

    def test_roundtrip_and_frozen_schedule_replay(self):
        sampled = self._compute()
        schedule = tuple(
            FrozenWindow(
                sampled.acquisitions[int(sampled.acquisition_index[row])].acquisition_id,
                int(sampled.length[row]),
                int(sampled.draw[row]),
                int(sampled.start_frame[row]),
                int(sampled.end_frame[row]),
            )
            for row in range(sampled.n_draws)
        )
        replayed = compute_fixed_information(
            self.dataset,
            self.groups,
            lengths=(20, 24),
            draws=3,
            sample_seed=999,
            schedule=schedule,
            schedule_source="schedule.tsv",
        )
        np.testing.assert_array_equal(replayed.mutual_information, sampled.mutual_information)
        np.testing.assert_array_equal(
            replayed.conditional_mutual_information,
            sampled.conditional_mutual_information,
        )
        np.testing.assert_array_equal(replayed.start_frame, sampled.start_frame)
        self.assertEqual(replayed.schedule_mode, "frozen")

        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "information.artifact"
            save_fixed_information(replayed, output)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"manifest.json", "arrays.npz"},
            )
            loaded = load_fixed_information(output)
            np.testing.assert_array_equal(
                loaded.conditional_mutual_information,
                replayed.conditional_mutual_information,
            )
            self.assertEqual(loaded.groups, self.groups)
            self.assertEqual(loaded.n_cells, 4)
            with self.assertRaises(FileExistsError):
                save_fixed_information(replayed, output)

    def test_parallel_jobs_preserve_exact_results(self):
        sequential = compute_fixed_information(
            self.dataset,
            self.groups,
            lengths=(20, 24),
            draws=3,
            sample_seed=91,
            jobs=1,
        )
        parallel = compute_fixed_information(
            self.dataset,
            self.groups,
            lengths=(20, 24),
            draws=3,
            sample_seed=91,
            jobs=3,
        )
        np.testing.assert_array_equal(parallel.start_frame, sequential.start_frame)
        np.testing.assert_array_equal(parallel.mutual_information, sequential.mutual_information)
        np.testing.assert_array_equal(
            parallel.conditional_mutual_information,
            sequential.conditional_mutual_information,
        )

    def test_ineligible_length_cells_are_omitted_but_each_acquisition_survives(self):
        artifact = compute_fixed_information(
            self.dataset,
            self.groups,
            lengths=(20, 100),
            draws=2,
            sample_seed=91,
        )
        self.assertEqual(
            [(cell.acquisition_index, cell.length) for cell in artifact.cells],
            [(0, 20), (1, 20)],
        )
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "partial-cells"
            save_fixed_information(artifact, output)
            loaded = load_fixed_information(output)
            self.assertEqual(loaded.n_cells, 2)

        short = TimeSeriesRun(
            values=np.random.default_rng(2).normal(size=(10, 5)),
            original_indices=np.arange(10),
            roi_names=self.dataset.roi_names,
            subject="sub-003",
            acquisition_id="sub-003_task-rest",
        )
        with self.assertRaisesRegex(ValueError, "every acquisition"):
            compute_fixed_information(
                TimeSeriesDataset((self.dataset.runs[0], short)),
                self.groups,
                lengths=(20,),
                draws=1,
                sample_seed=91,
            )

    def test_frozen_schedule_can_omit_whole_cells(self):
        sampled = compute_fixed_information(
            self.dataset,
            self.groups,
            lengths=(20,),
            draws=2,
            sample_seed=91,
        )
        schedule = tuple(
            FrozenWindow(
                sampled.acquisitions[int(sampled.acquisition_index[row])].acquisition_id,
                int(sampled.length[row]),
                int(sampled.draw[row]),
                int(sampled.start_frame[row]),
                int(sampled.end_frame[row]),
            )
            for row in range(2)
        )
        artifact = compute_fixed_information(
            self.dataset,
            self.groups,
            lengths=(20,),
            draws=2,
            sample_seed=999,
            schedule=schedule,
        )
        self.assertEqual(artifact.n_cells, 1)
        self.assertEqual(artifact.acquisitions[0].acquisition_id, schedule[0].acquisition_id)

    def test_partial_draw_inside_a_declared_schedule_cell_is_rejected(self):
        sampled = self._compute()
        all_windows = [
            FrozenWindow(
                sampled.acquisitions[int(sampled.acquisition_index[row])].acquisition_id,
                int(sampled.length[row]),
                int(sampled.draw[row]),
                int(sampled.start_frame[row]),
                int(sampled.end_frame[row]),
            )
            for row in range(sampled.n_draws)
        ]
        with self.assertRaisesRegex(ValueError, "must contain draws"):
            compute_fixed_information(
                self.dataset,
                self.groups,
                lengths=(20, 24),
                draws=3,
                sample_seed=91,
                schedule=tuple(all_windows[:-1]),
            )
        with self.assertRaises(TypeError):
            compute_fixed_information(
                self.dataset,
                self.groups,
                lengths=(20, 24),
                draws=3,
                sample_seed=91,
                schedule=({"bad": "schedule"},),  # type: ignore[arg-type]
            )

    def test_loader_rejects_array_tampering(self):
        artifact = self._compute()
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "information.artifact"
            save_fixed_information(artifact, output)
            with np.load(output / "arrays.npz", allow_pickle=False) as archive:
                arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
            arrays["mean_mutual_information"][0] += 1.0
            np.savez(output / "arrays.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "mean MI"):
                load_fixed_information(output)

    def test_mi_only_artifact_omits_conditional_arrays(self):
        groups = InformationGroups(("left-a",), ("right-a",), None)
        artifact = compute_fixed_information(
            self.dataset,
            groups,
            lengths=(20,),
            draws=1,
            sample_seed=4,
        )
        self.assertFalse(artifact.has_cmi)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "mi-only"
            save_fixed_information(artifact, output)
            with np.load(output / "arrays.npz", allow_pickle=False) as archive:
                self.assertNotIn("conditional_mutual_information", archive.files)
            loaded = load_fixed_information(output)
            self.assertIsNone(loaded.mean_conditional_mutual_information)

    def test_unknown_or_noncontiguous_frozen_window_is_rejected(self):
        unknown = (
            FrozenWindow("sub-unknown_ses-baseline_task-rest", 20, 0, 0, 19),
        )
        with self.assertRaisesRegex(ValueError, "unknown acquisition/length"):
            compute_fixed_information(
                self.dataset,
                self.groups,
                lengths=(20,),
                draws=1,
                sample_seed=1,
                schedule=unknown,
            )

        schedule = []
        for run in self.dataset.runs:
            start = 30 if run.subject == "sub-001" else 130
            schedule.append(FrozenWindow(str(run.acquisition_id), 20, 0, start, start + 19))
        with self.assertRaisesRegex(ValueError, "not one contiguous retained interval"):
            compute_fixed_information(
                self.dataset,
                self.groups,
                lengths=(20,),
                draws=1,
                sample_seed=1,
                schedule=tuple(schedule),
            )


class FixedInformationEligibilityShapeTests(unittest.TestCase):
    def test_realistic_cell_counts_are_preserved_in_sampled_and_frozen_artifacts(self):
        rng = np.random.default_rng(1908)
        segment_lengths = [7] * 3 + [8] * 3 + [9] * 2 + [10] * 108
        runs = tuple(
            TimeSeriesRun(
                values=rng.normal(size=(n_frames, 2)),
                original_indices=np.arange(n_frames),
                roi_names=("left", "right"),
                subject=f"sub-{index:03d}",
                acquisition_id=f"sub-{index:03d}_task-rest",
            )
            for index, n_frames in enumerate(segment_lengths)
        )
        dataset = TimeSeriesDataset(runs)
        groups = InformationGroups(("left",), ("right",), None)
        sampled = compute_fixed_information(
            dataset,
            groups,
            lengths=(6, 7, 8, 9, 10),
            draws=1,
            sample_seed=20260819,
            jobs=4,
        )
        expected = {6: 116, 7: 116, 8: 113, 9: 110, 10: 108}
        self.assertEqual(Counter(cell.length for cell in sampled.cells), expected)
        self.assertEqual(sampled.n_cells, 563)
        self.assertEqual(sampled.n_draws, 563)

        schedule = tuple(
            FrozenWindow(
                sampled.acquisitions[int(sampled.acquisition_index[row])].acquisition_id,
                int(sampled.length[row]),
                int(sampled.draw[row]),
                int(sampled.start_frame[row]),
                int(sampled.end_frame[row]),
            )
            for row in range(sampled.n_draws)
        )
        frozen = compute_fixed_information(
            dataset,
            groups,
            lengths=(6, 7, 8, 9, 10),
            draws=1,
            sample_seed=0,
            schedule=schedule,
            jobs=4,
        )
        self.assertEqual(Counter(cell.length for cell in frozen.cells), expected)
        np.testing.assert_array_equal(frozen.mutual_information, sampled.mutual_information)

        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "realistic-shape"
            save_fixed_information(frozen, output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["n_cells"], 563)
            self.assertEqual(len(manifest["cells"]), 563)
            self.assertEqual(load_fixed_information(output).n_cells, 563)


if __name__ == "__main__":
    unittest.main()
