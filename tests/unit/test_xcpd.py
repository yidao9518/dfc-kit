import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.io import discover_xcpd_files, discover_xcpd_runs, load_xcpd_dataset, load_xcpd_run


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


class XCPDInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.func = self.root / "sub-001" / "ses-off" / "func"
        self.func.mkdir(parents=True)
        self.base = "sub-001_ses-off_task-rest"
        self.spatial = f"{self.base}_space-MNI152NLin2009cAsym"
        self.outliers = self.func / f"{self.base}_outliers.tsv"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_atlas(self, atlas: str, names: list[str], values: np.ndarray) -> None:
        write_tsv(
            self.func / f"{self.spatial}_atlas-{atlas}_stat-mean_timeseries.tsv",
            names,
            values.tolist(),
        )
        write_tsv(
            self.func / f"{self.spatial}_atlas-{atlas}_stat-coverage_bold.tsv",
            ["Node", "coverage"],
            [[name, 0.95] for name in names],
        )

    def test_discovers_standard_xcpd_structure(self):
        self.write_atlas("Glasser", ["V1", "M1"], np.ones((4, 2)))
        write_tsv(self.outliers, ["outlier"], [[0], [0], [0], [0]])
        files = discover_xcpd_files(
            self.root, subject="001", session="off", atlases="Glasser", space="MNI152NLin2009cAsym"
        )
        self.assertEqual(files.atlases[0].atlas, "Glasser")
        self.assertEqual(files.outliers, self.outliers)
        self.assertEqual(files.subject, "sub-001")
        self.assertEqual(files.session, "off")
        self.assertEqual(files.acquisition_id, self.base)

    def test_full_length_tables_are_censored_and_multi_atlas_order_is_explicit(self):
        glasser = np.arange(10, dtype=float).reshape(5, 2)
        tian = np.arange(10, 20, dtype=float).reshape(5, 2)
        self.write_atlas("Glasser", ["M1", "V1"], glasser)
        self.write_atlas("Tian", ["PUT", "THA"], tian)
        write_tsv(self.outliers, ["outlier"], [[0], [1], [0], [0], [1]])
        result = load_xcpd_run(
            self.root,
            subject="sub-001",
            session="ses-off",
            atlases=("Glasser", "Tian"),
            space="MNI152NLin2009cAsym",
            roi_names={"Glasser": ("V1", "M1"), "Tian": ("THA", "PUT")},
            minimum_coverage=0.5,
            tr=0.75,
        )
        self.assertEqual(result.run.roi_names, ("V1", "M1", "THA", "PUT"))
        self.assertEqual(result.run.acquisition_id, "sub-001_ses-off_task-rest")
        np.testing.assert_array_equal(result.run.original_indices, [0, 2, 3])
        np.testing.assert_array_equal(result.run.values[:, :2], glasser[[0, 2, 3]][:, [1, 0]])
        np.testing.assert_array_equal(result.run.values[:, 2:], tian[[0, 2, 3]][:, [1, 0]])
        self.assertEqual(result.source_axes, ("full-length", "full-length"))

    def test_already_censored_table_recovers_original_indices(self):
        values = np.arange(6, dtype=float).reshape(3, 2)
        self.write_atlas("Glasser", ["V1", "M1"], values)
        write_tsv(self.outliers, ["outlier"], [[False], [True], [False], [False]])
        result = load_xcpd_run(
            self.root, subject="001", session="off", atlases="Glasser", tr=0.8
        )
        np.testing.assert_array_equal(result.run.values, values)
        np.testing.assert_array_equal(result.run.original_indices, [0, 2, 3])
        self.assertEqual(result.source_axes, ("censored",))

    def test_batch_loader_keeps_same_session_runs_as_distinct_acquisitions(self):
        for run_number in (1, 2):
            base = f"sub-001_ses-off_task-rest_run-{run_number}"
            spatial = f"{base}_space-MNI152NLin2009cAsym"
            values = np.arange(8, dtype=float).reshape(4, 2) + run_number
            write_tsv(
                self.func / f"{spatial}_atlas-Glasser_stat-mean_timeseries.tsv",
                ["V1", "M1"],
                values.tolist(),
            )
            write_tsv(
                self.func / f"{spatial}_atlas-Glasser_stat-coverage_bold.tsv",
                ["Node", "coverage"],
                [["V1", 1.0], ["M1", 1.0]],
            )
            write_tsv(
                self.func / f"{base}_outliers.tsv",
                ["outlier"],
                [[0], [0], [0], [0]],
            )

        files = discover_xcpd_runs(
            self.root,
            subject="001",
            session="off",
            task="rest",
            atlases="Glasser",
            space="MNI152NLin2009cAsym",
        )
        dataset = load_xcpd_dataset(
            self.root,
            subject="001",
            session="off",
            task="rest",
            atlases="Glasser",
            space="MNI152NLin2009cAsym",
            tr=0.8,
        )
        self.assertEqual(len(files), 2)
        self.assertEqual(
            [run.acquisition_id for run in dataset.runs],
            [
                "sub-001_ses-off_task-rest_run-1",
                "sub-001_ses-off_task-rest_run-2",
            ],
        )

    def test_discovery_enters_explicit_session_directory_symlinks(self):
        self.write_atlas("Glasser", ["V1", "M1"], np.ones((4, 2)))
        write_tsv(self.outliers, ["outlier"], [[0], [0], [0], [0]])
        view = self.root / "view"
        (view / "sub-001").mkdir(parents=True)
        (view / "sub-001" / "ses-off").symlink_to(
            self.root / "sub-001" / "ses-off",
            target_is_directory=True,
        )

        files = discover_xcpd_runs(
            view,
            subject="001",
            session="off",
            task="rest",
            atlases="Glasser",
            space="MNI152NLin2009cAsym",
        )
        single = discover_xcpd_files(
            view,
            subject="001",
            session="off",
            task="rest",
            atlases="Glasser",
            space="MNI152NLin2009cAsym",
        )
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].atlases[0].timeseries, single.atlases[0].timeseries)

    def test_single_without_session_keeps_subject_level_scope(self):
        self.write_atlas("Glasser", ["V1", "M1"], np.ones((4, 2)))
        write_tsv(self.outliers, ["outlier"], [[0], [0], [0], [0]])

        with self.assertRaisesRegex(FileNotFoundError, "expected exactly one"):
            discover_xcpd_files(self.root, subject="001", atlases="Glasser")
        self.assertEqual(
            len(discover_xcpd_runs(self.root, subject="001", atlases="Glasser")),
            1,
        )

    def test_time_axis_mismatch_is_rejected(self):
        self.write_atlas("Glasser", ["V1", "M1"], np.ones((3, 2)))
        write_tsv(self.outliers, ["outlier"], [[0], [1], [0], [0], [0]])
        with self.assertRaisesRegex(ValueError, "time-series/outlier mismatch"):
            load_xcpd_run(self.root, subject="001", session="off", atlases="Glasser")

    def test_invalid_outlier_value_is_rejected(self):
        self.write_atlas("Glasser", ["V1", "M1"], np.ones((3, 2)))
        write_tsv(self.outliers, ["outlier"], [[0], [2], [0]])
        with self.assertRaisesRegex(ValueError, "invalid binary outlier"):
            load_xcpd_run(self.root, subject="001", session="off", atlases="Glasser")

    def test_selected_roi_below_coverage_is_rejected(self):
        self.write_atlas("Glasser", ["V1", "M1"], np.arange(8, dtype=float).reshape(4, 2))
        coverage = self.func / f"{self.spatial}_atlas-Glasser_stat-coverage_bold.tsv"
        write_tsv(coverage, ["Node", "coverage"], [["V1", 0.49], ["M1", 0.95]])
        write_tsv(self.outliers, ["outlier"], [[0], [0], [0], [0]])
        with self.assertRaisesRegex(ValueError, "coverage below 0.5"):
            load_xcpd_run(
                self.root,
                subject="001",
                session="off",
                atlases="Glasser",
                roi_names={"Glasser": ("V1", "M1")},
                minimum_coverage=0.5,
            )

    def test_multiple_spaces_require_an_explicit_space(self):
        self.write_atlas("Glasser", ["V1", "M1"], np.ones((4, 2)))
        second = "sub-001_ses-off_task-rest_space-fsLR"
        write_tsv(
            self.func / f"{second}_atlas-Glasser_stat-mean_timeseries.tsv",
            ["V1", "M1"],
            np.ones((4, 2)).tolist(),
        )
        write_tsv(
            self.func / f"{second}_atlas-Glasser_stat-coverage_bold.tsv",
            ["Node", "coverage"],
            [["V1", 1.0], ["M1", 1.0]],
        )
        write_tsv(self.outliers, ["outlier"], [[0], [0], [0], [0]])
        with self.assertRaisesRegex(FileNotFoundError, "specify space"):
            discover_xcpd_files(
                self.root, subject="001", session="off", atlases="Glasser"
            )
        with self.assertRaisesRegex(ValueError, "duplicate XCP-D acquisition"):
            discover_xcpd_runs(
                self.root, subject="001", session="off", atlases="Glasser"
            )


if __name__ == "__main__":
    unittest.main()
