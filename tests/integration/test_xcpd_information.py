import contextlib
import csv
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.cli import main
from dfckit.io import load_fixed_information


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class XCPDInformationIntegrationTests(unittest.TestCase):
    def _write_run(self, root: Path, subject: str, session: str, seed: int) -> None:
        func = root / subject / f"ses-{session}" / "func"
        stem = f"{subject}_ses-{session}_task-rest_run-1"
        spatial = f"{stem}_space-MNI152NLin2009cAsym"
        rng = np.random.default_rng(seed)
        driver = rng.normal(size=36)
        values = np.column_stack(
            (
                driver + rng.normal(scale=0.2, size=36),
                driver + rng.normal(scale=0.3, size=36),
                driver,
            )
        )
        write_tsv(
            func / f"{spatial}_atlas-Example_stat-mean_timeseries.tsv",
            ["left", "right", "condition"],
            values.tolist(),
        )
        write_tsv(
            func / f"{spatial}_atlas-Example_stat-coverage_bold.tsv",
            ["Node", "coverage"],
            [["left", 1.0], ["right", 1.0], ["condition", 1.0]],
        )
        write_tsv(
            func / f"{stem}_outliers.tsv",
            ["outlier"],
            [[0]] * 18 + [[1], [1]] + [[0]] * 18,
        )

    def test_cli_multi_subject_session_and_frozen_replay(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "xcpd"
            self._write_run(root, "sub-001", "first", 1)
            self._write_run(root, "sub-001", "second", 2)
            self._write_run(root, "sub-002", "first", 3)
            groups = Path(temporary) / "groups.json"
            groups.write_text(
                json.dumps(
                    {"left": ["left"], "right": ["right"], "conditioning": ["condition"]}
                ),
                encoding="utf-8",
            )
            output = Path(temporary) / "sampled"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "fixed-information",
                        str(root),
                        str(output),
                        "--atlas",
                        "Example",
                        "--subject",
                        "001",
                        "--subject",
                        "002",
                        "--space",
                        "MNI152NLin2009cAsym",
                        "--information-groups",
                        str(groups),
                        "--length",
                        "10",
                        "--length",
                        "12",
                        "--draws",
                        "2",
                        "--sample-seed",
                        "73",
                        "--jobs",
                        "2",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["n_runs"], 3)
            self.assertEqual(summary["n_cells"], 6)
            self.assertEqual(summary["n_draws"], 12)
            self.assertEqual(summary["subjects"], ["sub-001", "sub-002"])
            self.assertEqual(summary["jobs"], 2)
            sampled = load_fixed_information(output)

            schedule = Path(temporary) / "schedule.tsv"
            with (output / "draw_metrics.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as source, schedule.open("w", encoding="utf-8", newline="") as target:
                reader = csv.DictReader(source, delimiter="\t")
                fields = ["acquisition_id", "length", "draw", "start_frame", "end_frame"]
                writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                for row in reader:
                    writer.writerow({field: row[field] for field in fields})

            replay_output = Path(temporary) / "replayed"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "fixed-information",
                        str(root),
                        str(replay_output),
                        "--atlas",
                        "Example",
                        "--subject",
                        "001",
                        "--subject",
                        "002",
                        "--space",
                        "MNI152NLin2009cAsym",
                        "--information-groups",
                        str(groups),
                        "--length",
                        "10",
                        "--length",
                        "12",
                        "--draws",
                        "2",
                        "--sample-seed",
                        "999",
                        "--window-schedule",
                        str(schedule),
                    ]
                )
            self.assertEqual(status, 0)
            replayed = load_fixed_information(replay_output)
            np.testing.assert_array_equal(replayed.mutual_information, sampled.mutual_information)
            np.testing.assert_array_equal(
                replayed.conditional_mutual_information,
                sampled.conditional_mutual_information,
            )
            self.assertEqual(replayed.schedule_mode, "frozen")


if __name__ == "__main__":
    unittest.main()
