import contextlib
import csv
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.cli import main


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class XCPDWindowPatternIntegrationTests(unittest.TestCase):
    def _write_run(self, root: Path, subject: str, session: str, seed: int) -> None:
        func = root / subject / f"ses-{session}" / "func"
        stem = f"{subject}_ses-{session}_task-rest"
        spatial = f"{stem}_space-MNI152NLin2009cAsym"
        rng = np.random.default_rng(seed)
        driver = rng.normal(size=32)
        values = np.column_stack(
            [driver + rng.normal(scale=scale, size=32) for scale in (0.1, 0.2, 0.3, 0.4)]
        )
        write_tsv(
            func / f"{spatial}_atlas-Example_stat-mean_timeseries.tsv",
            ["visual", "motor", "putamen", "thalamus"],
            values.tolist(),
        )
        write_tsv(
            func / f"{spatial}_atlas-Example_stat-coverage_bold.tsv",
            ["Node", "coverage"],
            [[name, 1.0] for name in ("visual", "motor", "putamen", "thalamus")],
        )
        write_tsv(
            func / f"{stem}_outliers.tsv",
            ["outlier"],
            [[0]] * 16 + [[1], [1]] + [[0]] * 16,
        )

    def test_cli_window_pattern_endpoints_feed_paired_inference(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "xcpd"
            for subject_index in (1, 2):
                subject = f"sub-{subject_index:03d}"
                self._write_run(root, subject, "off", subject_index)
                self._write_run(root, subject, "on", subject_index + 10)
            endpoints = Path(temporary) / "patterns.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "window-pattern-endpoints",
                        str(root),
                        str(endpoints),
                        "--atlas",
                        "Example",
                        "--space",
                        "MNI152NLin2009cAsym",
                        "--tr",
                        "0.75",
                        "--window-length",
                        "8",
                        "--window-step",
                        "8",
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(endpoints.read_text(encoding="utf-8"))
            self.assertEqual(payload["n_acquisitions"], 4)
            self.assertEqual(len(payload["rows"]), 12)
            self.assertTrue(all(row["n_windows"] == 4 for row in payload["rows"]))
            self.assertTrue(
                all(row["n_adjacent_pairs"] == 2 for row in payload["rows"])
            )

            inference = Path(temporary) / "inference.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "infer-paired-endpoints",
                        str(endpoints),
                        str(inference),
                        "--condition-a",
                        "on",
                        "--condition-b",
                        "off",
                        "--fdr-family",
                        "whole-edge pattern similarity",
                        "--seed",
                        "17",
                        "--exact",
                    ]
                )
            self.assertEqual(status, 0)
            result = json.loads(inference.read_text(encoding="utf-8"))
            self.assertEqual(result["n_endpoints"], 3)
            self.assertEqual(result["n_tested"], 3)


if __name__ == "__main__":
    unittest.main()
