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


class XCPDStaticNBSIntegrationTests(unittest.TestCase):
    def _write_run(self, root: Path, subject: str, session: str, seed: int) -> None:
        func = root / subject / f"ses-{session}" / "func"
        stem = f"{subject}_ses-{session}_task-rest"
        spatial = f"{stem}_space-MNI152NLin2009cAsym"
        rng = np.random.default_rng(seed)
        first = rng.normal(size=30)
        second = rng.normal(size=30)
        coupling = 0.2 if session == "off" else 0.65
        values = np.column_stack(
            (
                first,
                coupling * first + rng.normal(scale=0.7, size=30),
                second,
            )
        )
        write_tsv(
            func / f"{spatial}_atlas-Example_stat-mean_timeseries.tsv",
            ["visual", "motor", "putamen"],
            values.tolist(),
        )
        write_tsv(
            func / f"{spatial}_atlas-Example_stat-coverage_bold.tsv",
            ["Node", "coverage"],
            [[name, 1.0] for name in ("visual", "motor", "putamen")],
        )
        write_tsv(
            func / f"{stem}_outliers.tsv",
            ["outlier"],
            [[0]] * 14 + [[1], [1]] + [[0]] * 14,
        )

    def test_static_fc_endpoints_feed_paired_nbs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "xcpd"
            for subject_index in range(6):
                subject = f"sub-{subject_index:03d}"
                self._write_run(root, subject, "off", subject_index + 1)
                self._write_run(root, subject, "on", subject_index + 101)
            endpoints = Path(temporary) / "static-fc.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "static-fc-endpoints",
                        str(root),
                        str(endpoints),
                        "--atlas",
                        "Example",
                        "--space",
                        "MNI152NLin2009cAsym",
                        "--tr",
                        "0.75",
                    ]
                )
            self.assertEqual(status, 0)
            payload = json.loads(endpoints.read_text(encoding="utf-8"))
            self.assertEqual(payload["n_acquisitions"], 12)
            self.assertEqual(payload["n_features"], 3)
            self.assertEqual(len(payload["rows"]), 36)
            self.assertTrue(
                all(
                    row["measure"] == "whole_acquisition_fisher_z_fc"
                    and row["n_samples"] == 28
                    for row in payload["rows"]
                )
            )

            inference = Path(temporary) / "nbs.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "infer-paired-nbs",
                        str(endpoints),
                        str(inference),
                        "--condition-a",
                        "on",
                        "--condition-b",
                        "off",
                        "--threshold",
                        "2.0",
                        "--permutations",
                        "100",
                        "--seed",
                        "9",
                    ]
                )
            self.assertEqual(status, 0)
            result = json.loads(inference.read_text(encoding="utf-8"))
            self.assertEqual(result["source_contract"], payload["source_contract"])
            self.assertEqual(result["n_subjects"], 6)
            self.assertEqual(result["n_edges"], 3)


if __name__ == "__main__":
    unittest.main()
