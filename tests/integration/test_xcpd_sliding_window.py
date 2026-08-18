import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.connectivity import SlidingWindowFC
from dfckit.io import load_xcpd_run


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


class XCPDSlidingWindowIntegrationTests(unittest.TestCase):
    def test_windows_follow_gaps_recovered_from_xcpd(self):
        with TemporaryDirectory() as temporary:
            func = Path(temporary) / "sub-001" / "ses-off" / "func"
            base = "sub-001_ses-off_task-rest"
            spatial = f"{base}_space-MNI152NLin2009cAsym"
            rng = np.random.default_rng(17)
            retained_values = rng.normal(size=(12, 3))
            write_tsv(
                func / f"{spatial}_atlas-Example_stat-mean_timeseries.tsv",
                ["A", "B", "C"],
                retained_values.tolist(),
            )
            write_tsv(
                func / f"{spatial}_atlas-Example_stat-coverage_bold.tsv",
                ["Node", "coverage"],
                [["A", 1.0], ["B", 1.0], ["C", 1.0]],
            )
            write_tsv(
                func / f"{base}_outliers.tsv",
                ["outlier"],
                [[0], [0], [0], [0], [0], [0], [1], [1], [0], [0], [0], [0], [0], [0]],
            )

            loaded = load_xcpd_run(
                temporary,
                subject="001",
                session="off",
                atlases="Example",
                tr=0.75,
            )
            result = SlidingWindowFC(length=4, step=2).transform(loaded.run)

            np.testing.assert_array_equal(result.start_frames, [0, 2, 8, 10])
            np.testing.assert_array_equal(result.end_frames, [3, 5, 11, 13])
            np.testing.assert_array_equal(result.segment_ids, [0, 0, 1, 1])
            self.assertEqual(result.features.shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
