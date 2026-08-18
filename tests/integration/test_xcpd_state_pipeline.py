import csv
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.connectivity import ETS, MTD, SlidingWindowFC
from dfckit.io import load_xcpd_run
from dfckit.states import (
    fit_kmeans_states,
    summarize_state_assignments,
    window_fc_sequences,
)

HAS_STATES_EXTRA = importlib.util.find_spec("sklearn") is not None


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def write_xcpd_run(root: Path, subject: str, shift: float) -> None:
    func = root / subject / "ses-off" / "func"
    base = f"{subject}_ses-off_task-rest"
    spatial = f"{base}_space-MNI152NLin2009cAsym"
    original = np.asarray([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13], dtype=float)
    values = np.column_stack(
        [
            np.sin(original * 0.43 + shift),
            np.cos(original * 0.31 - shift),
            np.sin(original * 0.27 + 0.6 + 2.0 * shift),
        ]
    )
    write_tsv(
        func / f"{spatial}_atlas-Example_stat-mean_timeseries.tsv",
        ["visual", "motor", "putamen"],
        values.tolist(),
    )
    write_tsv(
        func / f"{spatial}_atlas-Example_stat-coverage_bold.tsv",
        ["Node", "coverage"],
        [["visual", 1.0], ["motor", 1.0], ["putamen", 1.0]],
    )
    write_tsv(
        func / f"{base}_outliers.tsv",
        ["outlier"],
        [[0], [0], [0], [0], [0], [0], [1], [1], [0], [0], [0], [0], [0], [0]],
    )


@unittest.skipUnless(HAS_STATES_EXTRA, "requires dfc-kit[states]")
class XCPDStatePipelineIntegrationTests(unittest.TestCase):
    def test_xcpd_to_gap_safe_state_metrics(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_xcpd_run(root, "sub-001", shift=0.0)
            write_xcpd_run(root, "sub-002", shift=0.2)
            runs = [
                load_xcpd_run(
                    root,
                    subject=subject,
                    session="off",
                    atlases="Example",
                    tr=0.8,
                ).run
                for subject in ("sub-001", "sub-002")
            ]
            self.assertEqual([MTD().transform(run).features.shape[0] for run in runs], [10, 10])
            self.assertEqual([ETS().rss(run).rss.shape[0] for run in runs], [12, 12])

            windows = [SlidingWindowFC(length=4, step=2).transform(run) for run in runs]
            sequences = window_fc_sequences(windows)
            fit = fit_kmeans_states(sequences, n_states=2, seed=21, n_init=10)
            metrics = summarize_state_assignments(fit.assignments)

            self.assertEqual(fit.model.fit_subjects, ("sub-001", "sub-002"))
            self.assertEqual(len(sequences.sequences), 4)
            self.assertEqual(len(metrics), 2)
            for run_metrics in metrics:
                self.assertEqual(run_metrics.n_sequences, 2)
                self.assertEqual(run_metrics.n_possible_transitions, 2)
                np.testing.assert_allclose(run_metrics.occupancy.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
