import unittest

import numpy as np

from dfckit import TimeSeriesDataset, TimeSeriesRun, validate_subject_disjoint


class TimeSeriesRunTests(unittest.TestCase):
    def test_segments_follow_original_frame_gaps(self):
        run = TimeSeriesRun(
            values=np.arange(16, dtype=float).reshape(8, 2),
            original_indices=np.array([0, 1, 2, 5, 6, 10, 11, 12]),
            roi_names=("left", "right"),
        )
        segments = run.segments()
        self.assertEqual([segment.tolist() for segment in segments], [[0, 1, 2], [3, 4], [5, 6, 7]])

    def test_windows_never_cross_a_gap(self):
        run = TimeSeriesRun(
            values=np.arange(20, dtype=float).reshape(10, 2),
            original_indices=np.array([0, 1, 2, 3, 7, 8, 9, 10, 11, 12]),
            roi_names=("left", "right"),
        )
        windows = run.windows(length=3, step=2)
        self.assertEqual(
            [(window.segment_id, window.start_frame, window.end_frame) for window in windows],
            [(0, 0, 2), (1, 7, 9), (1, 9, 11)],
        )
        for window in windows:
            self.assertTrue(np.all(np.diff(window.original_indices) == 1))

    def test_rejects_duplicate_roi_names(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            TimeSeriesRun(np.ones((4, 2)), np.arange(4), ("same", "same"))

    def test_rejects_nonincreasing_original_indices(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            TimeSeriesRun(np.ones((4, 2)), np.array([0, 1, 1, 2]), ("a", "b"))


def make_run(
    subject: str | None,
    session: str | None,
    *,
    acquisition_id: str | None = None,
    roi_names: tuple[str, ...] = ("visual", "motor"),
    tr: float | None = 0.8,
) -> TimeSeriesRun:
    return TimeSeriesRun(
        values=np.arange(12, dtype=float).reshape(6, 2),
        original_indices=np.arange(6),
        roi_names=roi_names,
        subject=subject,
        session=session,
        acquisition_id=acquisition_id,
        tr=tr,
    )


class TimeSeriesDatasetTests(unittest.TestCase):
    def test_groups_multiple_sessions_by_subject(self):
        off = make_run("sub-001", "off")
        on = make_run("sub-001", "on")
        control = make_run("sub-002", "baseline")

        dataset = TimeSeriesDataset([off, on, control])

        self.assertEqual(dataset.n_runs, 3)
        self.assertEqual(dataset.subjects, ("sub-001", "sub-002"))
        self.assertEqual(dataset.runs_by_subject()["sub-001"], (off, on))
        self.assertEqual(dataset.roi_names, ("visual", "motor"))
        self.assertEqual(dataset.tr, 0.8)

    def test_rejects_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            TimeSeriesDataset([])

    def test_rejects_roi_order_mismatch(self):
        with self.assertRaisesRegex(ValueError, "ROI identity or order"):
            TimeSeriesDataset(
                [
                    make_run("sub-001", "off"),
                    make_run("sub-002", "off", roi_names=("motor", "visual")),
                ]
            )

    def test_rejects_mixed_or_inconsistent_tr(self):
        with self.assertRaisesRegex(ValueError, "set for every run"):
            TimeSeriesDataset(
                [make_run("sub-001", "off"), make_run("sub-002", "off", tr=None)]
            )
        with self.assertRaisesRegex(ValueError, "same tr"):
            TimeSeriesDataset(
                [make_run("sub-001", "off"), make_run("sub-002", "off", tr=1.0)]
            )

    def test_rejects_duplicate_subject_session_acquisition(self):
        with self.assertRaisesRegex(ValueError, "duplicate subject/session"):
            TimeSeriesDataset(
                [make_run("sub-001", "off"), make_run("sub-001", "off")]
            )

    def test_allows_multiple_acquisitions_within_one_subject_session(self):
        dataset = TimeSeriesDataset(
            [
                make_run("sub-001", "off", acquisition_id="task-rest_run-1"),
                make_run("sub-001", "off", acquisition_id="task-rest_run-2"),
            ]
        )
        self.assertEqual(dataset.n_runs, 2)

    def test_rejects_acquisition_id_without_subject(self):
        with self.assertRaisesRegex(ValueError, "acquisition_id.*subject"):
            TimeSeriesDataset([make_run(None, None, acquisition_id="run-1")])

    def test_learned_operation_requires_subject_ids(self):
        dataset = TimeSeriesDataset([make_run(None, None)])
        with self.assertRaisesRegex(ValueError, "PCA fit requires subject IDs"):
            dataset.require_subject_ids("PCA fit")

    def test_different_sessions_of_one_subject_cannot_cross_split(self):
        training = [make_run("sub-001", "off"), make_run("sub-002", "off")]
        test = [make_run("sub-001", "on"), make_run("sub-003", "off")]

        with self.assertRaisesRegex(ValueError, "sub-001"):
            validate_subject_disjoint(training, test)

    def test_disjoint_subject_split_is_accepted(self):
        validate_subject_disjoint(
            [make_run("sub-001", "off"), make_run("sub-001", "on")],
            [make_run("sub-002", "off"), make_run("sub-003", "off")],
        )


if __name__ == "__main__":
    unittest.main()
