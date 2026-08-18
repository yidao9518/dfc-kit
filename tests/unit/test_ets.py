import unittest

import numpy as np

from dfckit import TimeSeriesDataset, TimeSeriesRun
from dfckit.connectivity import ETS, edge_rss, fit_ets_event_threshold


def make_run(subject="sub-001", session="off"):
    return TimeSeriesRun(
        values=np.asarray(
            [
                [0.0, 1.0, 5.0],
                [1.0, 3.0, 5.0],
                [3.0, 2.0, 5.0],
                [100.0, -4.0, 0.0],
                [102.0, -1.0, 2.0],
                [107.0, 5.0, 1.0],
            ]
        ),
        original_indices=np.asarray([0, 1, 2, 6, 7, 8]),
        roi_names=("visual", "motor", "putamen"),
        subject=subject,
        session=session,
        tr=0.8,
    )


class ETSTests(unittest.TestCase):
    def test_materialized_edges_match_direct_segment_standardization(self):
        run = make_run()
        result = ETS().transform(run)
        first = run.values[:3]
        second = run.values[3:]
        standardized = np.vstack(
            [
                (first - first.mean(axis=0))
                / np.where(first.std(axis=0) < 1e-8, 1.0, first.std(axis=0)),
                (second - second.mean(axis=0)) / second.std(axis=0),
            ]
        )
        expected = np.column_stack(
            [
                standardized[:, 0] * standardized[:, 1],
                standardized[:, 0] * standardized[:, 2],
                standardized[:, 1] * standardized[:, 2],
            ]
        )

        np.testing.assert_allclose(result.features, expected)
        np.testing.assert_array_equal(result.edge_i, [0, 0, 1])
        np.testing.assert_array_equal(result.edge_j, [1, 2, 2])

    def test_rss_only_path_matches_materialized_edge_norm(self):
        run = make_run()
        complete = ETS().transform(run)
        rss_only = ETS().rss(run)

        np.testing.assert_allclose(rss_only.rss, np.sqrt((complete.features**2).sum(axis=1)))
        np.testing.assert_allclose(rss_only.rss, complete.rss)

    def test_frame_metadata_preserves_gaps_and_singletons(self):
        run = make_run()
        result = ETS().rss(run)

        np.testing.assert_array_equal(result.original_indices, [0, 1, 2, 6, 7, 8])
        np.testing.assert_array_equal(result.segment_ids, [0, 0, 0, 1, 1, 1])

        singleton_run = TimeSeriesRun(
            values=np.vstack([[9.0, 9.0, 9.0], run.values]),
            original_indices=np.asarray([0, 3, 4, 5, 9, 10, 11]),
            roi_names=run.roi_names,
        )
        singleton_result = ETS().rss(singleton_run)
        np.testing.assert_array_equal(singleton_result.original_indices, [3, 4, 5, 9, 10, 11])
        np.testing.assert_array_equal(singleton_result.segment_ids, [1, 1, 1, 2, 2, 2])

    def test_optimized_identity_matches_explicit_edges(self):
        rng = np.random.default_rng(12)
        standardized = rng.normal(size=(17, 8))
        left, right = np.triu_indices(8, k=1)
        explicit = standardized[:, left] * standardized[:, right]

        np.testing.assert_allclose(edge_rss(standardized), np.linalg.norm(explicit, axis=1))

    def test_event_threshold_records_fit_subjects_and_classifies(self):
        dataset = TimeSeriesDataset(
            [make_run("sub-001", "off"), make_run("sub-002", "off")]
        )
        threshold = fit_ets_event_threshold(dataset, percentile=80.0)
        result = ETS().rss(dataset.runs[0])

        self.assertEqual(threshold.fit_subjects, ("sub-001", "sub-002"))
        np.testing.assert_array_equal(threshold.event_mask(result), result.rss >= threshold.value)

    def test_event_threshold_requires_subject_ids(self):
        dataset = TimeSeriesDataset([make_run(None, None)])
        with self.assertRaisesRegex(ValueError, "requires subject IDs"):
            fit_ets_event_threshold(dataset)

    def test_no_two_frame_segment_is_rejected(self):
        run = TimeSeriesRun(
            values=np.asarray([[0.0, 1.0], [2.0, 3.0]]),
            original_indices=np.asarray([0, 4]),
            roi_names=("visual", "motor"),
        )
        with self.assertRaisesRegex(ValueError, "two frames"):
            ETS().rss(run)


if __name__ == "__main__":
    unittest.main()
