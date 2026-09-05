import unittest

import numpy as np

from dfckit import TimeSeriesDataset, TimeSeriesRun
from dfckit.connectivity import (
    SlidingWindowFC,
    adjacent_window_pattern_similarity,
    all_pair_window_pattern_similarity,
    summarize_window_pattern_dataset,
    window_pattern_adjacency_excess,
)


class SlidingWindowFCTests(unittest.TestCase):
    def test_metadata_preserves_segment_boundaries(self):
        rng = np.random.default_rng(17)
        run = TimeSeriesRun(
            values=rng.normal(size=(12, 4)),
            original_indices=np.array([0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14]),
            roi_names=("a", "b", "c", "d"),
            subject="sub-001",
            session="off",
        )
        result = SlidingWindowFC(length=4, step=2).transform(run)
        np.testing.assert_array_equal(result.start_frames, [0, 8, 10])
        np.testing.assert_array_equal(result.end_frames, [3, 11, 13])
        np.testing.assert_array_equal(result.segment_ids, [0, 1, 1])
        self.assertEqual(result.features.shape, (3, 6))

    def test_raises_when_no_segment_is_long_enough(self):
        run = TimeSeriesRun(
            values=np.arange(12, dtype=float).reshape(6, 2),
            original_indices=np.array([0, 1, 4, 5, 8, 9]),
            roi_names=("a", "b"),
        )
        with self.assertRaisesRegex(ValueError, "no contiguous segment"):
            SlidingWindowFC(length=3, step=1).transform(run)

    def test_adjacent_pattern_similarity_stays_inside_segments(self):
        rng = np.random.default_rng(29)
        run = TimeSeriesRun(
            values=rng.normal(size=(16, 4)),
            original_indices=np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            roi_names=("a", "b", "c", "d"),
        )
        windows = SlidingWindowFC(length=4, step=2).transform(run)

        result = adjacent_window_pattern_similarity(windows)

        np.testing.assert_array_equal(result.left_windows, [0, 2, 3, 4])
        np.testing.assert_array_equal(result.right_windows, [1, 3, 4, 5])
        expected = [
            np.corrcoef(windows.features[left], windows.features[right])[0, 1]
            for left, right in zip(result.left_windows, result.right_windows, strict=True)
        ]
        np.testing.assert_allclose(result.similarities, expected)
        self.assertTrue(
            np.all(
                windows.segment_ids[result.left_windows]
                == windows.segment_ids[result.right_windows]
            )
        )

    def test_adjacent_pattern_similarity_can_return_no_pairs(self):
        run = TimeSeriesRun(
            values=np.arange(24, dtype=float).reshape(12, 2),
            original_indices=np.array([0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17]),
            roi_names=("a", "b"),
        )
        windows = SlidingWindowFC(length=3, step=1).transform(run)

        result = adjacent_window_pattern_similarity(windows)

        self.assertEqual(len(result.similarities), 0)

    def test_all_pair_similarity_uses_within_segment_m_minus_one_weights(self):
        rng = np.random.default_rng(31)
        run = TimeSeriesRun(
            values=rng.normal(size=(22, 4)),
            original_indices=np.r_[np.arange(8), np.arange(20, 34)],
            roi_names=("a", "b", "c", "d"),
        )
        windows = SlidingWindowFC(length=4, step=2).transform(run)

        segment_means = []
        weights = []
        for segment in np.unique(windows.segment_ids):
            patterns = windows.features[windows.segment_ids == segment]
            if len(patterns) < 2:
                continue
            correlation = np.corrcoef(patterns)
            segment_means.append(float(np.mean(correlation[np.triu_indices(len(patterns), 1)])))
            weights.append(len(patterns) - 1)
        expected = np.average(segment_means, weights=weights)

        observed = all_pair_window_pattern_similarity(windows)
        self.assertAlmostEqual(observed, expected)
        adjacent = adjacent_window_pattern_similarity(windows)
        self.assertAlmostEqual(
            window_pattern_adjacency_excess(windows),
            float(np.mean(adjacent.similarities) - observed),
        )

    def test_all_pair_similarity_requires_a_segment_with_two_windows(self):
        run = TimeSeriesRun(
            values=np.arange(24, dtype=float).reshape(12, 2),
            original_indices=np.array([0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17]),
            roi_names=("a", "b"),
        )
        windows = SlidingWindowFC(length=3, step=1).transform(run)

        with self.assertRaisesRegex(ValueError, "two windows"):
            all_pair_window_pattern_similarity(windows)
        with self.assertRaisesRegex(ValueError, "two windows"):
            window_pattern_adjacency_excess(windows)

    def test_dataset_summary_exposes_all_pair_and_order_specific_endpoints(self):
        rng = np.random.default_rng(37)
        run = TimeSeriesRun(
            values=rng.normal(size=(24, 4)),
            original_indices=np.arange(24),
            roi_names=("a", "b", "c", "d"),
            subject="sub-001",
            session="off",
            acquisition_id="sub-001_ses-off_task-rest",
        )
        estimator = SlidingWindowFC(length=6, step=6)
        payload = summarize_window_pattern_dataset(
            TimeSeriesDataset((run,)), estimator
        )

        self.assertEqual(payload["format"], "dfc-kit-window-pattern-endpoints")
        self.assertEqual(payload["format_version"], 1)
        self.assertEqual(len(payload["rows"]), 3)
        by_measure = {row["measure"]: row for row in payload["rows"]}
        self.assertEqual(
            set(by_measure),
            {"all_pair_similarity", "adjacent_similarity", "adjacency_excess"},
        )
        self.assertAlmostEqual(
            by_measure["adjacency_excess"]["value"],
            by_measure["adjacent_similarity"]["value"]
            - by_measure["all_pair_similarity"]["value"],
        )
        self.assertTrue(all(row["n_windows"] == 4 for row in payload["rows"]))
        self.assertTrue(
            all(row["n_adjacent_pairs"] == 3 for row in payload["rows"])
        )


if __name__ == "__main__":
    unittest.main()
