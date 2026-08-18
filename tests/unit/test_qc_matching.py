import importlib.util
import unittest

import numpy as np

from dfckit.qc import (
    match_within_subject,
    matched_subject_differences,
    summarize_window_motion,
)

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


class WindowMotionSummaryTests(unittest.TestCase):
    def test_window_statistics_use_inclusive_original_frame_bounds(self):
        fd = np.asarray([np.nan, 0.1, 0.2, 0.3, 0.4, 0.5])

        result = summarize_window_motion(fd, [0, 3], [2, 5])

        np.testing.assert_allclose(result.mean, [0.15, 0.4])
        np.testing.assert_allclose(result.p95, [0.195, 0.49])
        np.testing.assert_allclose(result.maximum, [0.2, 0.5])
        np.testing.assert_array_equal(result.finite_frame_counts, [2, 3])
        np.testing.assert_allclose(result.covariates, np.column_stack([result.mean, result.p95]))

    def test_too_many_nonfinite_values_and_invalid_bounds_are_rejected(self):
        fd = np.asarray([np.nan, np.nan, 0.2, 0.3])
        with self.assertRaisesRegex(ValueError, "contains 2 non-finite"):
            summarize_window_motion(fd, [0], [2])
        with self.assertRaisesRegex(ValueError, "bounds"):
            summarize_window_motion(fd, [2], [4])


@unittest.skipUnless(SCIPY_AVAILABLE, "requires dfc-kit[qc]")
class WithinSubjectMatchingTests(unittest.TestCase):
    def test_assignment_maximizes_valid_pairs_before_minimizing_cost(self):
        result = match_within_subject(
            left_covariates=[0.0, -0.9],
            left_subjects=["sub-a", "sub-a"],
            right_covariates=[0.0, 0.9],
            right_subjects=["sub-a", "sub-a"],
            calipers=[1.0],
            covariate_names=["mean_fd"],
        )

        self.assertEqual(result.n_pairs, 2)
        np.testing.assert_array_equal(result.left_indices, [0, 1])
        np.testing.assert_array_equal(result.right_indices, [1, 0])
        self.assertEqual(len(set(result.left_indices.tolist())), 2)
        self.assertEqual(len(set(result.right_indices.tolist())), 2)

    def test_matching_stays_within_subject_and_respects_all_calipers(self):
        left = np.asarray(
            [[0.10, 0.20], [0.12, 0.22], [0.50, 0.60], [0.10, 0.20]]
        )
        right = np.asarray(
            [[0.101, 0.201], [0.118, 0.219], [0.40, 0.60], [0.105, 0.205]]
        )
        result = match_within_subject(
            left,
            ["sub-a", "sub-a", "sub-a", "sub-b"],
            right,
            ["sub-a", "sub-a", "sub-a", "sub-b"],
            calipers=[0.01, 0.02],
            covariate_names=["mean_fd", "p95_fd"],
            ceilings=[0.20, np.inf],
            minimum_pairs=1,
        )

        self.assertEqual(result.subjects, ("sub-a", "sub-a", "sub-b"))
        self.assertEqual(result.n_pairs, 3)
        self.assertTrue(np.all(result.absolute_differences <= result.calipers))
        self.assertNotIn(2, result.left_indices)
        self.assertNotIn(2, result.right_indices)

    def test_subject_below_minimum_pair_count_is_dropped_entirely(self):
        result = match_within_subject(
            [0.10, 0.20],
            ["sub-a", "sub-a"],
            [0.11, 0.50],
            ["sub-a", "sub-a"],
            calipers=[0.025],
            minimum_pairs=2,
        )

        self.assertEqual(result.n_pairs, 0)
        self.assertEqual(result.absolute_differences.shape, (0, 1))

    def test_caliper_and_ceiling_boundaries_are_inclusive(self):
        result = match_within_subject(
            [0.10],
            ["sub-a"],
            [0.20],
            ["sub-a"],
            calipers=[0.10],
            ceilings=[0.20],
        )

        self.assertEqual(result.n_pairs, 1)
        self.assertAlmostEqual(result.absolute_differences[0, 0], 0.10)
        self.assertAlmostEqual(result.scaled_costs[0], 1.0)

    def test_matched_endpoint_differences_average_within_subject(self):
        matches = match_within_subject(
            [0.10, 0.20, 0.30],
            ["sub-a", "sub-a", "sub-b"],
            [0.11, 0.19, 0.31],
            ["sub-a", "sub-a", "sub-b"],
            calipers=[0.02],
        )
        left_endpoints = np.asarray([[1.0, 4.0], [2.0, 6.0], [10.0, 20.0]])
        right_endpoints = np.asarray([[2.0, 5.0], [4.0, 8.0], [13.0, 18.0]])

        result = matched_subject_differences(
            matches,
            left_endpoints,
            right_endpoints,
            endpoint_names=("amplitude", "occupancy"),
        )

        self.assertEqual(result.subjects, ("sub-a", "sub-b"))
        np.testing.assert_array_equal(result.pair_counts, [2, 1])
        np.testing.assert_allclose(result.differences, [[1.5, 1.5], [3.0, -2.0]])

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one value per covariate"):
            match_within_subject(
                [[0.1, 0.2]],
                ["sub-a"],
                [[0.1, 0.2]],
                ["sub-a"],
                calipers=[0.1],
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            match_within_subject(
                [0.1],
                ["sub-a"],
                [0.1],
                ["sub-a"],
                calipers=[0.0],
            )


if __name__ == "__main__":
    unittest.main()
