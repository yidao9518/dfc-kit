import unittest
from dataclasses import replace

import numpy as np

from dfckit.states import select_state_count


class StateCountSelectionTests(unittest.TestCase):
    def test_lower_is_better_and_one_standard_error_prefers_smaller_k(self):
        result = select_state_count(
            [
                [1.02, 0.965, 0.97],
                [1.01, 0.95, 0.95],
                [0.99, 0.96, 0.94],
                [1.00, 0.955, 0.96],
            ],
            [2, 4, 6],
            higher_is_better=False,
        )
        self.assertEqual(result.best_n_states, 6)
        self.assertEqual(result.one_standard_error_n_states, 4)
        self.assertEqual(result.ranks.tolist(), [3, 2, 1])
        self.assertEqual(result.within_one_standard_error.tolist(), [False, True, True])
        self.assertEqual(result.n_folds, 4)

    def test_higher_is_better_sorts_candidates_and_breaks_tie_by_smaller_k(self):
        result = select_state_count(
            [[-2.0, -1.0, -1.0], [-2.0, -1.0, -1.0]],
            [6, 2, 4],
            higher_is_better=True,
        )
        self.assertEqual(result.candidate_n_states.tolist(), [2, 4, 6])
        self.assertEqual(result.best_n_states, 2)
        self.assertEqual(result.one_standard_error_n_states, 2)
        self.assertEqual(result.ranks.tolist(), [1, 2, 3])
        np.testing.assert_allclose(result.fold_standard_errors, 0.0)

    def test_invalid_shapes_values_and_fold_count_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two folds"):
            select_state_count([[1.0, 2.0]], [2, 3], higher_is_better=False)
        with self.assertRaisesRegex(ValueError, "finite"):
            select_state_count(
                [[1.0, np.nan], [2.0, 3.0]],
                [2, 3],
                higher_is_better=False,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            select_state_count(
                [[1.0, 2.0], [2.0, 3.0]],
                [2, 2],
                higher_is_better=False,
            )
        with self.assertRaisesRegex(TypeError, "boolean"):
            select_state_count(
                [[1.0, 2.0], [2.0, 3.0]],
                [2, 3],
                higher_is_better=1,
            )
        with self.assertRaisesRegex(TypeError, "integers"):
            select_state_count(
                [[1.0, 2.0], [2.0, 3.0]],
                [2.0, 3.5],
                higher_is_better=False,
            )

    def test_derived_statistics_and_decisions_cannot_be_tampered(self):
        result = select_state_count(
            [[1.0, 0.9, 0.8], [1.1, 0.95, 0.85]],
            [2, 4, 6],
            higher_is_better=False,
        )
        changed_means = result.mean_scores.copy()
        changed_means[0] += 0.1
        with self.assertRaisesRegex(ValueError, "means disagree"):
            replace(result, mean_scores=changed_means)
        with self.assertRaisesRegex(ValueError, "best state count disagrees"):
            replace(result, best_n_states=2)
        with self.assertRaisesRegex(ValueError, "disagree"):
            replace(result, higher_is_better=True)


if __name__ == "__main__":
    unittest.main()
