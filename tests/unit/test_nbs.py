import unittest

import numpy as np

from dfckit.inference import intercept_t_statistic, paired_nbs, threshold_components


class NBSComponentTests(unittest.TestCase):
    def test_positive_and_negative_edges_never_share_a_component(self):
        edge_i, edge_j = np.triu_indices(4, 1)
        statistics = np.asarray([3.2, -3.1, 0.0, 3.4, 0.0, -4.0])

        components = threshold_components(statistics, edge_i, edge_j, 4, 2.0)

        self.assertEqual(components["positive"][0].node_indices, (0, 1, 2))
        self.assertEqual(components["negative"][0].node_indices, (0, 2, 3))
        self.assertTrue(
            np.all(statistics[list(components["positive"][0].edge_indices)] >= 2.0)
        )
        self.assertTrue(
            np.all(statistics[list(components["negative"][0].edge_indices)] <= -2.0)
        )

    def test_pooled_mode_joins_adjacent_opposite_sign_edges(self):
        edge_i, edge_j = np.triu_indices(4, 1)
        statistics = np.asarray([3.2, -3.1, 0.0, 3.4, 0.0, -4.0])

        components = threshold_components(
            statistics,
            edge_i,
            edge_j,
            4,
            2.0,
            component_sign_mode="pooled",
        )

        self.assertEqual(tuple(components), ("pooled",))
        self.assertEqual(components["pooled"][0].direction, "pooled")
        self.assertEqual(components["pooled"][0].node_indices, (0, 1, 2, 3))
        self.assertEqual(components["pooled"][0].edge_indices, (0, 1, 3, 5))

    def test_edge_extent_and_intensity_have_explicit_different_rankings(self):
        edge_i, edge_j = np.triu_indices(7, 1)
        lookup = {
            (int(left), int(right)): index
            for index, (left, right) in enumerate(zip(edge_i, edge_j, strict=True))
        }
        statistics = np.zeros(len(edge_i))
        for edge in ((0, 1), (1, 2), (0, 2)):
            statistics[lookup[edge]] = 2.1
        statistics[lookup[(3, 4)]] = 8.0

        extent = threshold_components(
            statistics, edge_i, edge_j, 7, 2.0, component_statistic="edge_extent"
        )["positive"]
        intensity = threshold_components(
            statistics,
            edge_i,
            edge_j,
            7,
            2.0,
            component_statistic="sum_abs_statistic",
        )["positive"]

        self.assertEqual(extent[0].node_indices, (0, 1, 2))
        self.assertEqual(extent[0].statistic_value, 3.0)
        self.assertEqual(intensity[0].node_indices, (3, 4))
        self.assertEqual(intensity[0].statistic_value, 8.0)

    def test_component_input_validation_rejects_duplicate_and_reversed_edges(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            threshold_components([3.0, 4.0], [0, 0], [1, 1], 3, 2.0)
        with self.assertRaisesRegex(ValueError, "edge_i < edge_j"):
            threshold_components([3.0], [1], [0], 3, 2.0)
        with self.assertRaisesRegex(ValueError, "component_sign_mode"):
            threshold_components(
                [3.0], [0], [1], 2, 2.0, component_sign_mode="unknown"
            )

    def test_edges_exactly_at_cutoff_are_not_suprathreshold(self):
        edge_i, edge_j = np.triu_indices(3, 1)

        components = threshold_components(
            [2.0, -2.0, 2.000001], edge_i, edge_j, 3, 2.0
        )

        self.assertEqual(components["negative"], ())
        self.assertEqual(components["positive"][0].edge_indices, (2,))


class PairedNBSTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.edge_i, self.edge_j = np.triu_indices(4, 1)
        self.differences = rng.normal(0.0, 0.45, size=(18, len(self.edge_i)))
        self.differences[:, 0] += 0.35
        self.differences[:, 5] -= 0.35
        self.confounds = rng.normal(size=(18, 2))
        self.subjects = [f"sub-{index:03d}" for index in range(18)]

    def _run(self, **overrides):
        arguments = {
            "differences": self.differences,
            "subject_ids": self.subjects,
            "edge_i": self.edge_i,
            "edge_j": self.edge_j,
            "n_nodes": 4,
            "thresholds": (2.0, 2.5),
            "n_permutations": 80,
            "seed": 1234,
            "difference_direction": "condition B minus condition A",
        }
        arguments.update(overrides)
        return paired_nbs(**arguments)

    def test_two_sided_null_is_tail_maximum_and_seed_is_deterministic(self):
        first = self._run()
        second = self._run()

        for threshold in (2.0, 2.5):
            observed = first.at_threshold(threshold)
            repeated = second.at_threshold(threshold)
            np.testing.assert_array_equal(
                observed.null_maximum,
                np.maximum(observed.null_positive, observed.null_negative),
            )
            np.testing.assert_array_equal(observed.null_maximum, repeated.null_maximum)
            self.assertEqual(observed.components, repeated.components)
        self.assertEqual(first.permutation_unit, "participant complete edge vector")
        self.assertIn("no correction", first.threshold_correction)

    def test_explicit_separate_mode_matches_default(self):
        default = self._run()
        explicit = self._run(component_sign_mode="separate")

        self.assertEqual(default.component_sign_mode, "separate")
        for threshold in (2.0, 2.5):
            default_threshold = default.at_threshold(threshold)
            explicit_threshold = explicit.at_threshold(threshold)
            self.assertEqual(default_threshold.components, explicit_threshold.components)
            self.assertEqual(default_threshold.pooled_components, ())
            self.assertIsNone(default_threshold.null_pooled)
            np.testing.assert_array_equal(
                default_threshold.null_maximum, explicit_threshold.null_maximum
            )

    def test_pooled_mode_has_its_own_null_and_plus_one_pvalues(self):
        pooled_result = self._run(
            thresholds=(1.5,), component_sign_mode="pooled"
        )
        pooled = pooled_result.at_threshold(1.5)
        separate = self._run(thresholds=(1.5,)).at_threshold(1.5)

        self.assertEqual(pooled.component_sign_mode, "pooled")
        self.assertIn("pooled", pooled_result.threshold_correction)
        self.assertEqual(pooled.positive_components, ())
        self.assertEqual(pooled.negative_components, ())
        self.assertIsNone(pooled.null_positive)
        self.assertIsNone(pooled.null_negative)
        np.testing.assert_array_equal(pooled.null_maximum, pooled.null_pooled)
        self.assertTrue(
            np.all(pooled.null_maximum >= separate.null_maximum)
        )
        self.assertTrue(np.any(pooled.null_maximum > separate.null_maximum))
        for component in pooled.components:
            expected = (
                1
                + np.count_nonzero(
                    pooled.null_maximum >= component.statistic_value
                )
            ) / 81
            self.assertEqual(component.direction, "pooled")
            self.assertEqual(component.fwe_pvalue, expected)

    def test_pooled_mode_rejects_one_sided_alternatives(self):
        for alternative in ("greater", "less"):
            with self.subTest(alternative=alternative), self.assertRaisesRegex(
                ValueError, "requires"
            ):
                self._run(component_sign_mode="pooled", alternative=alternative)

    def test_one_sided_tail_uses_only_requested_direction(self):
        greater = self._run(alternative="greater")
        less = self._run(alternative="less")

        for threshold in (2.0, 2.5):
            positive = greater.at_threshold(threshold)
            negative = less.at_threshold(threshold)
            self.assertEqual(positive.negative_components, ())
            self.assertEqual(negative.positive_components, ())
            np.testing.assert_array_equal(positive.null_maximum, positive.null_positive)
            np.testing.assert_array_equal(negative.null_maximum, negative.null_negative)

    def test_confounds_are_centered_and_reduced_model_is_recorded(self):
        shifted = self._run(
            confounds=self.confounds + np.asarray([100.0, -50.0]),
            confound_names=("motion", "censor"),
        )
        centered = self._run(
            confounds=self.confounds - self.confounds.mean(axis=0),
            confound_names=("motion", "censor"),
        )

        np.testing.assert_allclose(
            shifted.at_threshold(2.0).observed_t,
            centered.at_threshold(2.0).observed_t,
            atol=1e-13,
        )
        np.testing.assert_array_equal(
            shifted.at_threshold(2.0).null_maximum,
            centered.at_threshold(2.0).null_maximum,
        )
        self.assertIn("reduced-model", shifted.permutation_method)

    def test_component_fwe_pvalues_use_plus_one_maximum_null(self):
        result = self._run(thresholds=(1.5,))
        threshold = result.at_threshold(1.5)
        for component in threshold.components:
            expected = (1 + np.count_nonzero(
                threshold.null_maximum >= component.statistic_value
            )) / 81
            self.assertEqual(component.fwe_pvalue, expected)

    def test_t_statistic_matches_direct_intercept_only_formula(self):
        observed = intercept_t_statistic(self.differences)
        expected = self.differences.mean(axis=0) / (
            self.differences.std(axis=0, ddof=1) / np.sqrt(len(self.differences))
        )
        np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-13)

    def test_design_and_subject_validation_are_strict(self):
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            self._run(
                confounds=np.ones((18, 2)),
                confound_names=("same-a", "same-b"),
            )
        duplicated = self.subjects.copy()
        duplicated[-1] = duplicated[0]
        with self.assertRaisesRegex(ValueError, "unique"):
            self._run(subject_ids=duplicated)


if __name__ == "__main__":
    unittest.main()
