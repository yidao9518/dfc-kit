import unittest

import numpy as np

from dfckit.inference import (
    benjamini_hochberg,
    hc3_confidence_interval,
    ols_hc3,
    paired_bootstrap_mean_ci,
    paired_hc3,
    paired_sign_flip,
)


class MultipleTestingTests(unittest.TestCase):
    def test_bh_matches_known_family_and_preserves_missing_values(self):
        result = benjamini_hochberg(
            [0.01, 0.04, np.nan, 0.03, 0.002],
            family="four prespecified circuit endpoints",
        )

        np.testing.assert_allclose(
            result.adjusted_pvalues[[0, 1, 3, 4]],
            [0.02, 0.04, 0.04, 0.008],
        )
        self.assertTrue(np.isnan(result.adjusted_pvalues[2]))
        self.assertEqual(result.n_tests, 4)
        self.assertEqual(result.family, "four prespecified circuit endpoints")

    def test_bh_requires_an_explicit_family(self):
        with self.assertRaisesRegex(ValueError, "family"):
            benjamini_hochberg([0.1, 0.2], family="")


class PairedInferenceTests(unittest.TestCase):
    def test_exact_sign_flip_has_known_probability(self):
        result = paired_sign_flip(
            [1.0, 2.0],
            ["sub-001", "sub-002"],
            exact=True,
        )

        self.assertEqual(result.estimate, 1.5)
        self.assertEqual(result.pvalue, 0.5)
        self.assertEqual(result.n_permutations_performed, 4)
        self.assertIsNone(result.seed)
        self.assertEqual(result.permutation_unit, "participant")

    def test_monte_carlo_sign_flip_is_reproducible(self):
        arguments = {
            "differences": [-0.2, -0.5, 0.1, -0.4],
            "subject_ids": ["s1", "s2", "s3", "s4"],
            "n_permutations": 500,
            "seed": 17,
        }
        first = paired_sign_flip(**arguments)
        second = paired_sign_flip(**arguments)

        self.assertEqual(first, second)

    def test_duplicate_subjects_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            paired_sign_flip([1.0, 2.0], ["sub-001", "sub-001"], exact=True)

    def test_bootstrap_matches_direct_seeded_resampling(self):
        values = np.asarray([1.0, 2.0, 4.0])
        result = paired_bootstrap_mean_ci(
            values,
            ["s1", "s2", "s3"],
            n_resamples=200,
            seed=19,
            confidence=0.90,
        )
        rng = np.random.default_rng(19)
        indices = rng.integers(0, 3, size=(200, 3))
        expected = np.quantile(values[indices].mean(axis=1), [0.05, 0.95])

        np.testing.assert_allclose([result.lower, result.upper], expected)
        self.assertEqual(result.resampling_unit, "participant")


class HC3Tests(unittest.TestCase):
    def test_intercept_only_model_has_known_t_probability(self):
        center = 1.0 / np.sqrt(2.0)
        outcome = np.asarray([center - 1.0, center, center + 1.0])
        result = ols_hc3(
            outcome,
            np.ones((3, 1)),
            column_names=("intercept",),
        )

        np.testing.assert_allclose(result.coefficients, [center])
        np.testing.assert_allclose(result.standard_errors, [center])
        np.testing.assert_allclose(result.statistics, [1.0])
        np.testing.assert_allclose(result.pvalues, [0.42264973081037427], rtol=1e-12)
        self.assertEqual(result.degrees_of_freedom, 2)

        lower, upper = hc3_confidence_interval(result)
        critical = (upper[0] - result.coefficients[0]) / result.standard_errors[0]
        self.assertAlmostEqual(critical, 4.302652729749, places=10)
        self.assertLess(lower[0], result.coefficients[0])
        self.assertGreater(upper[0], result.coefficients[0])

    def test_hc3_matches_direct_sandwich_calculation(self):
        x = np.column_stack((np.ones(7), np.linspace(-1.0, 1.0, 7)))
        y = np.asarray([0.1, 0.4, 0.2, 0.9, 1.2, 0.7, 1.5])
        result = ols_hc3(y, x, column_names=("intercept", "slope"))

        inverse = np.linalg.inv(x.T @ x)
        beta = inverse @ x.T @ y
        residual = y - x @ beta
        leverage = np.sum((x @ inverse) * x, axis=1)
        scaled = residual / (1.0 - leverage)
        covariance = inverse @ (x.T @ (x * scaled[:, None] ** 2)) @ inverse
        np.testing.assert_allclose(result.coefficients, beta)
        np.testing.assert_allclose(result.covariance, covariance)
        np.testing.assert_allclose(result.standard_errors, np.sqrt(np.diag(covariance)))

    def test_paired_hc3_records_direction_and_zero_covariate_estimand(self):
        result = paired_hc3(
            differences=[-0.2, -0.1, 0.2, 0.4, 0.3],
            covariates=[-0.2, -0.1, 0.0, 0.1, 0.3],
            subject_ids=["s1", "s2", "s3", "s4", "s5"],
            covariate_names=["delta_censor"],
            difference_direction="ON minus OFF",
        )

        self.assertEqual(result.model.column_names, ("intercept", "delta_censor"))
        self.assertEqual(result.difference_direction, "ON minus OFF")
        self.assertIn("equals zero", result.estimand)

    def test_unit_leverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "leverage"):
            ols_hc3(
                [1.0, 2.0, 3.0],
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                column_names=("first", "second"),
            )

    def test_zero_residual_degrees_of_freedom_is_rejected_before_leverage(self):
        with self.assertRaisesRegex(ValueError, "positive residual degrees of freedom"):
            ols_hc3(
                [1.0, 2.0],
                np.eye(2),
                column_names=("first", "second"),
            )


if __name__ == "__main__":
    unittest.main()
