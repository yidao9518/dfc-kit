import unittest

import numpy as np

from dfckit.connectivity import conditional_granger
from dfckit.data import TimeSeriesRun


def _run(values, original_indices=None):
    if original_indices is None:
        original_indices = np.arange(len(values))
    return TimeSeriesRun(values, original_indices, tuple("xyz"[: values.shape[1]]))


class ConditionalGrangerTests(unittest.TestCase):
    def test_detects_known_source_to_target_direction(self):
        rng = np.random.default_rng(41)
        values = np.zeros((1200, 2))
        noise = rng.normal(size=values.shape)
        for time in range(1, len(values)):
            values[time, 0] = 0.55 * values[time - 1, 0] + noise[time, 0]
            values[time, 1] = (
                0.45 * values[time - 1, 1] + 0.8 * values[time - 1, 0] + noise[time, 1]
            )
        run = _run(values)
        forward = conditional_granger(run, source=[0], target=[1], lag_order=1)
        reverse = conditional_granger(run, source=[1], target=[0], lag_order=1)
        self.assertGreater(forward.geweke, 0.35)
        self.assertLess(reverse.geweke, 0.02)
        self.assertTrue(forward.stable)

    def test_conditioning_removes_common_driver_predictability(self):
        rng = np.random.default_rng(17)
        values = np.zeros((2500, 3))
        noise = rng.normal(size=values.shape)
        for time in range(2, len(values)):
            values[time, 2] = 0.7 * values[time - 1, 2] + noise[time, 2]
            values[time, 0] = 0.5 * values[time - 1, 2] + noise[time, 0]
            values[time, 1] = 0.9 * values[time - 2, 2] + noise[time, 1]
        run = _run(values)
        unconditioned = conditional_granger(run, source=[0], target=[1], lag_order=2)
        conditioned = conditional_granger(
            run, source=[0], target=[1], conditioning=[2], lag_order=2
        )
        self.assertGreater(unconditioned.geweke, conditioned.geweke + 0.05)
        self.assertLess(conditioned.geweke, 0.02)

    def test_lag_rows_never_cross_censor_gap(self):
        rng = np.random.default_rng(9)
        values = rng.normal(size=(10, 2))
        indices = np.asarray([0, 1, 2, 3, 4, 20, 21, 22, 23, 24])
        result = conditional_granger(_run(values, indices), source=[0], target=[1], lag_order=2)
        self.assertEqual(result.n_segments, 2)
        self.assertEqual(result.n_observations, 6)

    def test_independent_series_have_negligible_effect(self):
        rng = np.random.default_rng(22)
        values = rng.normal(size=(3000, 2))
        result = conditional_granger(_run(values), source=[0], target=[1], lag_order=2)
        self.assertLess(result.geweke, 0.01)

    def test_short_and_rank_deficient_designs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "long enough"):
            conditional_granger(
                _run(np.arange(6, dtype=float).reshape(3, 2)), source=[0], target=[1], lag_order=3
            )
        values = np.column_stack([np.arange(20, dtype=float), np.arange(20, dtype=float)])
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            conditional_granger(_run(values), source=[0], target=[1], lag_order=1)


if __name__ == "__main__":
    unittest.main()
