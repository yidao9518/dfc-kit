import importlib.util
import unittest

import numpy as np

from dfckit import TimeSeriesRun
from dfckit.information import (
    FixedLengthInformation,
    block_information,
    knn_cmi,
    knn_mi,
    sample_fixed_windows,
)
from dfckit.information.estimators import _eligible_fixed_window_count

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


class FixedWindowSamplingTests(unittest.TestCase):
    def setUp(self):
        first = np.arange(150, dtype=float)
        second = 10000.0 + np.arange(200, dtype=float)
        values = np.concatenate([first, second])[:, None]
        self.run = TimeSeriesRun(
            values=values,
            original_indices=np.r_[np.arange(150), np.arange(160, 360)],
            roi_names=("signal",),
            subject="sub-001",
            session="off",
        )

    def test_exact_length_and_censor_gap_are_preserved(self):
        sampled = sample_fixed_windows(self.run, length=120, draws=200, seed=7)

        self.assertEqual(_eligible_fixed_window_count(self.run, 120), 31 + 81)
        self.assertEqual(sampled.values.shape, (200, 120, 1))
        self.assertEqual(sampled.original_indices.shape, (200, 120))
        self.assertEqual(sampled.draw_indices.tolist(), list(range(200)))
        for values, original in zip(sampled.values, sampled.original_indices, strict=True):
            self.assertTrue(values.max() < 10000 or values.min() >= 10000)
            np.testing.assert_array_equal(np.diff(original), 1)
        np.testing.assert_array_equal(sampled.start_frames, sampled.original_indices[:, 0])
        np.testing.assert_array_equal(sampled.end_frames, sampled.original_indices[:, -1])
        self.assertEqual(sampled.seed, 7)

    def test_sampling_is_uniform_over_starts_not_segments(self):
        sampled = sample_fixed_windows(self.run, length=120, draws=11200, seed=29)
        keys = list(zip(sampled.segment_ids, sampled.starts_within_segment, strict=True))
        counts = np.asarray(list(__import__("collections").Counter(keys).values()))

        self.assertEqual(len(counts), 112)
        self.assertLess(np.max(np.abs(counts - 100)), 35)
        # The longer second segment owns 81/112 starts, rather than half the draws.
        fraction_second = np.mean(sampled.segment_ids == 1)
        self.assertAlmostEqual(fraction_second, 81 / 112, delta=0.02)

    def test_sampling_is_reproducible_and_rejects_ineligible_length(self):
        first = sample_fixed_windows(self.run, length=80, draws=20, seed=11)
        second = sample_fixed_windows(self.run, length=80, draws=20, seed=11)
        np.testing.assert_array_equal(first.segment_ids, second.segment_ids)
        np.testing.assert_array_equal(first.starts_within_segment, second.starts_within_segment)
        with self.assertRaisesRegex(ValueError, "no contiguous retained segment"):
            sample_fixed_windows(self.run, length=201, draws=1, seed=0)


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy is not installed")
class InformationKernelTests(unittest.TestCase):
    def test_dependent_gaussian_pair_exceeds_independent_pair(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=600)
        dependent = x + rng.normal(scale=0.08, size=600)
        independent = rng.normal(size=600)

        dependent_mi = knn_mi(x, dependent)
        independent_mi = knn_mi(x, independent)
        self.assertGreater(dependent_mi, independent_mi + 1.5)
        self.assertLess(abs(independent_mi), 0.15)

    def test_conditioning_removes_common_driver_dependence(self):
        rng = np.random.default_rng(31)
        driver = rng.normal(size=1000)
        x = driver + rng.normal(scale=0.3, size=1000)
        y = driver + rng.normal(scale=0.3, size=1000)

        marginal = knn_mi(x, y)
        conditional = knn_cmi(x, y, driver)
        self.assertGreater(marginal, 0.5)
        self.assertLess(abs(conditional), 0.15)

    def test_seeded_tie_jitter_is_deterministic(self):
        rng = np.random.default_rng(41)
        x = np.round(rng.normal(size=240), 1)
        y = np.round(0.7 * x + rng.normal(size=240), 1)

        first = knn_mi(x, y, jitter=1e-3, seed=7)
        second = knn_mi(x, y, jitter=1e-3, seed=7)
        different = knn_mi(x, y, jitter=1e-3, seed=8)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_block_aggregation_and_fixed_length_transform(self):
        rng = np.random.default_rng(47)
        driver = rng.normal(size=260)
        values = np.column_stack(
            [
                driver + rng.normal(scale=0.2, size=260),
                driver + rng.normal(scale=0.3, size=260),
                driver + rng.normal(scale=0.2, size=260),
                driver + rng.normal(scale=0.3, size=260),
                driver,
            ]
        )
        direct = block_information(values[:120], [0, 1], [2, 3], conditioning=[4])
        self.assertEqual(direct.mutual_information.shape, (2, 2))
        self.assertEqual(direct.conditional_mutual_information.shape, (2, 2))
        self.assertAlmostEqual(direct.mean_mutual_information, direct.mutual_information.mean())

        run = TimeSeriesRun(
            values=values,
            original_indices=np.r_[np.arange(130), np.arange(140, 270)],
            roi_names=("left-a", "left-b", "right-a", "right-b", "condition"),
            subject="sub-047",
        )
        result = FixedLengthInformation(
            length=100,
            draws=5,
            sample_seed=13,
        ).transform(run, [0, 1], [2, 3], conditioning=[4])

        self.assertEqual(result.mutual_information.shape, (5, 2, 2))
        self.assertEqual(result.conditional_mutual_information.shape, (5, 2, 2))
        self.assertEqual(result.mean_mutual_information.shape, (5,))
        self.assertEqual(result.mean_conditional_mutual_information.shape, (5,))
        self.assertTrue(np.all(result.samples.segment_ids[:-1] >= 0))
        with self.assertRaises(ValueError):
            result.mutual_information[0, 0, 0] = 0.0

    def test_parallel_transform_materializes_group_iterables_once(self):
        rng = np.random.default_rng(53)
        driver = rng.normal(size=140)
        values = np.column_stack(
            (
                driver + rng.normal(scale=0.2, size=140),
                driver + rng.normal(scale=0.3, size=140),
                driver + rng.normal(scale=0.2, size=140),
                driver,
            )
        )
        run = TimeSeriesRun(
            values=values,
            original_indices=np.arange(140),
            roi_names=("left-a", "left-b", "right", "condition"),
        )
        result = FixedLengthInformation(
            length=60,
            draws=4,
            sample_seed=17,
            jobs=2,
        ).transform(
            run,
            (node for node in (0, 1)),
            (node for node in (2,)),
            conditioning=(node for node in (3,)),
        )
        self.assertEqual(result.mutual_information.shape, (4, 2, 1))
        self.assertEqual(result.conditional_mutual_information.shape, (4, 2, 1))

    def test_invalid_blocks_removed_metric_option_and_short_series_are_rejected(self):
        values = np.arange(40, dtype=float).reshape(10, 4)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            block_information(values, [0, 1], [1, 2])
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'metric'"):
            knn_mi(values[:, 0], values[:, 1], metric="euclidean")
        with self.assertRaisesRegex(ValueError, r"k \+ 2"):
            knn_mi(values[:5, 0], values[:5, 1], k=3)


if __name__ == "__main__":
    unittest.main()
