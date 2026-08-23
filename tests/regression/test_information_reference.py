import importlib.util
import unittest

import numpy as np

from dfckit import TimeSeriesRun
from dfckit.information import block_information, knn_cmi, knn_mi, sample_fixed_windows

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy is not installed")
class AuditedInformationRegressionTests(unittest.TestCase):
    def test_kernels_and_block_means_match_frozen_research_outputs(self):
        rng = np.random.default_rng(20260818)
        values = rng.normal(size=(240, 7))
        values[:, 3] += 0.65 * values[:, 0]
        values[:, 4] += 0.45 * values[:, 1]

        self.assertEqual(
            knn_mi(values[:, 0], values[:, 3], k=3),
            0.1847943721630072,
        )
        self.assertEqual(
            knn_cmi(values[:, 0], values[:, 3], values[:, 6], k=3),
            0.22145108731264973,
        )
        result = block_information(values, [0, 1], [3, 4], conditioning=[5, 6])
        self.assertEqual(result.mean_mutual_information, 0.06705220787794008)
        self.assertEqual(
            result.mean_conditional_mutual_information,
            0.07152210379839707,
        )

    def test_uniform_start_sampler_matches_frozen_research_draws(self):
        rng = np.random.default_rng(20260818)
        values = rng.normal(size=(240, 7))
        values[:, 3] += 0.65 * values[:, 0]
        values[:, 4] += 0.45 * values[:, 1]
        segments = [rng.normal(size=(150, 2)), rng.normal(size=(200, 2))]
        run = TimeSeriesRun(
            values=np.vstack(segments),
            original_indices=np.r_[np.arange(150), np.arange(160, 360)],
            roi_names=("a", "b"),
        )

        sampled = sample_fixed_windows(run, length=120, draws=25, seed=17)
        observed = list(
            zip(
                sampled.segment_ids.tolist(),
                sampled.starts_within_segment.tolist(),
                strict=True,
            )
        )
        expected = [
            (1, 51), (1, 63), (0, 12), (0, 18), (1, 20),
            (1, 31), (1, 54), (1, 10), (0, 4), (0, 24),
            (1, 20), (1, 12), (1, 71), (1, 16), (1, 40),
            (1, 37), (0, 10), (1, 51), (0, 5), (0, 1),
            (1, 34), (0, 28), (1, 9), (1, 36), (0, 17),
        ]
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
