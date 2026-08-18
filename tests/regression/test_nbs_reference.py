import unittest

import numpy as np

from dfckit.inference import paired_nbs


class AuditedNBSRegressionTests(unittest.TestCase):
    def test_adjusted_edge_extent_nbs_matches_frozen_research_output(self):
        rng = np.random.default_rng(20260818)
        edge_i, edge_j = np.triu_indices(4, 1)
        differences = rng.normal(scale=0.5, size=(12, len(edge_i)))
        differences[:, [0, 1]] += 0.45
        differences[:, [4, 5]] -= 0.4
        confounds = rng.normal(size=(12, 2))

        result = paired_nbs(
            differences,
            [f"sub-{index:03d}" for index in range(12)],
            edge_i,
            edge_j,
            4,
            thresholds=(2.0, 2.5),
            n_permutations=12,
            seed=778,
            difference_direction="B minus A",
            confounds=confounds,
            confound_names=("c1", "c2"),
        )
        expected_t = [
            1.801693872657956,
            2.3394362344056483,
            -0.612390425419494,
            1.110654772669218,
            -2.0207375799861405,
            -2.955998083493027,
        ]
        expected_mean = [
            0.33574424353127147,
            0.2586943526010885,
            -0.13781071754850846,
            0.14536367037666853,
            -0.3617708320852986,
            -0.4986254530585023,
        ]
        threshold_two = result.at_threshold(2.0)
        np.testing.assert_array_equal(threshold_two.observed_t, expected_t)
        np.testing.assert_array_equal(
            threshold_two.observed_mean_difference, expected_mean
        )
        np.testing.assert_array_equal(
            threshold_two.null_positive,
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
        )
        np.testing.assert_array_equal(
            threshold_two.null_negative,
            [0, 0, 2, 0, 2, 0, 1, 0, 0, 1, 1, 0],
        )
        self.assertEqual(
            [
                (component.direction, component.node_indices, component.edge_indices)
                for component in threshold_two.components
            ],
            [
                ("positive", (0, 2), (1,)),
                ("negative", (1, 2, 3), (4, 5)),
            ],
        )

        threshold_two_five = result.at_threshold(2.5)
        np.testing.assert_array_equal(
            threshold_two_five.null_maximum,
            [1, 0, 2, 0, 2, 0, 0, 0, 0, 1, 0, 0],
        )
        self.assertEqual(threshold_two_five.positive_components, ())
        self.assertEqual(
            threshold_two_five.negative_components[0].edge_indices,
            (5,),
        )


if __name__ == "__main__":
    unittest.main()
