import unittest

import numpy as np

from dfckit.inference import paired_nbs


class AuditedNBSRegressionTests(unittest.TestCase):
    def test_one_sided_results_match_bctpy_0_6_1_reference(self):
        # BCTPy's paired nbs_bct was driven with the identical sign sequence
        # generated below. Its hit/P values were 34/2048 and 25/2048; dfc-kit
        # applies the plus-one correction to the same maximum-component nulls.
        n_subjects = 24
        edge_i, edge_j = np.triu_indices(6, 1)
        rng = np.random.default_rng(8417)
        differences = rng.normal(scale=0.55, size=(n_subjects, len(edge_i)))
        differences[:, [0, 1, 5]] += 0.42
        differences[:, [8, 12, 14]] -= 0.45
        arguments = {
            "differences": differences,
            "subject_ids": [f"sub-{index:03d}" for index in range(n_subjects)],
            "edge_i": edge_i,
            "edge_j": edge_j,
            "n_nodes": 6,
            "thresholds": (2.0,),
            "n_permutations": 2048,
            "seed": 44019,
            "difference_direction": "X minus Y",
        }

        positive = paired_nbs(**arguments, alternative="greater").at_threshold(2.0)
        negative = paired_nbs(**arguments, alternative="less").at_threshold(2.0)

        self.assertEqual(
            tuple(component.edge_indices for component in positive.positive_components),
            ((0, 1, 5),),
        )
        self.assertEqual(
            tuple(component.edge_indices for component in negative.negative_components),
            ((8, 12, 14),),
        )
        self.assertEqual(np.count_nonzero(positive.null_positive >= 3), 34)
        self.assertEqual(np.count_nonzero(negative.null_negative >= 3), 25)
        self.assertEqual(positive.positive_components[0].fwe_pvalue, 35 / 2049)
        self.assertEqual(negative.negative_components[0].fwe_pvalue, 26 / 2049)

    def test_pooled_two_sided_result_matches_bctpy_0_6_1_reference(self):
        # BCTPy's native tail="both" was supplied the exact sign sequence
        # generated below. The observed component and all 256 null maxima
        # matched; dfc-kit alone applies the plus-one p-value correction.
        n_subjects = 24
        edge_i, edge_j = np.triu_indices(6, 1)
        rng = np.random.default_rng(80421)
        differences = rng.normal(scale=0.5, size=(n_subjects, len(edge_i)))
        differences[:, [0, 1, 5]] += 0.38
        differences[:, [8, 12, 14]] -= 0.42

        pooled = paired_nbs(
            differences,
            [f"sub-{index:03d}" for index in range(n_subjects)],
            edge_i,
            edge_j,
            6,
            thresholds=(2.0,),
            n_permutations=256,
            seed=9917,
            difference_direction="X minus Y",
            component_sign_mode="pooled",
        ).at_threshold(2.0)

        self.assertEqual(
            tuple(component.edge_indices for component in pooled.pooled_components),
            ((0, 1, 5, 6, 8, 12, 14),),
        )
        np.testing.assert_array_equal(
            np.unique(pooled.null_pooled, return_counts=True)[1],
            [101, 123, 24, 5, 2, 1],
        )
        self.assertEqual(pooled.pooled_components[0].fwe_pvalue, 1 / 257)

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
