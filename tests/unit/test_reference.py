import unittest

import numpy as np

from dfckit.connectivity import mean_projector_basis, subspace_distance, subspace_similarity
from dfckit.reference import (
    fit_feature_reference,
    fit_subspace_reference,
    leave_one_subject_out_feature_similarity,
    pearson_pattern_similarity,
    reference_subspace_similarity,
    score_feature_reference,
    score_subspace_reference,
    subject_balanced_quantiles,
)


class FeatureReferenceTests(unittest.TestCase):
    def test_reference_gives_each_subject_equal_weight_despite_repeated_observations(self):
        values = np.r_[np.zeros((100, 1)), np.asarray([[10.0]])]
        subjects = ("sub-a",) * 100 + ("sub-b",)

        model = fit_feature_reference(values, subjects, (("metric",),))

        self.assertAlmostEqual(model.template[0], 5.0)
        np.testing.assert_array_equal(model.observation_counts, [100, 1])
        self.assertEqual(model.fit_subjects, ("sub-a", "sub-b"))

    def test_pattern_similarity_matches_rowwise_pearson_correlation(self):
        values = np.asarray([[1.0, 2.0, 4.0], [4.0, 1.0, 0.0]])
        template = np.asarray([0.0, 1.0, 3.0])

        observed = pearson_pattern_similarity(values, template)
        expected = [np.corrcoef(row, template)[0, 1] for row in values]

        np.testing.assert_allclose(observed, expected)

    def test_leave_one_subject_out_similarity_uses_excluding_template(self):
        values = np.asarray(
            [
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        subjects = ("sub-a", "sub-a", "sub-b", "sub-c")
        keys = (("x",), ("y",), ("z",))
        model = fit_feature_reference(values, subjects, keys)

        scores = leave_one_subject_out_feature_similarity(model, values, subjects, keys)
        expected_a = pearson_pattern_similarity(values[:2], np.asarray([0.0, 1.0, 0.5]))
        expected_b = pearson_pattern_similarity(
            values[2:3], np.asarray([1.5, 0.0, 0.5])
        )
        expected_c = pearson_pattern_similarity(
            values[3:], np.asarray([1.5, 1.0, 0.0])
        )

        np.testing.assert_allclose(scores, np.r_[expected_a, expected_b, expected_c])

    def test_external_scoring_rejects_fit_subject_overlap_and_feature_reordering(self):
        values = np.asarray([[2.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
        keys = (("a",), ("b",), ("c",))
        model = fit_feature_reference(values, ("sub-a", "sub-b"), keys)
        test = np.asarray([[1.0, 0.0, 2.0]])

        with self.assertRaisesRegex(ValueError, "overlap"):
            score_feature_reference(model, test, ("sub-a",), keys)
        with self.assertRaisesRegex(ValueError, "feature identities"):
            score_feature_reference(
                model,
                test,
                ("sub-c",),
                tuple(reversed(keys)),
            )
        score = score_feature_reference(model, test, ("sub-c",), keys)
        self.assertEqual(score.shape, (1,))

    def test_subject_balanced_quantile_is_not_dominated_by_many_windows(self):
        values = np.asarray([0.0] * 100 + [10.0])
        subjects = ("sub-a",) * 100 + ("sub-b",)

        median = subject_balanced_quantiles(values, subjects, [0.5])[0]

        self.assertGreater(median, 0.0)
        self.assertLess(median, 10.0)


class SubspaceReferenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(61)
        self.a, _ = np.linalg.qr(rng.normal(size=(7, 2)))
        self.b, _ = np.linalg.qr(rng.normal(size=(7, 2)))
        self.c, _ = np.linalg.qr(rng.normal(size=(7, 2)))
        self.d, _ = np.linalg.qr(rng.normal(size=(7, 2)))
        rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)))
        self.bases = (self.a, self.a @ rotation, self.b, self.c)
        self.subjects = ("sub-a", "sub-a", "sub-b", "sub-c")
        self.roi_names = tuple(f"roi-{index}" for index in range(7))

    def test_fit_uses_session_within_subject_then_between_subject_projector_means(self):
        model = fit_subspace_reference(self.bases, self.subjects, self.roi_names)
        participant_a = mean_projector_basis(self.bases[:2], rank=2)
        expected = mean_projector_basis([participant_a, self.b, self.c], rank=2)

        self.assertEqual(model.fit_subjects, ("sub-a", "sub-b", "sub-c"))
        np.testing.assert_array_equal(model.observation_counts, [2, 1, 1])
        self.assertAlmostEqual(subspace_similarity(model.basis, expected), 1.0)
        self.assertEqual(model.rank, 2)

    def test_loo_distances_exclude_each_participant(self):
        model = fit_subspace_reference(self.bases, self.subjects, self.roi_names)
        expected = []
        for index, basis in enumerate(model.subject_bases):
            template = mean_projector_basis(
                [other for position, other in enumerate(model.subject_bases) if position != index],
                rank=2,
            )
            expected.append(subspace_distance(basis, template))

        np.testing.assert_allclose(model.loo_distances, expected)
        self.assertAlmostEqual(model.loo_distance_mean, np.mean(expected))
        self.assertAlmostEqual(model.loo_distance_scale, np.std(expected, ddof=1))

    def test_external_scores_use_fixed_reference_and_loo_standardization(self):
        model = fit_subspace_reference(self.bases, self.subjects, self.roi_names)

        result = score_subspace_reference(
            model,
            [self.d],
            ["sub-pd"],
            self.roi_names,
        )
        expected_distance = subspace_distance(self.d, model.basis)
        expected_z = (
            expected_distance - model.loo_distance_mean
        ) / model.loo_distance_scale

        self.assertAlmostEqual(result.distances[0], expected_distance)
        self.assertAlmostEqual(result.standardized_distances[0], expected_z)
        self.assertEqual(result.reference_fit_subjects, model.fit_subjects)

    def test_overlap_roi_order_and_rank_mismatch_are_rejected(self):
        model = fit_subspace_reference(self.bases, self.subjects, self.roi_names)
        with self.assertRaisesRegex(ValueError, "overlap"):
            score_subspace_reference(model, [self.d], ["sub-a"], self.roi_names)
        with self.assertRaisesRegex(ValueError, "ROI identities"):
            score_subspace_reference(
                model,
                [self.d],
                ["sub-pd"],
                tuple(reversed(self.roi_names)),
            )
        with self.assertRaisesRegex(ValueError, "ROI or rank"):
            score_subspace_reference(
                model,
                [self.d[:, :1]],
                ["sub-pd"],
                self.roi_names,
            )

    def test_single_basis_similarity_validates_roi_identity(self):
        model = fit_subspace_reference(self.bases, self.subjects, self.roi_names)

        self.assertAlmostEqual(
            reference_subspace_similarity(model, model.basis, self.roi_names),
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "ROI identities"):
            reference_subspace_similarity(
                model,
                model.basis,
                tuple(reversed(self.roi_names)),
            )


if __name__ == "__main__":
    unittest.main()
