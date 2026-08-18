import importlib.util
import unittest
import warnings
from dataclasses import replace

import numpy as np

from dfckit import TimeSeriesDataset, TimeSeriesRun
from dfckit.connectivity import SlidingWindowFC
from dfckit.states import (
    FeatureSequence,
    FeatureSequenceDataset,
    GaussianHMMStateModel,
    GaussianHMMStateResult,
    KMeansStateModel,
    StateAssignments,
    StateLabelSequence,
    align_gaussian_hmm_emissions,
    align_kmeans_centroids,
    apply_gaussian_hmm_alignment,
    apply_state_alignment,
    cap_sequences,
    fit_cap_states,
    fit_kmeans_states,
    predict_gaussian_hmm_states,
    predict_kmeans_states,
    relabel_gaussian_hmm_model,
    relabel_kmeans_model,
    summarize_state_assignments,
    window_fc_sequences,
)

HAS_STATES_EXTRA = (
    importlib.util.find_spec("sklearn") is not None
    and importlib.util.find_spec("scipy") is not None
)
HAS_HMM_EXTRA = HAS_STATES_EXTRA and importlib.util.find_spec("hmmlearn") is not None


def feature_sequence(subject, values, segment_id=0):
    values = np.asarray(values, dtype=float)
    indices = np.arange(len(values), dtype=np.int64)
    return FeatureSequence(
        values=values,
        sample_start_indices=indices,
        sample_end_indices=indices,
        feature_keys=(("visual",), ("motor",)),
        subject=subject,
        session="off",
        segment_id=segment_id,
        source_contract="synthetic-two-feature",
        sample_interval_seconds=0.8,
    )


class FeatureSequenceTests(unittest.TestCase):
    def test_window_results_are_split_at_censor_segments(self):
        rng = np.random.default_rng(4)
        run = TimeSeriesRun(
            values=rng.normal(size=(12, 4)),
            original_indices=[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14],
            roi_names=("a", "b", "c", "d"),
            subject="sub-001",
            session="off",
            tr=0.8,
        )
        windows = SlidingWindowFC(length=4, step=2).transform(run)
        dataset = window_fc_sequences([windows])

        self.assertEqual(len(dataset.sequences), 2)
        self.assertEqual([sequence.n_samples for sequence in dataset.sequences], [1, 2])
        self.assertEqual(dataset.sample_interval_seconds, 1.6)
        self.assertEqual(dataset.source_contract, "window-fc:length=4;step=2;taper=hamming")

    def test_cap_standardizes_each_retained_segment_separately(self):
        run = TimeSeriesRun(
            values=np.asarray(
                [
                    [0.0, 1.0],
                    [1.0, 3.0],
                    [3.0, 2.0],
                    [100.0, -5.0],
                    [104.0, 1.0],
                    [109.0, 4.0],
                ]
            ),
            original_indices=[0, 1, 2, 7, 8, 9],
            roi_names=("visual", "motor"),
            subject="sub-001",
            session="off",
            tr=0.8,
        )
        sequences = cap_sequences(TimeSeriesDataset([run]))

        self.assertEqual(len(sequences.sequences), 2)
        for sequence in sequences.sequences:
            np.testing.assert_allclose(sequence.values.mean(axis=0), 0.0, atol=1e-14)
            np.testing.assert_allclose(sequence.values.std(axis=0), 1.0)
        np.testing.assert_array_equal(sequences.sequences[1].sample_start_indices, [7, 8, 9])

    def test_feature_contract_mismatch_is_rejected(self):
        first = feature_sequence("sub-001", [[0.0, 1.0], [1.0, 2.0]])
        second = FeatureSequence(
            values=[[0.0, 1.0]],
            sample_start_indices=[0],
            sample_end_indices=[0],
            feature_keys=(("motor",), ("visual",)),
            subject="sub-002",
            session="off",
            segment_id=0,
            source_contract="synthetic-two-feature",
            sample_interval_seconds=0.8,
        )
        with self.assertRaisesRegex(ValueError, "feature identity"):
            FeatureSequenceDataset([first, second])


@unittest.skipUnless(HAS_STATES_EXTRA, "requires dfc-kit[states]")
class KMeansStateTests(unittest.TestCase):
    def setUp(self):
        self.dataset = FeatureSequenceDataset(
            [
                feature_sequence(
                    f"sub-{index:03d}",
                    [[-2.1, -1.9], [-1.8, -2.2], [2.0, 2.1], [2.2, 1.8]],
                )
                for index in range(1, 5)
            ]
        )

    def test_kmeans_records_fit_subjects_and_is_reproducible(self):
        first = fit_kmeans_states(self.dataset, n_states=2, seed=9, n_init=10)
        second = fit_kmeans_states(self.dataset, n_states=2, seed=9, n_init=10)

        np.testing.assert_allclose(first.model.centers, second.model.centers)
        self.assertEqual(first.model.fit_subjects, self.dataset.subjects)
        self.assertEqual(first.model.seed, 9)
        self.assertEqual(first.model.algorithm, "lloyd")
        for sequence in first.assignments.sequences:
            self.assertEqual(sequence.labels[0], sequence.labels[1])
            self.assertEqual(sequence.labels[2], sequence.labels[3])
            self.assertNotEqual(sequence.labels[0], sequence.labels[2])

    def test_prediction_rejects_fit_subject_overlap_by_default(self):
        model = fit_kmeans_states(self.dataset, n_states=2, seed=9, n_init=5).model
        with self.assertRaisesRegex(ValueError, "overlap"):
            predict_kmeans_states(model, self.dataset)

        unseen = FeatureSequenceDataset(
            [feature_sequence("sub-099", [[-2.0, -2.0], [2.0, 2.0]])]
        )
        predicted = predict_kmeans_states(model, unseen)
        self.assertNotEqual(predicted.sequences[0].labels[0], predicted.sequences[0].labels[1])

    def test_cap_uses_minibatch_without_second_global_standardization(self):
        runs = []
        for subject, shift in (("sub-001", 0.0), ("sub-002", 0.2)):
            runs.append(
                TimeSeriesRun(
                    values=np.asarray(
                        [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]]
                    )
                    + shift,
                    original_indices=np.arange(4),
                    roi_names=("visual", "motor"),
                    subject=subject,
                    session="off",
                    tr=0.8,
                )
            )
        fit = fit_cap_states(
            TimeSeriesDataset(runs), n_states=2, seed=11, n_init=5, max_iter=50
        )

        self.assertEqual(fit.model.algorithm, "minibatch")
        self.assertFalse(fit.model.standardize_features)
        self.assertEqual(fit.model.batch_size, 4096)
        self.assertEqual(fit.model.reassignment_ratio, 0.01)
        np.testing.assert_array_equal(fit.model.feature_mean, [0.0, 0.0])
        np.testing.assert_array_equal(fit.model.feature_scale, [1.0, 1.0])

    def test_degenerate_requested_state_count_is_rejected(self):
        from sklearn.exceptions import ConvergenceWarning

        degenerate = FeatureSequenceDataset(
            [feature_sequence("sub-001", [[1.0, 1.0], [1.0, 1.0]])]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            with self.assertRaisesRegex(RuntimeError, "empty state"):
                fit_kmeans_states(degenerate, n_states=2, seed=3, n_init=2)


class StateMetricTests(unittest.TestCase):
    def test_metrics_never_bridge_sequence_boundaries(self):
        assignments = StateAssignments(
            sequences=(
                StateLabelSequence(
                    labels=[0, 0, 1],
                    sample_start_indices=[0, 1, 2],
                    sample_end_indices=[0, 1, 2],
                    subject="sub-001",
                    session="off",
                    segment_id=0,
                ),
                StateLabelSequence(
                    labels=[1, 0],
                    sample_start_indices=[8, 9],
                    sample_end_indices=[8, 9],
                    subject="sub-001",
                    session="off",
                    segment_id=1,
                ),
            ),
            n_states=2,
            source_contract="manual",
            sample_interval_seconds=0.8,
        )
        metrics = summarize_state_assignments(assignments)[0]

        np.testing.assert_allclose(metrics.occupancy, [0.6, 0.4])
        np.testing.assert_allclose(metrics.mean_dwell_samples, [1.5, 1.0])
        np.testing.assert_allclose(metrics.mean_dwell_seconds, [1.2, 0.8])
        np.testing.assert_array_equal(metrics.transition_counts, [[1, 1], [1, 0]])
        self.assertEqual(metrics.n_possible_transitions, 3)
        self.assertEqual(metrics.n_switches, 2)
        self.assertAlmostEqual(metrics.switch_rate, 2.0 / 3.0)

    def test_metrics_keep_acquisitions_separate(self):
        assignments = StateAssignments(
            sequences=(
                StateLabelSequence(
                    labels=[0, 1],
                    sample_start_indices=[0, 1],
                    sample_end_indices=[0, 1],
                    subject="sub-001",
                    session="off",
                    acquisition_id="run-1",
                    segment_id=0,
                ),
                StateLabelSequence(
                    labels=[1, 1],
                    sample_start_indices=[0, 1],
                    sample_end_indices=[0, 1],
                    subject="sub-001",
                    session="off",
                    acquisition_id="run-2",
                    segment_id=0,
                ),
            ),
            n_states=2,
            source_contract="manual",
            sample_interval_seconds=0.8,
        )
        metrics = summarize_state_assignments(assignments)
        self.assertEqual([(item.acquisition_id, item.n_samples) for item in metrics], [("run-1", 2), ("run-2", 2)])
        self.assertEqual(metrics[0].n_switches, 1)
        self.assertEqual(metrics[1].n_switches, 0)


def state_model(centers, seed):
    centers = np.asarray(centers, dtype=float)
    return KMeansStateModel(
        centers=centers,
        standardized_centers=centers,
        feature_mean=np.zeros(centers.shape[1]),
        feature_scale=np.ones(centers.shape[1]),
        feature_keys=(("a",), ("b",), ("c",)),
        source_contract="alignment-test",
        sample_interval_seconds=1.0,
        n_states=2,
        seed=seed,
        n_init=1,
        max_iter=10,
        algorithm="lloyd",
        standardize_features=False,
        batch_size=None,
        reassignment_ratio=None,
        init_sample_size=None,
        iterations=1,
        inertia=0.0,
        fit_subjects=(f"fit-{seed}",),
        fit_sample_count=4,
        training_data_fingerprint=None,
        implementation="test",
    )


def hmm_state_model(seed, order=(0, 1), *, compact=False):
    order = np.asarray(order, dtype=np.int64)
    start = np.asarray([0.7, 0.3])
    transition = np.asarray([[0.85, 0.15], [0.2, 0.8]])
    means = np.asarray([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
    covariances = np.asarray([np.eye(3) * 0.4, np.eye(3) * 0.8])
    return GaussianHMMStateModel(
        start_probabilities=start[order],
        transition_matrix=transition[np.ix_(order, order)],
        reduced_means=means[order],
        reduced_covariances=covariances[order],
        emission_means=means[order],
        emission_covariances=None if compact else covariances[order],
        feature_mean=np.zeros(3),
        feature_scale=np.ones(3),
        pca_mean=np.zeros(3),
        pca_components=np.eye(3),
        pca_explained_variance_ratio=np.asarray([0.5, 0.3, 0.2]),
        feature_keys=(("a",), ("b",), ("c",)),
        source_contract="alignment-test",
        sample_interval_seconds=1.0,
        n_states=2,
        n_pca_components=3,
        covariance_type="full",
        seed=seed,
        n_init=1,
        n_iter=20,
        tol=1e-3,
        minimum_sequence_length=2,
        pca_batch_size=None,
        selected_initialization=0,
        initialization_seeds=(seed,),
        initialization_log_likelihoods=np.asarray([-10.0]),
        iterations=5,
        converged=True,
        log_likelihood=-10.0,
        fit_subjects=(f"fit-{seed}",),
        fit_sample_count=20,
        fit_sequence_count=2,
        omitted_short_sequence_count=0,
        training_data_fingerprint=None,
        implementation="test",
    )


@unittest.skipUnless(HAS_STATES_EXTRA, "requires dfc-kit[states]")
class StateAlignmentTests(unittest.TestCase):
    def test_hungarian_alignment_recovers_swapped_states(self):
        reference = state_model([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]], seed=1)
        candidate = state_model([[-1.1, 0.0, 1.0], [0.9, 0.1, -1.0]], seed=2)
        alignment = align_kmeans_centroids(reference, candidate)

        np.testing.assert_array_equal(alignment.candidate_to_reference, [1, 0])
        self.assertTrue(np.all(alignment.matched_correlations > 0.99))

        assignments = StateAssignments(
            sequences=(
                StateLabelSequence(
                    labels=[0, 1, 1],
                    sample_start_indices=[0, 1, 2],
                    sample_end_indices=[0, 1, 2],
                    subject="sub-099",
                    session="off",
                    segment_id=0,
                ),
            ),
            n_states=2,
            source_contract="alignment-test",
            sample_interval_seconds=1.0,
        )
        aligned = apply_state_alignment(assignments, alignment)
        np.testing.assert_array_equal(aligned.sequences[0].labels, [1, 0, 0])

    def test_relabel_kmeans_model_reorders_centers_for_future_predictions(self):
        reference = state_model([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]], seed=1)
        candidate = state_model([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], seed=2)
        alignment = align_kmeans_centroids(reference, candidate)
        relabeled = relabel_kmeans_model(candidate, alignment)

        np.testing.assert_array_equal(relabeled.centers, reference.centers)
        np.testing.assert_array_equal(
            relabeled.standardized_centers,
            reference.standardized_centers,
        )
        self.assertEqual(relabeled.seed, candidate.seed)
        self.assertFalse(relabeled.centers.flags.writeable)

    def test_hmm_alignment_reorders_every_state_axis_and_posterior_column(self):
        reference = hmm_state_model(seed=3)
        candidate = hmm_state_model(seed=4, order=(1, 0))
        alignment = align_gaussian_hmm_emissions(reference, candidate)
        np.testing.assert_array_equal(alignment.candidate_to_reference, [1, 0])
        self.assertTrue(np.all(alignment.matched_correlations > 0.999))

        relabeled = relabel_gaussian_hmm_model(candidate, alignment)
        for name in (
            "start_probabilities",
            "transition_matrix",
            "reduced_means",
            "reduced_covariances",
            "emission_means",
            "emission_covariances",
        ):
            np.testing.assert_array_equal(getattr(relabeled, name), getattr(reference, name))
        self.assertEqual(relabeled.seed, candidate.seed)
        self.assertFalse(relabeled.transition_matrix.flags.writeable)

        assignments = StateAssignments(
            sequences=(
                StateLabelSequence(
                    labels=[0, 1, 1],
                    sample_start_indices=[0, 1, 2],
                    sample_end_indices=[0, 1, 2],
                    subject="sub-099",
                    session="off",
                    segment_id=0,
                ),
            ),
            n_states=2,
            source_contract="alignment-test",
            sample_interval_seconds=1.0,
        )
        posterior = np.asarray([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7]])
        result = GaussianHMMStateResult(
            assignments=assignments,
            posterior_probabilities=(posterior,),
            log_likelihood=-5.0,
        )
        aligned = apply_gaussian_hmm_alignment(result, alignment)
        np.testing.assert_array_equal(aligned.assignments.sequences[0].labels, [1, 0, 0])
        np.testing.assert_array_equal(aligned.posterior_probabilities[0], posterior[:, [1, 0]])
        self.assertFalse(aligned.posterior_probabilities[0].flags.writeable)
        self.assertEqual(aligned.log_likelihood, result.log_likelihood)

    def test_compact_hmm_alignment_keeps_covariance_unmaterialized(self):
        reference = hmm_state_model(seed=5, compact=True)
        candidate = hmm_state_model(seed=6, order=(1, 0), compact=True)
        alignment = align_gaussian_hmm_emissions(reference, candidate)
        relabeled = relabel_gaussian_hmm_model(candidate, alignment)
        self.assertIsNone(relabeled.emission_covariances)

    def test_model_relabel_rejects_an_alignment_for_another_candidate_seed(self):
        reference = hmm_state_model(seed=7)
        candidate = hmm_state_model(seed=8, order=(1, 0))
        alignment = align_gaussian_hmm_emissions(reference, candidate)
        with self.assertRaisesRegex(ValueError, "candidate seed"):
            relabel_gaussian_hmm_model(hmm_state_model(seed=9), alignment)

    def test_model_relabel_rejects_same_seed_in_a_different_feature_space(self):
        reference = hmm_state_model(seed=16)
        candidate = hmm_state_model(seed=17, order=(1, 0))
        alignment = align_gaussian_hmm_emissions(reference, candidate)
        different_features = replace(
            candidate,
            feature_keys=(("x",), ("y",), ("z",)),
        )
        with self.assertRaisesRegex(ValueError, "feature identities"):
            relabel_gaussian_hmm_model(different_features, alignment)

    def test_hmm_alignment_rejects_different_sampling_intervals(self):
        reference = hmm_state_model(seed=12)
        candidate = replace(
            hmm_state_model(seed=13, order=(1, 0)),
            sample_interval_seconds=2.0,
        )
        with self.assertRaisesRegex(ValueError, "sample intervals"):
            align_gaussian_hmm_emissions(reference, candidate)

    def test_hmm_alignment_rejects_invalid_posterior_probabilities(self):
        reference = hmm_state_model(seed=14)
        candidate = hmm_state_model(seed=15, order=(1, 0))
        alignment = align_gaussian_hmm_emissions(reference, candidate)
        assignments = StateAssignments(
            sequences=(
                StateLabelSequence(
                    labels=[0, 1],
                    sample_start_indices=[0, 1],
                    sample_end_indices=[0, 1],
                    subject="sub-099",
                    session="off",
                    segment_id=0,
                ),
            ),
            n_states=2,
            source_contract="alignment-test",
            sample_interval_seconds=1.0,
        )
        invalid = GaussianHMMStateResult(
            assignments=assignments,
            posterior_probabilities=(np.asarray([[0.8, 0.8], [0.1, 0.9]]),),
            log_likelihood=-5.0,
        )
        with self.assertRaisesRegex(ValueError, "posterior probabilities"):
            apply_gaussian_hmm_alignment(invalid, alignment)

    @unittest.skipUnless(HAS_HMM_EXTRA, "requires dfc-kit[hmm]")
    def test_relabel_then_decode_matches_decode_then_align(self):
        reference = hmm_state_model(seed=10)
        candidate = hmm_state_model(seed=11, order=(1, 0))
        alignment = align_gaussian_hmm_emissions(reference, candidate)
        relabeled = relabel_gaussian_hmm_model(candidate, alignment)
        values = np.asarray(
            [[-1.1, 0.0, 1.0], [-0.9, 0.1, 0.8], [1.0, 0.0, -1.1]]
        )
        sequence = FeatureSequence(
            values=values,
            sample_start_indices=np.arange(len(values)),
            sample_end_indices=np.arange(len(values)),
            feature_keys=(("a",), ("b",), ("c",)),
            subject="sub-heldout",
            session="off",
            segment_id=0,
            source_contract="alignment-test",
            sample_interval_seconds=1.0,
        )
        dataset = FeatureSequenceDataset((sequence,))

        decoded_candidate = predict_gaussian_hmm_states(candidate, dataset)
        expected = apply_gaussian_hmm_alignment(decoded_candidate, alignment)
        observed = predict_gaussian_hmm_states(relabeled, dataset)
        np.testing.assert_array_equal(
            observed.assignments.sequences[0].labels,
            expected.assignments.sequences[0].labels,
        )
        np.testing.assert_allclose(
            observed.posterior_probabilities[0],
            expected.posterior_probabilities[0],
        )
        self.assertAlmostEqual(observed.log_likelihood, expected.log_likelihood)


if __name__ == "__main__":
    unittest.main()
