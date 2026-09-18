"""CAP preprocessing, fitting, and observed state-map contracts."""

import importlib.util
import unittest
from dataclasses import replace

import numpy as np

from dfckit import TimeSeriesDataset, TimeSeriesRun
from dfckit.states import (
    FeatureSequence,
    FeatureSequenceDataset,
    StateAssignments,
    StateLabelSequence,
    cap_sequences,
    cap_state_maps,
    fit_cap_states,
    summarize_state_assignments,
)

HAS_STATES_EXTRA = importlib.util.find_spec("sklearn") is not None
CONTRACT = "cap:within-run-roi-zscore-ddof0"
KEYS = (("visual",), ("motor",))


def _run(values, indices, *, acquisition="run-1"):
    return TimeSeriesRun(
        values=np.asarray(values, dtype=float),
        original_indices=indices,
        roi_names=("visual", "motor"),
        subject="sub-001",
        session="off",
        acquisition_id=acquisition,
        tr=0.8,
    )


def _feature(values, indices, segment, *, contract=CONTRACT, acquisition="run-1"):
    return FeatureSequence(
        values=values,
        sample_start_indices=indices,
        sample_end_indices=indices,
        feature_keys=KEYS,
        subject="sub-001",
        session="off",
        acquisition_id=acquisition,
        segment_id=segment,
        source_contract=contract,
        sample_interval_seconds=0.8,
    )


def _labels(values, indices, segment, *, acquisition="run-1"):
    return StateLabelSequence(
        labels=values,
        sample_start_indices=indices,
        sample_end_indices=indices,
        subject="sub-001",
        session="off",
        acquisition_id=acquisition,
        segment_id=segment,
    )


class CAPPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.run = _run(
            [[0.0, 1.0], [2.0, 2.0], [4.0, 5.0], [9.0, 3.0], [12.0, 8.0]],
            [0, 1, 4, 8, 9],
        )

    def test_default_run_standardizes_all_retained_frames_and_keeps_isolated_frame(self):
        result = cap_sequences(TimeSeriesDataset((self.run,)))
        expected = (self.run.values - self.run.values.mean(axis=0)) / self.run.values.std(
            axis=0, ddof=0
        )
        self.assertEqual(result.source_contract, CONTRACT)
        self.assertEqual([s.n_samples for s in result.sequences], [2, 1, 2])
        self.assertEqual([s.segment_id for s in result.sequences], [0, 1, 2])
        np.testing.assert_array_equal(result.sequences[1].sample_start_indices, [4])
        np.testing.assert_allclose(
            np.concatenate([s.values for s in result.sequences]), expected
        )

    def test_segment_option_preserves_legacy_per_segment_zscore(self):
        result = cap_sequences(TimeSeriesDataset((self.run,)), standardization="segment")
        self.assertEqual(result.source_contract, "cap:within-segment-roi-zscore-ddof0")
        self.assertEqual([s.segment_id for s in result.sequences], [0, 2])
        for sequence in result.sequences:
            np.testing.assert_allclose(sequence.values.mean(axis=0), 0.0, atol=1e-14)
            np.testing.assert_allclose(sequence.values.std(axis=0, ddof=0), 1.0)

    def test_run_requires_two_retained_frames_and_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            cap_sequences(TimeSeriesDataset((_run([[1.0, 2.0]], [4]),)))
        with self.assertRaises(ValueError):
            cap_sequences(TimeSeriesDataset((self.run,)), standardization="whole_dataset")

    def test_scaling_is_per_run_and_constant_rois_become_zero(self):
        first = _run([[1, 5], [2, 5], [3, 5]], [0, 3, 4])
        second = _run([[100, 7], [120, 7], [140, 7]], [0, 1, 4], acquisition="run-2")
        result = cap_sequences(TimeSeriesDataset((first, second)))
        for acquisition in ("run-1", "run-2"):
            rows = np.concatenate([
                s.values for s in result.sequences if s.acquisition_id == acquisition
            ])
            np.testing.assert_allclose(rows.mean(axis=0), 0.0, atol=1e-14)
            self.assertAlmostEqual(rows[:, 0].std(ddof=0), 1.0)
            np.testing.assert_array_equal(rows[:, 1], np.zeros(3))

    def test_censor_gap_does_not_create_state_transition(self):
        features = cap_sequences(TimeSeriesDataset((self.run,)))
        assignments = StateAssignments(
            sequences=(
                _labels([0, 1], [0, 1], 0),
                _labels([0], [4], 1),
                _labels([1, 0], [8, 9], 2),
            ),
            n_states=2,
            source_contract=features.source_contract,
            sample_interval_seconds=0.8,
        )
        metrics = summarize_state_assignments(assignments)[0]
        self.assertEqual(metrics.n_samples, 5)
        self.assertEqual(metrics.n_possible_transitions, 2)
        np.testing.assert_array_equal(metrics.transition_counts, [[0, 1], [1, 0]])


class CAPMapTests(unittest.TestCase):
    def setUp(self):
        self.first = _feature([[1.0, 2.0], [3.0, 4.0]], [0, 1], 0)
        self.second = _feature([[10.0, 6.0], [7.0, 8.0]], [5, 6], 1)
        self.features = FeatureSequenceDataset((self.first, self.second))
        self.assignments = StateAssignments(
            sequences=(_labels([1, 0], [5, 6], 1), _labels([0, 1], [0, 1], 0)),
            n_states=3,
            source_contract=CONTRACT,
            sample_interval_seconds=0.8,
        )

    def test_maps_average_final_labels_across_reordered_sequences_and_are_read_only(self):
        maps = cap_state_maps(self.features, self.assignments)
        self.assertEqual(maps.shape, (3, 2))
        np.testing.assert_allclose(maps[:2], [[4.0, 5.0], [6.5, 5.0]])
        self.assertTrue(np.isnan(maps[2]).all())
        self.assertFalse(maps.flags.writeable)

    def test_maps_reject_non_cap_or_misaligned_input(self):
        non_cap = FeatureSequenceDataset(
            (
                _feature(self.first.values, [0, 1], 0, contract="other"),
                _feature(self.second.values, [5, 6], 1, contract="other"),
            )
        )
        with self.assertRaises(ValueError):
            cap_state_maps(non_cap, self.assignments)
        wrong_indices = StateAssignments(
            sequences=(_labels([0, 1], [0, 2], 0), _labels([1, 0], [5, 6], 1)),
            n_states=2,
            source_contract=CONTRACT,
            sample_interval_seconds=0.8,
        )
        with self.assertRaises(ValueError):
            cap_state_maps(self.features, wrong_indices)
        wrong_acquisition = StateAssignments(
            sequences=(_labels([0, 1], [0, 1], 0, acquisition="run-2"), _labels([1, 0], [5, 6], 1)),
            n_states=2,
            source_contract=CONTRACT,
            sample_interval_seconds=0.8,
        )
        with self.assertRaises(ValueError):
            cap_state_maps(self.features, wrong_acquisition)

    def test_maps_reject_different_standardization_sampling_or_sequence_set(self):
        for assignments in (
            replace(self.assignments, source_contract="cap:within-segment-roi-zscore-ddof0"),
            replace(self.assignments, sample_interval_seconds=1.0),
            replace(self.assignments, sequences=self.assignments.sequences[:1]),
        ):
            with self.subTest(assignments=assignments), self.assertRaises(ValueError):
                cap_state_maps(self.features, assignments)


@unittest.skipUnless(HAS_STATES_EXTRA, "requires dfc-kit[states]")
class CAPFitTests(unittest.TestCase):
    def setUp(self):
        self.dataset = TimeSeriesDataset(
            (
                _run([[-2, -1], [-1, -2], [1, 2], [2, 1]], [0, 1, 2, 3]),
                _run([[-3, -2], [-1, -1], [2, 1], [3, 2]], [0, 1, 2, 3], acquisition="run-2"),
            )
        )

    def test_default_lloyd_matches_sklearn_on_run_standardized_frames(self):
        from sklearn.cluster import KMeans

        features = cap_sequences(self.dataset)
        pooled = np.concatenate([s.values for s in features.sequences])
        direct = KMeans(
            n_clusters=2, n_init=5, max_iter=100, random_state=17, algorithm="lloyd"
        ).fit(pooled)
        fit = fit_cap_states(self.dataset, n_states=2, seed=17, n_init=5, max_iter=100)
        self.assertEqual(fit.model.algorithm, "lloyd")
        self.assertEqual(fit.model.source_contract, CONTRACT)
        self.assertFalse(fit.model.standardize_features)
        self.assertIsNone(fit.model.n_pca_components)
        np.testing.assert_allclose(fit.model.feature_mean, 0.0)
        np.testing.assert_allclose(fit.model.feature_scale, 1.0)
        np.testing.assert_allclose(fit.model.centers, direct.cluster_centers_)
        np.testing.assert_array_equal(
            np.concatenate([s.labels for s in fit.assignments.sequences]), direct.labels_
        )

    def test_minibatch_maps_use_final_assignments_without_mutating_model_centers(self):
        features = cap_sequences(self.dataset)
        fit = fit_cap_states(
            self.dataset, n_states=2, seed=17, n_init=5, max_iter=100,
            algorithm="minibatch", batch_size=3, reassignment_ratio=0.0,
        )
        original_centers = fit.model.centers.copy()
        maps = cap_state_maps(features, fit.assignments)
        rows = np.concatenate([s.values for s in features.sequences])
        labels = np.concatenate([s.labels for s in fit.assignments.sequences])
        for state in range(2):
            np.testing.assert_allclose(maps[state], rows[labels == state].mean(axis=0))
        np.testing.assert_array_equal(fit.model.centers, original_centers)


if __name__ == "__main__":
    unittest.main()
