import unittest

import numpy as np

from dfckit.states import StateAssignments, StateLabelSequence, summarize_state_stability


def _assignments(first: list[int], second: list[int]) -> StateAssignments:
    return StateAssignments(
        sequences=(
            StateLabelSequence(
                labels=first,
                sample_start_indices=np.arange(len(first)),
                sample_end_indices=np.arange(len(first)),
                subject="sub-001",
                session="off",
                acquisition_id="sub-001_ses-off_task-rest_run-1",
                segment_id=0,
            ),
            StateLabelSequence(
                labels=second,
                sample_start_indices=np.arange(10, 10 + len(second)),
                sample_end_indices=np.arange(10, 10 + len(second)),
                subject="sub-001",
                session="off",
                acquisition_id="sub-001_ses-off_task-rest_run-1",
                segment_id=1,
            ),
        ),
        n_states=2,
        source_contract="stability-test:v1",
        sample_interval_seconds=0.8,
    )


class StateStabilityTests(unittest.TestCase):
    def test_metrics_are_stacked_in_fit_order_without_crossing_gaps(self):
        first = _assignments([0, 0, 1, 1], [1, 1])
        second = _assignments([0, 1, 1, 1], [0, 0])
        stability = summarize_state_stability((first, second))[0]
        self.assertEqual(stability.n_fits, 2)
        self.assertEqual(stability.n_states, 2)
        self.assertEqual(stability.n_sequences, 2)
        np.testing.assert_allclose(
            stability.occupancy_by_fit,
            [[2 / 6, 4 / 6], [3 / 6, 3 / 6]],
        )
        np.testing.assert_allclose(
            stability.transition_probabilities_by_fit[0],
            [[0.5, 0.5], [0.0, 1.0]],
        )
        self.assertFalse(stability.occupancy_by_fit.flags.writeable)

    def test_mismatched_runs_and_sampling_boundaries_are_rejected(self):
        reference = _assignments([0, 0, 1, 1], [1, 1])
        different_length = _assignments([0, 1, 1], [0, 0])
        with self.assertRaisesRegex(ValueError, "sampling boundaries"):
            summarize_state_stability((reference, different_length))

        wrong_identity = StateAssignments(
            sequences=(
                StateLabelSequence(
                    labels=[0, 1, 1, 1],
                    sample_start_indices=[0, 1, 2, 3],
                    sample_end_indices=[0, 1, 2, 3],
                    subject="sub-002",
                    session="off",
                    acquisition_id="sub-002_ses-off_task-rest_run-1",
                    segment_id=0,
                ),
            ),
            n_states=2,
            source_contract="stability-test:v1",
            sample_interval_seconds=0.8,
        )
        with self.assertRaisesRegex(ValueError, "run identities"):
            summarize_state_stability((reference, wrong_identity))

    def test_exact_sample_indices_are_required(self):
        reference = _assignments([0, 0, 1, 1], [1, 1])
        shifted = StateAssignments(
            sequences=(
                StateLabelSequence(
                    labels=[0, 1, 1, 1],
                    sample_start_indices=[1, 2, 3, 4],
                    sample_end_indices=[1, 2, 3, 4],
                    subject="sub-001",
                    session="off",
                    acquisition_id="sub-001_ses-off_task-rest_run-1",
                    segment_id=0,
                ),
                reference.sequences[1],
            ),
            n_states=2,
            source_contract="stability-test:v1",
            sample_interval_seconds=0.8,
        )
        with self.assertRaisesRegex(ValueError, "sequence sampling boundaries"):
            summarize_state_stability((reference, shifted))


if __name__ == "__main__":
    unittest.main()
