import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.artifacts import state_stability_payload, write_state_stability
from dfckit.states import RunStateStability, StateAlignment


def _run() -> RunStateStability:
    return RunStateStability(
        subject="sub-010",
        session="off",
        acquisition_id="sub-010_run-1",
        n_samples=6,
        n_sequences=2,
        occupancy_by_fit=[[0.5, 0.5], [1 / 3, 2 / 3]],
        mean_dwell_samples_by_fit=[[1.5, 3.0], [2.0, 2.0]],
        mean_dwell_seconds_by_fit=[[1.2, 2.4], [1.6, 1.6]],
        switch_rate_by_fit=[0.5, np.nan],
        transition_probabilities_by_fit=[
            [[0.5, 0.5], [0.0, 1.0]],
            [[1.0, 0.0], [np.nan, np.nan]],
        ],
    )


def _alignment() -> StateAlignment:
    return StateAlignment(
        candidate_to_reference=[1, 0],
        matched_costs=[0.09, 0.13],
        cost_matrix=[[1.2, 0.09], [0.13, 1.1]],
        reference_seed=17,
        candidate_seed=29,
        feature_keys=(("visual", "motor"), ("visual", "putamen")),
        source_contract="stability-io-test:v1",
        sample_interval_seconds=0.8,
    )


def _metadata() -> dict[str, object]:
    return {
        "model_kind": "kmeans-state",
        "reference_seed": 17,
        "candidate_seeds": (29,),
        "alignments": (_alignment(),),
        "source_contract": "stability-io-test:v1",
        "sample_interval_seconds": 0.8,
        "allow_fit_subjects": False,
    }


class StateStabilityIOTests(unittest.TestCase):
    def test_payload_records_alignment_and_finite_statistics(self):
        payload = state_stability_payload((_run(),), **_metadata())
        self.assertEqual(payload["fits"][0]["candidate_to_reference"], [0, 1])
        self.assertEqual(payload["fits"][1]["candidate_to_reference"], [1, 0])
        self.assertIn("ddof=0", payload["dispersion_standard_deviation"])
        self.assertIsNone(payload["runs"][0]["switch_rate"]["standard_deviation"])

    def test_writer_is_strict_and_refuses_overwrite(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "stability.json"
            write_state_stability((_run(),), output, **_metadata())
            self.assertNotIn("NaN", output.read_text())
            with self.assertRaises(FileExistsError):
                write_state_stability((_run(),), output, **_metadata())

    def test_seed_mismatch_is_rejected(self):
        mismatch = _metadata()
        mismatch["candidate_seeds"] = (31,)
        with self.assertRaisesRegex(ValueError, "seed identity"):
            state_stability_payload((_run(),), **mismatch)


if __name__ == "__main__":
    unittest.main()
