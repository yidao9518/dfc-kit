import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.io import state_stability_payload, write_state_stability
from dfckit.states import RunStateStability, StateAlignment


def _run() -> RunStateStability:
    return RunStateStability(
        subject="sub-010",
        session="off",
        acquisition_id="sub-010_ses-off_task-rest_run-1",
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
        matched_correlations=[0.91, 0.87],
        correlation_matrix=[[0.2, 0.91], [0.87, 0.1]],
        reference_seed=17,
        candidate_seed=29,
        feature_keys=(("visual", "motor"), ("visual", "putamen")),
        source_contract="stability-io-test:v1",
        sample_interval_seconds=0.8,
    )


def _metadata() -> dict[str, object]:
    return {
        "model_kind": "kmeans-state",
        "reference_model_fingerprint": "a" * 64,
        "reference_seed": 17,
        "candidate_model_fingerprints": ("b" * 64,),
        "candidate_seeds": (29,),
        "alignments": (_alignment(),),
        "training_data_fingerprint": "c" * 64,
        "source_contract": "stability-io-test:v1",
        "sample_interval_seconds": 0.8,
        "allow_fit_subjects": False,
    }


class StateStabilityIOTests(unittest.TestCase):
    def test_payload_records_reference_identity_alignment_and_finite_statistics(self):
        payload = state_stability_payload((_run(),), **_metadata())
        self.assertEqual(payload["format"], "dfckit-state-stability")
        self.assertEqual(payload["reference_model_fingerprint"], "a" * 64)
        self.assertEqual(payload["training_data_fingerprint"], "c" * 64)
        self.assertEqual(payload["fits"][0]["candidate_to_reference"], [0, 1])
        self.assertEqual(payload["fits"][1]["candidate_to_reference"], [1, 0])
        self.assertEqual(payload["fits"][1]["correlation_matrix"], [[0.2, 0.91], [0.87, 0.1]])
        self.assertIn("ddof=0", payload["dispersion_standard_deviation"])
        switch = payload["runs"][0]["switch_rate"]
        self.assertEqual(switch["by_fit"], [0.5, None])
        self.assertEqual(switch["valid_fit_count"], 1)
        self.assertIsNone(switch["standard_deviation"])
        self.assertAlmostEqual(
            payload["runs"][0]["occupancy"]["standard_deviation"][0],
            1 / 12,
        )
        transition = payload["runs"][0]["transition_probabilities"]
        self.assertEqual(transition["valid_fit_count"][1], [1, 1])
        self.assertEqual(payload["n_samples"], 6)

    def test_writer_uses_strict_json_and_refuses_overwrite(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "stability.json"
            write_state_stability((_run(),), output, **_metadata())
            text = output.read_text(encoding="utf-8")
            payload = json.loads(text)
            self.assertNotIn("NaN", text)
            self.assertEqual(payload["n_fits"], 2)
            with self.assertRaises(FileExistsError):
                write_state_stability((_run(),), output, **_metadata())

    def test_duplicate_model_identity_and_seed_mismatch_are_rejected(self):
        duplicate = _metadata()
        duplicate["candidate_model_fingerprints"] = ("a" * 64,)
        with self.assertRaisesRegex(ValueError, "unique"):
            state_stability_payload((_run(),), **duplicate)

        mismatch = _metadata()
        mismatch["candidate_seeds"] = (31,)
        with self.assertRaisesRegex(ValueError, "seed identity"):
            state_stability_payload((_run(),), **mismatch)


if __name__ == "__main__":
    unittest.main()
