import unittest

from dfckit.inference import infer_paired_state_metrics


def _run(subject, session, occupancy, dwell):
    return {
        "subject": subject,
        "session": session,
        "acquisition_id": f"{subject}_{session}",
        "occupancy": occupancy,
        "mean_dwell_seconds": dwell,
        "switch_rate": 0.2,
        "transition_probabilities": [[0.8, 0.2], [0.3, 0.7]],
    }


class StateMetricInferenceTests(unittest.TestCase):
    def test_exact_sign_flip_fdr_and_not_testable_are_reported(self):
        runs = []
        for index in range(6):
            subject = f"sub-{index:03d}"
            off_dwell = [2.0, 1.0 if index == 0 else None]
            on_dwell = [3.0, 2.0 if index == 0 else None]
            runs.extend(
                [
                    _run(subject, "off", [0.4, 0.6], off_dwell),
                    _run(subject, "on", [0.6, 0.4], on_dwell),
                ]
            )
        payload = {
            "n_states": 2,
            "model_kind": "kmeans-state",
            "source_contract": "window-fc:test",
            "runs": runs,
        }
        result = infer_paired_state_metrics(
            payload,
            condition_a="on",
            condition_b="off",
            metrics=("occupancy", "mean_dwell_seconds"),
            fdr_family="state dynamics",
            exact=True,
            n_bootstrap=100,
            seed=9,
        )
        by_endpoint = {item["endpoint"]: item for item in result["results"]}
        self.assertAlmostEqual(by_endpoint["occupancy.state_0"]["estimate"], 0.2)
        self.assertEqual(by_endpoint["occupancy.state_0"]["p"], 0.03125)
        self.assertEqual(by_endpoint["occupancy.state_0"]["result_status"], "positive")
        self.assertEqual(
            by_endpoint["mean_dwell_seconds.state_1"]["result_status"],
            "not_testable",
        )
        self.assertEqual(result["fdr_family"], "state dynamics")

    def test_duplicate_subject_condition_is_rejected(self):
        record = _run("sub-001", "on", [0.5, 0.5], [1.0, 1.0])
        payload = {"n_states": 2, "runs": [record, record]}
        with self.assertRaisesRegex(ValueError, "duplicate runs"):
            infer_paired_state_metrics(
                payload,
                condition_a="on",
                condition_b="off",
                metrics=("occupancy",),
                fdr_family="states",
                exact=True,
                seed=0,
            )


if __name__ == "__main__":
    unittest.main()
