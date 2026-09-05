import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dfckit.inference import infer_paired_endpoints, infer_paired_state_metrics
from dfckit.inference.paired import paired_bootstrap_mean_ci, paired_sign_flip
from dfckit.inference.state_metrics import _write_paired_state_inference


def _run(subject, session, occupancy, dwell, acquisition_id=None):
    return {
        "subject": subject,
        "session": session,
        "acquisition_id": acquisition_id or f"{subject}_{session}",
        "occupancy": occupancy,
        "mean_dwell_seconds": dwell,
        "switch_rate": 0.2,
        "transition_probabilities": [[0.8, 0.2], [0.3, 0.7]],
    }


class StateMetricInferenceTests(unittest.TestCase):
    def test_absent_canonical_endpoints_still_occupy_their_seed_indices(self):
        subjects = tuple(f"sub-{index:03d}" for index in range(6))
        differences = [(index + 1) / 16 for index in range(6)]
        runs = [
            _run(subject, condition, [value, None], [None, None])
            for subject, difference in zip(subjects, differences, strict=True)
            for condition, value in (("off", 0.125), ("on", 0.125 + difference))
        ]
        result = infer_paired_state_metrics(
            {"n_states": 2, "runs": runs},
            condition_a="on",
            condition_b="off",
            metrics=("occupancy", "mean_dwell_seconds"),
            fdr_family="canonical",
            n_permutations=37,
            n_bootstrap=49,
            seed=23,
        )
        self.assertEqual(result["n_endpoints"], 4)
        self.assertEqual(result["n_tested"], 1)
        row = result["results"][2]
        self.assertEqual(row["endpoint"], "occupancy.state_0")
        test = paired_sign_flip(differences, subjects, n_permutations=37, seed=25)
        interval = paired_bootstrap_mean_ci(differences, subjects, n_resamples=49, seed=100_025)
        self.assertEqual(
            (row["p"], row["ci_low"], row["ci_high"]), (test.pvalue, interval.lower, interval.upper)
        )
        self.assertNotIn("standardized_estimate", row)
        self.assertEqual(result["results"][0]["n"], 0)

    def test_state_errors_become_not_testable_but_generic_errors_propagate(self):
        runs = [
            _run(f"sub-{index}", condition, [value, 1.0 - value], [1.0, 1.0])
            for index in range(3)
            for condition, value in (("off", 0.25), ("on", 0.5))
        ]
        rows = [
            {
                "subject": run["subject"],
                "session": run["session"],
                "endpoint": "occupancy.state_0",
                "value": run["occupancy"][0],
            }
            for run in runs
        ]
        for argument in ("n_permutations", "n_bootstrap"):
            with self.subTest(argument=argument):
                options = {"n_permutations": 7, "n_bootstrap": 9, argument: 0}
                kwargs = dict(condition_a="on", condition_b="off", fdr_family="states", **options)
                result = infer_paired_state_metrics(
                    {"n_states": 2, "runs": runs}, metrics=("occupancy",), **kwargs
                )
                self.assertEqual(result["n_tested"], 0)
                self.assertTrue(
                    all(row["result_status"] == "not_testable" for row in result["results"])
                )
                self.assertTrue(
                    all("positive integer" in row["reason"] for row in result["results"])
                )
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    infer_paired_endpoints({"rows": rows}, **kwargs)

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

    def test_repeated_runs_can_be_averaged_without_crossing_run_boundaries(self):
        runs = []
        for index in range(6):
            subject = f"sub-{index:03d}"
            runs.extend(
                [
                    _run(subject, "off", [0.2, 0.8], [1.0, 1.0], f"{subject}_off_1"),
                    _run(subject, "off", [0.4, 0.6], [1.0, 1.0], f"{subject}_off_2"),
                    _run(subject, "on", [0.6, 0.4], [1.0, 1.0], f"{subject}_on_1"),
                    _run(subject, "on", [0.8, 0.2], [1.0, 1.0], f"{subject}_on_2"),
                ]
            )
        result = infer_paired_state_metrics(
            {"n_states": 2, "runs": runs},
            condition_a="on",
            condition_b="off",
            metrics=("occupancy",),
            fdr_family="states",
            exact=True,
            within_condition_aggregation="mean",
            n_bootstrap=50,
            seed=3,
        )
        endpoint = next(
            item for item in result["results"] if item["endpoint"] == "occupancy.state_0"
        )
        self.assertAlmostEqual(endpoint["estimate"], 0.4)
        self.assertEqual(endpoint["condition_a_acquisitions_min"], 2)
        self.assertEqual(endpoint["condition_b_acquisitions_max"], 2)
        self.assertEqual(result["within_condition_aggregation"], "mean")
        with TemporaryDirectory() as temporary:
            output = _write_paired_state_inference(
                result,
                Path(temporary) / "inference",
            )
            with (output / "results.tsv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            written = next(item for item in rows if item["endpoint"] == "occupancy.state_0")
            self.assertEqual(written["condition_a_acquisitions_min"], "2")
            self.assertEqual(written["condition_b_acquisitions_max"], "2")


if __name__ == "__main__":
    unittest.main()
