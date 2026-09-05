import unittest

import numpy as np

from dfckit.inference import infer_paired_endpoints


class EndpointInferenceTests(unittest.TestCase):
    @staticmethod
    def _multi_endpoint_rows():
        rows = []
        for index in range(6):
            subject = f"sub-{index:03d}"
            for endpoint_index, endpoint in enumerate(("alpha", "beta", "gamma")):
                off = float(index + endpoint_index)
                difference = float((index + 1) * (endpoint_index + 1)) / 10.0
                rows.extend(
                    [
                        {
                            "subject": subject,
                            "session": "off",
                            "endpoint": endpoint,
                            "value": off,
                        },
                        {
                            "subject": subject,
                            "session": "on",
                            "endpoint": endpoint,
                            "value": off + difference,
                        },
                    ]
                )
        return rows

    def test_features_are_preserved_through_paired_inference(self):
        rows = []
        for index in range(6):
            subject = f"sub-{index:03d}"
            rows.extend(
                [
                    {
                        "subject": subject,
                        "session": "off",
                        "endpoint": "feature_0",
                        "feature": ["visual", "motor"],
                        "statistic": "variance",
                        "value": 0.0,
                    },
                    {
                        "subject": subject,
                        "session": "on",
                        "endpoint": "feature_0",
                        "feature": ["visual", "motor"],
                        "statistic": "variance",
                        "value": 1.0,
                    },
                ]
            )
        result = infer_paired_endpoints(
            {"format": "test", "source_contract": "mtd:test", "rows": rows},
            condition_a="on",
            condition_b="off",
            fdr_family="edges",
            exact=True,
            n_bootstrap=50,
            seed=4,
        )
        endpoint = result["results"][0]
        self.assertEqual(endpoint["feature"], ["visual", "motor"])
        self.assertEqual(endpoint["statistic"], "variance")
        self.assertEqual(endpoint["estimate"], 1.0)
        self.assertEqual(endpoint["p"], 0.03125)
        self.assertEqual(endpoint["result_status"], "positive")

    def test_duplicate_acquisition_endpoint_is_rejected(self):
        row = {
            "subject": "sub-001",
            "session": "on",
            "endpoint": "feature_0",
            "feature": ["a", "b"],
            "value": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "duplicate condition"):
            infer_paired_endpoints(
                {"rows": [row, row]},
                condition_a="on",
                condition_b="off",
                fdr_family="edges",
                exact=True,
            )

    def test_declared_missing_endpoint_is_not_testable(self):
        rows = [
            {
                "subject": f"sub-{index:03d}",
                "session": session,
                "endpoint": "mean_cmi.length_20",
                "measure": "mean_cmi",
                "length": 20,
                "value": None,
            }
            for index in range(3)
            for session in ("off", "on")
        ]
        result = infer_paired_endpoints(
            {"rows": rows},
            condition_a="on",
            condition_b="off",
            fdr_family="information",
            exact=True,
        )
        self.assertEqual(result["results"][0]["result_status"], "not_testable")
        self.assertEqual(result["results"][0]["measure"], "mean_cmi")

    def test_repeated_acquisitions_can_be_averaged_within_condition(self):
        rows = []
        for index in range(6):
            subject = f"sub-{index:03d}"
            for session, values in (("off", (0.0, 2.0)), ("on", (3.0, 5.0))):
                for run, value in enumerate(values, start=1):
                    rows.append(
                        {
                            "subject": subject,
                            "session": session,
                            "acquisition_id": f"{subject}_{session}_run-{run}",
                            "endpoint": "feature_0",
                            "value": value,
                        }
                    )
        result = infer_paired_endpoints(
            {"rows": rows},
            condition_a="on",
            condition_b="off",
            fdr_family="edges",
            exact=True,
            within_condition_aggregation="mean",
            n_bootstrap=50,
            seed=4,
        )
        endpoint = result["results"][0]
        self.assertEqual(endpoint["estimate"], 3.0)
        self.assertIsNone(endpoint["standardized_estimate"])
        self.assertEqual(
            endpoint["standardized_estimate_definition"], "paired Cohen dz"
        )
        self.assertEqual(endpoint["condition_a_acquisitions_min"], 2)
        self.assertEqual(endpoint["condition_b_acquisitions_max"], 2)
        self.assertEqual(result["within_condition_aggregation"], "mean")

    def test_paired_cohen_dz_uses_complete_pair_differences(self):
        rows = []
        differences = [1.0, 2.0, 3.0, 4.0]
        for index, difference in enumerate(differences):
            subject = f"sub-{index:03d}"
            rows.extend(
                [
                    {
                        "subject": subject,
                        "session": "off",
                        "endpoint": "feature_0",
                        "value": 10.0 + index,
                    },
                    {
                        "subject": subject,
                        "session": "on",
                        "endpoint": "feature_0",
                        "value": 10.0 + index + difference,
                    },
                ]
            )

        result = infer_paired_endpoints(
            {"rows": rows},
            condition_a="on",
            condition_b="off",
            fdr_family="edges",
            exact=True,
            n_bootstrap=50,
        )

        observed = result["results"][0]["standardized_estimate"]
        expected = np.mean(differences) / np.std(differences, ddof=1)
        self.assertAlmostEqual(observed, expected)

    def test_explicit_endpoint_selection_defines_the_tested_family(self):
        payload = {"format": "test", "rows": self._multi_endpoint_rows()}
        all_endpoints = infer_paired_endpoints(
            payload,
            condition_a="on",
            condition_b="off",
            fdr_family="all endpoints",
            n_permutations=101,
            n_bootstrap=101,
            seed=23,
        )
        selected = infer_paired_endpoints(
            payload,
            condition_a="on",
            condition_b="off",
            fdr_family="prespecified beta and gamma endpoints",
            endpoint_names=("gamma", "beta"),
            n_permutations=101,
            n_bootstrap=101,
            seed=23,
        )

        self.assertEqual(
            all_endpoints["endpoint_selection"],
            {
                "mode": "all",
                "requested_endpoint_names": None,
                "selected_endpoint_names": ["alpha", "beta", "gamma"],
                "n_source_endpoints": 3,
            },
        )
        self.assertEqual(
            [result["endpoint"] for result in selected["results"]], ["beta", "gamma"]
        )
        self.assertEqual(selected["n_endpoints"], 2)
        self.assertEqual(
            selected["endpoint_selection"],
            {
                "mode": "explicit",
                "requested_endpoint_names": ["gamma", "beta"],
                "selected_endpoint_names": ["beta", "gamma"],
                "n_source_endpoints": 3,
            },
        )
        all_by_name = {result["endpoint"]: result for result in all_endpoints["results"]}
        for result in selected["results"]:
            reference = all_by_name[result["endpoint"]]
            self.assertEqual(result["p"], reference["p"])
            self.assertEqual(result["ci_low"], reference["ci_low"])
            self.assertEqual(result["ci_high"], reference["ci_high"])

    def test_endpoint_selection_rejects_empty_duplicate_and_unknown_names(self):
        payload = {"rows": self._multi_endpoint_rows()}
        cases = (
            ((), "must not be empty"),
            (("alpha", "alpha"), "must not contain duplicates"),
            (("missing",), "unknown endpoint_names: 'missing'"),
            (("",), "only non-empty names"),
        )
        for endpoint_names, message in cases:
            with (
                self.subTest(endpoint_names=endpoint_names),
                self.assertRaisesRegex(ValueError, message),
            ):
                infer_paired_endpoints(
                    payload,
                    condition_a="on",
                    condition_b="off",
                    fdr_family="selected endpoints",
                    endpoint_names=endpoint_names,
                    exact=True,
                )


if __name__ == "__main__":
    unittest.main()
