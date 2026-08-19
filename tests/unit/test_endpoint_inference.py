import unittest

from dfckit.inference import infer_paired_endpoints


class EndpointInferenceTests(unittest.TestCase):
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
                        "value": 0.0,
                    },
                    {
                        "subject": subject,
                        "session": "on",
                        "endpoint": "feature_0",
                        "feature": ["visual", "motor"],
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


if __name__ == "__main__":
    unittest.main()
