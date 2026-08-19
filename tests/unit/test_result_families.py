import unittest

from dfckit.inference import adjust_result_families


class ResultFamilyTests(unittest.TestCase):
    def test_fdr_is_applied_across_all_members_of_each_named_family(self):
        records = [
            {"result_id": "a", "p": 0.01, "q": None, "fdr_family": "states"},
            {"result_id": "b", "p": 0.04, "q": None, "fdr_family": "states"},
            {"result_id": "c", "p": 0.04, "q": None, "fdr_family": "edges"},
            {
                "result_id": "d",
                "p": None,
                "q": None,
                "fdr_family": "states",
                "result_status": "not_testable",
            },
        ]
        adjusted = adjust_result_families(records)
        self.assertEqual(adjusted[0]["q"], 0.02)
        self.assertEqual(adjusted[1]["q"], 0.04)
        self.assertEqual(adjusted[2]["q"], 0.04)
        self.assertEqual(adjusted[3]["result_status"], "not_testable")
        self.assertTrue(all(item["result_status"] == "positive" for item in adjusted[:3]))

    def test_tested_result_requires_a_family(self):
        with self.assertRaisesRegex(ValueError, "fdr_family"):
            adjust_result_families([{"result_id": "a", "p": 0.1}])


if __name__ == "__main__":
    unittest.main()
