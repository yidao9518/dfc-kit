import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.inference import infer_independent_endpoints, load_group_covariates, ols_hc3


def _payload(values: list[float], *, repeated: bool = False) -> dict[str, object]:
    rows = []
    for index, value in enumerate(values):
        subject = f"sub-{index:03d}"
        rows.append(
            {
                "subject": subject,
                "session": "ses-01",
                "acquisition_id": f"{subject}_run-1",
                "endpoint": "putamen_internal",
                "feature": ["putamen", "putamen"],
                "value": value,
            }
        )
        if repeated:
            rows.append(
                {
                    "subject": subject,
                    "session": "ses-02",
                    "acquisition_id": f"{subject}_run-2",
                    "endpoint": "putamen_internal",
                    "feature": ["putamen", "putamen"],
                    "value": value + 2.0,
                }
            )
    return {"format": "test-endpoints", "source_contract": "test:v1", "rows": rows}


class IndependentEndpointInferenceTests(unittest.TestCase):
    def test_hc3_group_effect_matches_direct_model_with_overlapping_subject_ids(self):
        left_values = np.asarray([4.0, 5.5, 5.0, 7.0, 6.2, 8.0])
        right_values = np.asarray([1.0, 2.5, 2.0, 3.0, 3.2, 4.0])
        covariates = {
            (group, f"sub-{index:03d}"): (age, male)
            for group, ages, males in (
                ("PD2", [50, 55, 60, 65, 70, 75], [0, 1, 0, 1, 0, 1]),
                ("HC", [45, 50, 55, 60, 65, 70], [1, 0, 1, 0, 1, 0]),
            )
            for index, (age, male) in enumerate(zip(ages, males, strict=True))
        }
        result = infer_independent_endpoints(
            _payload(left_values.tolist()),
            _payload(right_values.tolist()),
            group_a="PD2",
            group_b="HC",
            fdr_family="static FC",
            covariates=covariates,
            covariate_names=("age", "male"),
        )
        row = result["results"][0]
        outcome = np.concatenate((left_values, right_values))
        indicator = np.r_[np.ones(6), np.zeros(6)]
        nuisance = np.asarray(
            [covariates[("PD2", f"sub-{i:03d}")] for i in range(6)]
            + [covariates[("HC", f"sub-{i:03d}")] for i in range(6)]
        )
        nuisance = (nuisance - nuisance.mean(axis=0)) / nuisance.std(axis=0, ddof=1)
        expected = ols_hc3(
            outcome,
            np.column_stack((np.ones(12), indicator, nuisance)),
            column_names=("intercept", "group", "age", "male"),
        )
        self.assertAlmostEqual(row["estimate"], expected.coefficients[1])
        self.assertAlmostEqual(row["p"], expected.pvalues[1])
        self.assertAlmostEqual(
            row["standardized_estimate"],
            expected.coefficients[1] / outcome.std(ddof=1),
        )
        self.assertEqual(row["n_group_a"], 6)
        self.assertEqual(row["n_group_b"], 6)
        self.assertEqual(row["residual_degrees_of_freedom"], 8)
        self.assertLess(row["ci_low"], row["estimate"])
        self.assertGreater(row["ci_high"], row["estimate"])
        self.assertEqual(row["feature"], ["putamen", "putamen"])

    def test_repeated_acquisitions_are_subject_averaged(self):
        result = infer_independent_endpoints(
            _payload([3.0, 4.0, 5.0, 6.0], repeated=True),
            _payload([1.0, 2.0, 3.0, 4.0], repeated=True),
            group_a="PD2",
            group_b="HC",
            fdr_family="static FC",
        )
        row = result["results"][0]
        self.assertAlmostEqual(row["group_a_mean"], 5.5)
        self.assertAlmostEqual(row["group_b_mean"], 3.5)
        self.assertEqual(row["group_a_acquisitions_max"], 2)
        self.assertEqual(row["group_b_acquisitions_max"], 2)

    def test_covariate_loader_uses_group_and_subject_as_joint_identity(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "covariates.tsv"
            path.write_text(
                "group\tsubject\tage\tmale\n"
                "PD2\tsub-001\t70\t1\n"
                "HC\tsub-001\t60\t0\n",
                encoding="utf-8",
            )
            result = load_group_covariates(
                path, covariate_names=("age", "male")
            )
        self.assertEqual(result[("PD2", "sub-001")], (70.0, 1.0))
        self.assertEqual(result[("HC", "sub-001")], (60.0, 0.0))

    def test_metadata_mismatch_between_groups_is_rejected(self):
        left = _payload([1.0, 2.0, 3.0])
        right = json.loads(json.dumps(_payload([0.0, 1.0, 2.0])))
        for row in right["rows"]:
            row["feature"] = ["visual", "putamen"]
        with self.assertRaisesRegex(ValueError, "metadata differs between groups"):
            infer_independent_endpoints(
                left,
                right,
                group_a="PD2",
                group_b="HC",
                fdr_family="static FC",
            )


if __name__ == "__main__":
    unittest.main()
