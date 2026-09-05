import unittest

import numpy as np

from dfckit.inference import infer_paired_nbs_endpoints


def _payload(*, repeated: bool = False) -> dict[str, object]:
    rng = np.random.default_rng(27)
    nodes = ("visual", "motor", "putamen", "thalamus")
    edges = tuple(
        (nodes[left], nodes[right])
        for left in range(len(nodes))
        for right in range(left + 1, len(nodes))
    )
    rows = []
    for subject_index in range(12):
        subject = f"sub-{subject_index:03d}"
        baseline = rng.normal(0.0, 0.25, len(edges))
        effect = rng.normal(0.0, 0.15, len(edges))
        effect[:2] += 0.65
        for session, values in (("off", baseline), ("on", baseline + effect)):
            runs = (values, values + 0.01) if repeated else (values,)
            for run_index, run_values in enumerate(runs, start=1):
                for edge_index, (edge, value) in enumerate(zip(edges, run_values, strict=True)):
                    rows.append(
                        {
                            "subject": subject,
                            "session": session,
                            "acquisition_id": f"{subject}_{session}_run-{run_index}",
                            "endpoint": f"feature_{edge_index}.mean",
                            "feature": list(edge),
                            "statistic": "mean",
                            "value": float(value),
                        }
                    )
    return {
        "format": "dfc-kit-store-endpoints",
        "format_version": 2,
        "source_contract": "sliding-window-fc:test",
        "feature_type": "edge",
        "n_features": len(edges),
        "rows": rows,
    }


class NBSEndpointTests(unittest.TestCase):
    def test_endpoint_adapter_preserves_network_component_semantics(self):
        result = infer_paired_nbs_endpoints(
            _payload(),
            condition_a="on",
            condition_b="off",
            statistic="mean",
            thresholds=(2.0, 3.0),
            n_permutations=100,
            seed=43,
        )

        self.assertEqual(result["format"], "dfc-kit-paired-nbs")
        self.assertEqual(result["node_names"], ["visual", "motor", "putamen", "thalamus"])
        self.assertEqual(result["n_subjects"], 12)
        self.assertEqual(len(result["threshold_results"]), 2)
        self.assertTrue(result["results"])
        self.assertTrue(all(row["q"] == row["p"] for row in result["results"]))
        self.assertTrue(
            any("visual" in row["component_nodes"] for row in result["results"])
        )
        self.assertIn("component FWE", result["correction_method"])

    def test_repeated_acquisitions_require_explicit_mean_aggregation(self):
        with self.assertRaisesRegex(ValueError, "duplicate acquisitions"):
            infer_paired_nbs_endpoints(
                _payload(repeated=True),
                condition_a="on",
                condition_b="off",
                statistic="mean",
                thresholds=(2.0,),
                n_permutations=20,
                seed=2,
            )

        result = infer_paired_nbs_endpoints(
            _payload(repeated=True),
            condition_a="on",
            condition_b="off",
            statistic="mean",
            thresholds=(2.0,),
            n_permutations=20,
            seed=2,
            within_condition_aggregation="mean",
        )
        self.assertEqual(result["condition_a_acquisitions_min"], 2)
        self.assertEqual(result["condition_b_acquisitions_max"], 2)

    def test_incomplete_graph_is_rejected(self):
        payload = _payload()
        payload["rows"] = [
            row for row in payload["rows"] if row["endpoint"] != "feature_5.mean"
        ]
        with self.assertRaisesRegex(ValueError, "every indexed edge"):
            infer_paired_nbs_endpoints(
                payload,
                condition_a="on",
                condition_b="off",
                statistic="mean",
                thresholds=(2.0,),
                n_permutations=20,
                seed=2,
            )


if __name__ == "__main__":
    unittest.main()
