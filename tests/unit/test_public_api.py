import unittest

import dfckit
from dfckit import artifacts, connectivity, inference, information, states, storage


class PublicApiTests(unittest.TestCase):
    def test_root_package_keeps_only_data_and_topology_objects(self):
        self.assertEqual(
            set(dfckit.__all__),
            {"TimeSeriesDataset", "TimeSeriesRun", "TimeWindow", "validate_subject_disjoint"},
        )
        self.assertFalse(hasattr(dfckit, "fit_feature_reference"))

    def test_internal_builders_are_not_package_exports(self):
        forbidden = {
            "state_model_scores_payload",
            "state_model_specification",
            "state_stability_payload",
            "intercept_t_statistic",
            "threshold_components",
            "infer_paired_endpoints_file",
            "write_paired_endpoint_inference",
            "infer_paired_state_metrics_file",
            "write_paired_state_inference",
            "state_count_comparison_payload",
            "write_state_count_comparison",
            "zscore_columns",
            "eligible_fixed_window_count",
            "estimate_fixed_windows",
            "STORE_STATISTICS",
            "summarize_store_file",
            "write_store_summary",
        }
        for module in (artifacts, inference, information, states, storage):
            with self.subTest(module=module.__name__):
                self.assertTrue(forbidden.isdisjoint(module.__all__))

    def test_low_level_connectivity_kernels_are_not_top_level_exports(self):
        forbidden = {
            "analytic_phase",
            "cross_block_phase_coherence",
            "leading_phase_eigenvectors",
            "within_block_phase_coherence",
            "cross_block_products",
            "edge_products",
            "edge_rss",
            "within_block_products",
        }
        self.assertTrue(forbidden.isdisjoint(connectivity.__all__))
        self.assertIn("ETS", connectivity.__all__)
        self.assertIn("LEiDA", connectivity.__all__)


if __name__ == "__main__":
    unittest.main()
