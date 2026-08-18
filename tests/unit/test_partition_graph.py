import unittest

import numpy as np

from dfckit.networks import (
    FixedPartitionGraph,
    achieved_density,
    fixed_partition_modularity,
    participation_coefficient,
    partition_edge_summary,
    positive_proportional_adjacency,
)


class PartitionGraphKernelTests(unittest.TestCase):
    def test_positive_threshold_uses_only_available_positive_edges(self):
        connectivity = np.eye(5)
        connectivity[0, 1] = connectivity[1, 0] = 0.8
        connectivity[2, 3] = connectivity[3, 2] = 0.4
        connectivity[0, 2] = connectivity[2, 0] = -0.9

        adjacency = positive_proportional_adjacency(connectivity, density=0.8)

        self.assertEqual(np.count_nonzero(np.triu(adjacency, k=1)), 2)
        self.assertAlmostEqual(achieved_density(adjacency), 0.2)
        np.testing.assert_allclose(adjacency, adjacency.T)
        np.testing.assert_allclose(np.diag(adjacency), 0.0)
        self.assertTrue(np.all(adjacency >= 0.0))

    def test_positive_threshold_reaches_rounded_nominal_count(self):
        connectivity = np.eye(5)
        left, right = np.triu_indices(5, k=1)
        connectivity[left, right] = np.arange(1, 11) / 10
        connectivity[right, left] = connectivity[left, right]

        adjacency = positive_proportional_adjacency(connectivity, density=0.3)
        selected = adjacency[left, right]

        self.assertEqual(np.count_nonzero(selected), 3)
        np.testing.assert_allclose(sorted(selected[selected > 0]), [0.8, 0.9, 1.0])

    def test_modularity_and_participation_follow_fixed_partition(self):
        partition = ("left",) * 3 + ("right",) * 3
        adjacency = np.zeros((6, 6), dtype=float)
        for nodes in (range(3), range(3, 6)):
            for left in nodes:
                for right in nodes:
                    if left != right:
                        adjacency[left, right] = 1.0

        self.assertAlmostEqual(fixed_partition_modularity(adjacency, partition), 0.5)
        np.testing.assert_allclose(participation_coefficient(adjacency, partition), 0.0)

        adjacency[0, 3] = adjacency[3, 0] = 1.0
        participation = participation_coefficient(adjacency, partition)
        self.assertGreater(participation[0], 0.0)
        self.assertGreater(participation[3], 0.0)
        np.testing.assert_allclose(participation[[1, 2, 4, 5]], 0.0)

    def test_partition_edge_summary_counts_each_undirected_edge_once(self):
        partition = ("visual", "visual", "motor", "motor")
        weights = np.full((4, 4), -0.1, dtype=float)
        np.fill_diagonal(weights, 0.0)
        weights[0, 1] = weights[1, 0] = 0.5
        weights[2, 3] = weights[3, 2] = 0.5

        result = partition_edge_summary(weights, partition)

        self.assertAlmostEqual(result.within_mean, 0.5)
        self.assertAlmostEqual(result.between_mean, -0.1)
        self.assertAlmostEqual(result.within_minus_between, 0.6)
        self.assertAlmostEqual(result.segregation, 1.2)
        np.testing.assert_allclose(result.within_by_community, [0.5, 0.5])
        self.assertAlmostEqual(result.between_by_community[0, 1], -0.1)
        self.assertEqual(result.community_labels, ("visual", "motor"))

    def test_invalid_adjacency_and_partition_are_rejected(self):
        nonsymmetric = np.asarray([[0.0, 1.0], [0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "symmetric"):
            achieved_density(nonsymmetric)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            fixed_partition_modularity(np.asarray([[0.0, -1.0], [-1.0, 0.0]]), [0, 1])
        with self.assertRaisesRegex(ValueError, "one label per node"):
            participation_coefficient(np.zeros((3, 3)), [0, 1])


class FixedPartitionGraphTests(unittest.TestCase):
    def test_density_auc_matches_direct_trapezoidal_average(self):
        connectivity = np.eye(5)
        left, right = np.triu_indices(5, k=1)
        connectivity[left, right] = np.linspace(0.1, 1.0, len(left))
        connectivity[right, left] = connectivity[left, right]
        densities = np.asarray([0.2, 0.4, 0.6])

        result = FixedPartitionGraph(densities).transform(
            connectivity, ("a", "a", "b", "b", "b")
        )
        widths = np.diff(densities)
        expected_modularity_auc = np.sum(
            (result.modularity[:-1] + result.modularity[1:]) * widths / 2.0
        ) / (densities[-1] - densities[0])

        self.assertAlmostEqual(result.modularity_auc, expected_modularity_auc)
        np.testing.assert_allclose(result.achieved_densities, densities)
        self.assertEqual(result.participation.shape, (3, 5))
        self.assertEqual(result.participation_auc.shape, (5,))
        self.assertEqual(result.node_strength_auc.shape, (5,))

    def test_no_positive_edges_produce_zero_graph_metrics(self):
        connectivity = -np.ones((6, 6), dtype=float)
        np.fill_diagonal(connectivity, 1.0)

        result = FixedPartitionGraph((0.1, 0.2, 0.3)).transform(
            connectivity, (0, 0, 0, 1, 1, 1)
        )

        np.testing.assert_allclose(result.achieved_densities, 0.0)
        np.testing.assert_allclose(result.modularity, 0.0)
        np.testing.assert_allclose(result.participation, 0.0)
        np.testing.assert_allclose(result.node_strength, 0.0)
        self.assertEqual(result.achieved_density_auc, 0.0)

    def test_configuration_and_results_are_read_only(self):
        estimator = FixedPartitionGraph((0.1, 0.2))
        connectivity = np.eye(4)
        connectivity[0, 1] = connectivity[1, 0] = 0.8
        result = estimator.transform(connectivity, ("a", "a", "b", "b"))

        with self.assertRaises(ValueError):
            estimator.densities[0] = 0.5
        with self.assertRaises(ValueError):
            result.participation[0, 0] = 1.0

    def test_density_grid_must_be_valid_and_increasing(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            FixedPartitionGraph((0.1,))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            FixedPartitionGraph((0.2, 0.1))
        with self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
            FixedPartitionGraph((0.1, 1.1))


if __name__ == "__main__":
    unittest.main()
