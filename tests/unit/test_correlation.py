import unittest

import numpy as np

from dfckit.connectivity import (
    edge_vector_to_symmetric_matrix,
    fisher_z_edges,
    weighted_correlation,
)


class CorrelationTests(unittest.TestCase):
    def test_perfect_positive_and_negative_relations(self):
        values = np.column_stack(
            [
                np.arange(6, dtype=float),
                np.arange(6, dtype=float) * 2.0,
                -np.arange(6, dtype=float),
            ]
        )
        correlation = weighted_correlation(values)
        np.testing.assert_allclose(correlation[0], [1.0, 1.0, -1.0], atol=1e-12)

    def test_fisher_edges_have_stable_order(self):
        matrix = np.array([[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]])
        edges, left, right = fisher_z_edges(matrix)
        np.testing.assert_array_equal(left, [0, 0, 1])
        np.testing.assert_array_equal(right, [1, 2, 2])
        np.testing.assert_allclose(edges, np.arctanh([0.1, 0.2, 0.3]))

    def test_constant_roi_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "constant"):
            weighted_correlation(np.column_stack([np.arange(5), np.ones(5)]))

    def test_edge_vector_restores_symmetric_matrix(self):
        observed = edge_vector_to_symmetric_matrix(
            [0.1, -0.2, 0.3],
            [0, 0, 1],
            [1, 2, 2],
            n_nodes=3,
            diagonal=1.0,
        )
        expected = np.asarray(
            [[1.0, 0.1, -0.2], [0.1, 1.0, 0.3], [-0.2, 0.3, 1.0]]
        )
        np.testing.assert_allclose(observed, expected)
        with self.assertRaises(ValueError):
            observed[0, 1] = 0.0

    def test_edge_vector_rejects_invalid_identity(self):
        with self.assertRaisesRegex(ValueError, "edge_i < edge_j"):
            edge_vector_to_symmetric_matrix([0.1], [1], [0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            edge_vector_to_symmetric_matrix([0.1, 0.2], [0, 0], [1, 1])
        with self.assertRaisesRegex(TypeError, "integer"):
            edge_vector_to_symmetric_matrix([0.1], [0.0], [1.0])


if __name__ == "__main__":
    unittest.main()
