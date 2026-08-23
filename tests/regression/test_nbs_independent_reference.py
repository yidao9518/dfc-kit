import importlib.util
import unittest

import numpy as np

from dfckit.inference import paired_nbs

SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


def _reference_design(values, confounds):
    if confounds is None:
        centered = None
        design = np.ones((len(values), 1), dtype=float)
    else:
        nuisance = np.asarray(confounds, dtype=float)
        if nuisance.ndim == 1:
            nuisance = nuisance[:, None]
        centered = nuisance - nuisance.mean(axis=0, keepdims=True)
        design = np.column_stack((np.ones(len(values)), centered))
    inverse = np.linalg.solve(design.T @ design, np.eye(design.shape[1]))
    return design, centered, inverse, len(values) - np.linalg.matrix_rank(design)


def _reference_t(values, design, inverse, degrees):
    coefficients = np.linalg.solve(design.T @ design, design.T @ values)
    residuals = values - design @ coefficients
    variance = np.einsum("ij,ij->j", residuals, residuals) / degrees
    return coefficients[0] / np.sqrt(variance * inverse[0, 0])


def _reference_components(
    statistics, edge_i, edge_j, n_nodes, threshold, component_sign_mode="separate"
):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    output = {}
    selections = (
        (("pooled", np.abs(statistics) > threshold),)
        if component_sign_mode == "pooled"
        else (
            ("positive", statistics > threshold),
            ("negative", statistics < -threshold),
        )
    )
    for direction, selected in selections:
        chosen = np.flatnonzero(selected)
        if not len(chosen):
            output[direction] = ()
            continue
        rows = np.concatenate((edge_i[chosen], edge_j[chosen]))
        columns = np.concatenate((edge_j[chosen], edge_i[chosen]))
        adjacency = coo_matrix(
            (np.ones(len(rows)), (rows, columns)), shape=(n_nodes, n_nodes)
        ).tocsr()
        _, labels = connected_components(adjacency, directed=False, return_labels=True)
        grouped = {}
        for edge in chosen:
            grouped.setdefault(int(labels[edge_i[edge]]), []).append(int(edge))
        components = []
        for indices in grouped.values():
            edges = tuple(sorted(indices))
            nodes = tuple(
                sorted(
                    {int(edge_i[index]) for index in edges}
                    | {int(edge_j[index]) for index in edges}
                )
            )
            components.append((nodes, edges, float(len(edges))))
        components.sort(key=lambda item: (-item[2], -len(item[1]), -len(item[0]), item[0], item[1]))
        output[direction] = tuple(components)
    return output


def _reference_nbs(
    values,
    edge_i,
    edge_j,
    n_nodes,
    thresholds,
    permutations,
    seed,
    confounds,
    component_sign_mode="separate",
):
    design, centered, inverse, degrees = _reference_design(values, confounds)
    observed_t = _reference_t(values, design, inverse, degrees)
    observed = {
        threshold: _reference_components(
            observed_t,
            edge_i,
            edge_j,
            n_nodes,
            threshold,
            component_sign_mode,
        )
        for threshold in thresholds
    }
    if centered is None:
        fitted = np.zeros_like(values)
        residuals = values.copy()
    else:
        nuisance_coefficients = np.linalg.lstsq(centered, values, rcond=None)[0]
        fitted = centered @ nuisance_coefficients
        residuals = values - fitted

    directions = (
        ("pooled",) if component_sign_mode == "pooled" else ("positive", "negative")
    )
    nulls = {
        direction: {
            threshold: np.zeros(permutations) for threshold in thresholds
        }
        for direction in directions
    }
    rng = np.random.default_rng(seed)
    for index in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=(len(values), 1))
        permuted_t = _reference_t(fitted + residuals * signs, design, inverse, degrees)
        for threshold in thresholds:
            components = _reference_components(
                permuted_t,
                edge_i,
                edge_j,
                n_nodes,
                threshold,
                component_sign_mode,
            )
            for direction in directions:
                nulls[direction][threshold][index] = max(
                    (component[2] for component in components[direction]), default=0.0
                )
    return observed_t, observed, nulls


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy is not installed")
class IndependentNBSReferenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(80421)
        self.edge_i, self.edge_j = np.triu_indices(6, 1)
        self.values = rng.normal(scale=0.5, size=(24, len(self.edge_i)))
        self.values[:, [0, 1, 5]] += 0.38
        self.values[:, [8, 12, 14]] -= 0.42
        self.confounds = rng.normal(size=(24, 2))
        self.subjects = tuple(f"sub-{index:03d}" for index in range(24))

    def test_primary_and_adjusted_results_match_independent_scipy_reference(self):
        for confounds, names in ((None, ()), (self.confounds, ("motion", "censor"))):
            with self.subTest(adjusted=confounds is not None):
                expected_t, expected_components, nulls = _reference_nbs(
                    self.values,
                    self.edge_i,
                    self.edge_j,
                    6,
                    (2.0, 2.5, 3.0),
                    256,
                    9917,
                    confounds,
                )
                result = paired_nbs(
                    self.values,
                    self.subjects,
                    self.edge_i,
                    self.edge_j,
                    6,
                    thresholds=(2.0, 2.5, 3.0),
                    n_permutations=256,
                    seed=9917,
                    difference_direction="condition B minus condition A",
                    confounds=confounds,
                    confound_names=names,
                )
                for threshold in (2.0, 2.5, 3.0):
                    actual = result.at_threshold(threshold)
                    np.testing.assert_allclose(actual.observed_t, expected_t, atol=1e-12)
                    positive = nulls["positive"][threshold]
                    negative = nulls["negative"][threshold]
                    np.testing.assert_array_equal(actual.null_positive, positive)
                    np.testing.assert_array_equal(actual.null_negative, negative)
                    np.testing.assert_array_equal(
                        actual.null_maximum,
                        np.maximum(positive, negative),
                    )
                    for direction, components in (
                        ("positive", actual.positive_components),
                        ("negative", actual.negative_components),
                    ):
                        signatures = tuple(
                            (
                                component.node_indices,
                                component.edge_indices,
                                component.statistic_value,
                            )
                            for component in components
                        )
                        self.assertEqual(
                            signatures, expected_components[threshold][direction]
                        )
                        for component in components:
                            expected_p = (
                                1
                                + np.count_nonzero(
                                    actual.null_maximum >= component.statistic_value
                                )
                            ) / 257
                            self.assertEqual(component.fwe_pvalue, expected_p)

    def test_pooled_results_match_independent_scipy_reference(self):
        expected_t, expected_components, nulls = _reference_nbs(
            self.values,
            self.edge_i,
            self.edge_j,
            6,
            (2.0, 2.5, 3.0),
            256,
            9917,
            self.confounds,
            component_sign_mode="pooled",
        )
        result = paired_nbs(
            self.values,
            self.subjects,
            self.edge_i,
            self.edge_j,
            6,
            thresholds=(2.0, 2.5, 3.0),
            n_permutations=256,
            seed=9917,
            difference_direction="condition B minus condition A",
            component_sign_mode="pooled",
            confounds=self.confounds,
            confound_names=("motion", "censor"),
        )

        for threshold in (2.0, 2.5, 3.0):
            actual = result.at_threshold(threshold)
            expected_null = nulls["pooled"][threshold]
            np.testing.assert_allclose(actual.observed_t, expected_t, atol=1e-12)
            np.testing.assert_array_equal(actual.null_pooled, expected_null)
            np.testing.assert_array_equal(actual.null_maximum, expected_null)
            signatures = tuple(
                (
                    component.node_indices,
                    component.edge_indices,
                    component.statistic_value,
                )
                for component in actual.pooled_components
            )
            self.assertEqual(signatures, expected_components[threshold]["pooled"])
            for component in actual.pooled_components:
                expected_p = (
                    1
                    + np.count_nonzero(
                        actual.null_maximum >= component.statistic_value
                    )
                ) / 257
                self.assertEqual(component.fwe_pvalue, expected_p)


if __name__ == "__main__":
    unittest.main()
