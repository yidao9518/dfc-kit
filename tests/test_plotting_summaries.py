from __future__ import annotations

import io
import unittest

import numpy as np


class SummaryPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import matplotlib
        except ImportError:
            raise unittest.SkipTest("matplotlib is not installed") from None
        matplotlib.use("Agg")
        from dfckit.plotting import (
            plot_association,
            plot_curve,
            plot_effects,
            plot_null_distribution,
            plot_paired,
        )

        cls.plot_association = staticmethod(plot_association)
        cls.plot_curve = staticmethod(plot_curve)
        cls.plot_effects = staticmethod(plot_effects)
        cls.plot_null_distribution = staticmethod(plot_null_distribution)
        cls.plot_paired = staticmethod(plot_paired)

    def tearDown(self) -> None:
        from matplotlib import pyplot as plt

        plt.close("all")

    def test_paired_preserves_rows_and_raw_values(self) -> None:
        values = np.array([[1.0, 4.0, 2.0], [7.0, 3.0, 9.0]])
        axes = self.plot_paired(values, labels=("OFF", "ON", "follow-up"), seed=5)
        self.assertEqual(len(axes.lines), 2)
        for line, row in zip(axes.lines, values, strict=True):
            np.testing.assert_array_equal(line.get_ydata(), row)
            self.assertAlmostEqual(np.ptp(line.get_xdata() - np.arange(3)), 0.0)
        plotted = np.column_stack(
            [collection.get_offsets()[:, 1] for collection in axes.collections]
        )
        np.testing.assert_array_equal(plotted, values)

    def test_paired_uses_local_seed_and_explicit_summary(self) -> None:
        state_before = np.random.get_state()
        axes = self.plot_paired(
            [[1.0, 2.0], [2.0, 3.0]],
            labels=("OFF", "ON"),
            estimates=(1.5, 2.5),
            errors=(0.2, 0.3),
            error_label="SE",
            seed=3,
        )
        state_after = np.random.get_state()
        self.assertEqual(state_before[0], state_after[0])
        np.testing.assert_array_equal(state_before[1], state_after[1])
        self.assertEqual(state_before[2:], state_after[2:])
        handles, labels = axes.get_legend_handles_labels()
        self.assertTrue(handles)
        self.assertIn("SE", labels)

    def test_paired_rejects_invalid_data_and_orphan_errors(self) -> None:
        masked = np.ma.array([[1.0, 2.0]], mask=[[False, True]])
        invalid = (
            lambda: self.plot_paired([[1.0]], labels=("one",)),
            lambda: self.plot_paired([[1.0, np.nan]], labels=("a", "b")),
            lambda: self.plot_paired(masked, labels=("a", "b")),
            lambda: self.plot_paired([[1.0, 2.0]], labels=("a",)),
            lambda: self.plot_paired(
                [[1.0, 2.0]], labels=("a", "b"), errors=(0.1, 0.1), error_label="SE"
            ),
            lambda: self.plot_paired([[1.0, 2.0]], labels=("a", "b"), seed=-1),
            lambda: self.plot_paired([[1.0, 2.0]], labels=("a", "b"), jitter=0.5),
        )
        for call in invalid:
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                call()

    def test_effects_supports_samples_bands_annotations_and_asymmetric_errors(self) -> None:
        axes = self.plot_effects(
            (0.4, -0.2),
            ("first", "second"),
            errors=((0.1, 0.2), (0.3, 0.4)),
            error_label="95% CI",
            samples=((0.2, 0.3, 0.5), (-0.5, -0.1)),
            annotations=("q=.01", "q=.20"),
            row_bands=True,
        )
        self.assertEqual([tick.get_text() for tick in axes.get_yticklabels()], ["first", "second"])
        # Two half-density envelopes, two raw scatters and two interval collections.
        self.assertEqual(len(axes.collections), 6)
        self.assertEqual([text.get_text() for text in axes.texts], ["q=.01", "q=.20"])
        self.assertEqual(axes.texts[0].get_position()[0], 1.01)
        self.assertIn("95% CI", axes.get_legend_handles_labels()[1])
        first_interval = axes.containers[0].lines[2][0].get_segments()[0]
        second_interval = axes.containers[1].lines[2][0].get_segments()[0]
        np.testing.assert_allclose(first_interval[:, 0], (0.3, 0.7))
        np.testing.assert_allclose(second_interval[:, 0], (-0.4, 0.2))

    def test_effects_do_not_infer_estimates_from_samples(self) -> None:
        axes = self.plot_effects((10.0,), ("effect",), samples=((1.0, 2.0, 3.0),))
        estimate_lines = [line for line in axes.lines if line.get_marker() == "D"]
        self.assertEqual(len(estimate_lines), 1)
        self.assertAlmostEqual(float(estimate_lines[0].get_xdata()[0]), 10.0)
        self.assertEqual(estimate_lines[0].get_markerfacecolor(), "white")

    def test_raincloud_retains_all_observations_and_handles_constant_samples(self) -> None:
        from matplotlib.collections import PathCollection, PolyCollection

        samples = ([0.0, 0.1, 0.2, 100.0], [2.0, 2.0], [3.0])
        axes = self.plot_effects((10.0, 2.0, 3.0), ("varying", "constant", "single"),
                                 samples=samples)
        points = [item for item in axes.collections if isinstance(item, PathCollection)]
        self.assertEqual(len(points), 3)
        for collection, sample in zip(points, samples, strict=True):
            np.testing.assert_array_equal(collection.get_offsets()[:, 0], sample)
        self.assertEqual(sum(isinstance(c, PolyCollection) for c in axes.collections), 1)
        self.assertGreaterEqual(axes.get_xlim()[1], 100.0)

    def test_association_marginals_preserve_points_and_supplied_fit(self) -> None:
        axes = self.plot_association((3.0, 1.0, 2.0), (7.0, 3.0, 5.0),
                                     fit=(2.0, 1.0), marginals=True)
        self.assertEqual(len(axes.child_axes), 2)
        np.testing.assert_array_equal(axes.collections[0].get_offsets(),
                                      [[3, 7], [1, 3], [2, 5]])
        np.testing.assert_array_equal(axes.lines[0].get_ydata(), (3.0, 7.0))
        self.assertTrue(all(len(marginal.collections) == 1 for marginal in axes.child_axes))

    def test_association_only_draws_supplied_fit(self) -> None:
        axes = self.plot_association((3.0, 1.0, 2.0), (7.0, 3.0, 5.0))
        self.assertEqual(len(axes.lines), 0)
        axes = self.plot_association((3.0, 1.0, 2.0), (7.0, 3.0, 5.0), fit=(2.0, 1.0))
        np.testing.assert_array_equal(axes.lines[0].get_xdata(), (1.0, 3.0))
        np.testing.assert_array_equal(axes.lines[0].get_ydata(), (3.0, 7.0))
        with self.assertRaisesRegex(ValueError, "distinct x"):
            self.plot_association((1.0, 1.0), (2.0, 3.0), fit=(1.0, 0.0))

    def test_curve_preserves_order_and_names_uncertainty(self) -> None:
        axes = self.plot_curve(
            (3.0, 1.0, 2.0),
            (0.3, 0.1, 0.2),
            errors=(0.03, 0.01, 0.02),
            error_label="SE",
            label="Validation error",
        )
        np.testing.assert_array_equal(axes.lines[0].get_xdata(), (3.0, 1.0, 2.0))
        self.assertIn("Validation error (SE)", axes.get_legend_handles_labels()[1])

    def test_uncertainty_requires_valid_distances_and_label(self) -> None:
        invalid = (
            lambda: self.plot_curve((1.0, 2.0), (3.0, 4.0), errors=(0.1, 0.2)),
            lambda: self.plot_curve(
                (1.0, 2.0), (3.0, 4.0), errors=(-0.1, 0.2), error_label="SE"
            ),
            lambda: self.plot_curve(
                (1.0, 2.0), (3.0, 4.0), errors=(np.inf, 0.2), error_label="SE"
            ),
            lambda: self.plot_effects(
                (1.0, 2.0), ("a", "b"), errors=((0.1,), (0.2,)), error_label="CI"
            ),
        )
        for call in invalid:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_null_distribution_counts_ties_in_requested_tail(self) -> None:
        null = np.array([-2.0, -1.0, 0.0, 1.0, 1.0, 2.0])
        axes = self.plot_null_distribution(null, 1.0, tail="greater", bins=4)
        self.assertEqual(sum(patch.get_height() for patch in axes.patches[:4]), null.size)
        self.assertEqual(sum(patch.get_height() for patch in axes.patches[4:]), 3)
        self.assertEqual(axes.get_ylabel(), "Null draws (count)")
        self.assertEqual(len(axes.lines), 1)

    def test_null_distribution_uses_absolute_two_sided_convention(self) -> None:
        axes = self.plot_null_distribution((-3.0, -1.0, 0.0, 2.0), 2.0, tail="two-sided", bins=4)
        self.assertEqual(sum(patch.get_height() for patch in axes.patches[4:]), 2)
        for invalid in (0, -1, 2.5, True):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                self.plot_null_distribution((0.0, 1.0), 1.0, bins=invalid)
        with self.assertRaisesRegex(ValueError, "tail"):
            self.plot_null_distribution((0.0, 1.0), 1.0, tail="upper")

    def test_null_distribution_less_tail_and_no_extremes(self) -> None:
        axes = self.plot_null_distribution((-2.0, -1.0, 0.0, 1.0), -1.0, tail="less", bins=4)
        self.assertEqual(sum(patch.get_height() for patch in axes.patches[4:]), 2)
        axes = self.plot_null_distribution((0.0, 1.0), 2.0, tail="greater", bins=2)
        self.assertEqual(sum(patch.get_height() for patch in axes.patches[2:]), 0)
        self.assertIn("Extreme null draws", axes.get_legend_handles_labels()[1])

    def test_each_plot_exports_vector_and_raster(self) -> None:
        axes_values = (
            self.plot_paired([[1.0, 2.0]], labels=("a", "b")),
            self.plot_effects((0.2,), ("effect",)),
            self.plot_association((0.0, 1.0), (1.0, 2.0)),
            self.plot_curve((1.0, 2.0), (2.0, 1.0)),
            self.plot_null_distribution((-1.0, 0.0, 1.0), 0.5),
        )
        for axes in axes_values:
            for format_name in ("svg", "pdf", "png"):
                with self.subTest(plot=axes, format_name=format_name):
                    output = io.BytesIO()
                    axes.figure.savefig(output, format=format_name)
                    self.assertGreater(output.tell(), 100)


if __name__ == "__main__":
    unittest.main()
