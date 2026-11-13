from __future__ import annotations

import io
import unittest

import numpy as np


class ConnectivityMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import matplotlib
        except ImportError:
            raise unittest.SkipTest("matplotlib is not installed") from None
        matplotlib.use("Agg")
        from dfckit.plotting import plot_connectivity_matrix

        cls.plot = staticmethod(plot_connectivity_matrix)

    def tearDown(self) -> None:
        from matplotlib import pyplot as plt

        plt.close("all")

    def test_preserves_sign_order_zero_nan_and_mask(self) -> None:
        source = np.ma.array(
            [[0.0, -0.4, np.nan], [0.2, 0.0, 0.6], [0.1, -0.3, 0.0]],
            mask=[[False, False, False], [False, False, True], [False, False, False]],
        )
        axes = self.plot(source, value_limit=0.8, colorbar=False)
        shown = axes.images[0].get_array()
        expected_mask = np.ma.getmaskarray(source) | np.isnan(source.data)
        np.testing.assert_array_equal(np.ma.getmaskarray(shown), expected_mask)
        np.testing.assert_allclose(shown.data[~expected_mask], source.data[~expected_mask])
        self.assertEqual(shown.data[0, 0], 0.0)
        self.assertLess(shown.data[0, 1], 0.0)
        self.assertGreater(shown.data[1, 0], 0.0)
        self.assertEqual(axes.images[0].norm.vmin, -0.8)
        self.assertEqual(axes.images[0].norm.vmax, 0.8)

    def test_shared_limit_and_optional_colorbar(self) -> None:
        first = self.plot(np.eye(2) * 0.5, value_limit=1.0, colorbar=False)
        second = self.plot(-np.eye(2) * 0.5, value_limit=1.0)
        self.assertEqual((first.images[0].norm.vmin, first.images[0].norm.vmax), (-1, 1))
        self.assertEqual((second.images[0].norm.vmin, second.images[0].norm.vmax), (-1, 1))
        self.assertEqual(len(first.figure.axes), 1)
        self.assertEqual(len(second.figure.axes), 2)

    def test_labels_color_strips_and_group_boundaries(self) -> None:
        axes = self.plot(
            np.zeros((3, 3)),
            value_limit=1.0,
            labels=("V1", "V2", "PUT"),
            node_colors={"V1": "blue", "V2": "blue", "PUT": "purple"},
            group_sizes=(2, 1),
            colorbar=False,
        )
        self.assertEqual([tick.get_text() for tick in axes.get_xticklabels()], ["V1", "V2", "PUT"])
        self.assertEqual([tick.get_text() for tick in axes.get_yticklabels()], ["V1", "V2", "PUT"])
        self.assertEqual(len(axes.images), 1)
        self.assertEqual(len(axes.patches), 6)
        self.assertEqual(len(axes.lines), 2)

    def test_rejects_data_or_metadata_that_change_meaning(self) -> None:
        invalid_calls = (
            lambda: self.plot([], value_limit=1.0),
            lambda: self.plot(np.zeros((2, 3)), value_limit=1.0),
            lambda: self.plot([[0, np.inf], [0, 0]], value_limit=1.0),
            lambda: self.plot([[0, 1.1], [0, 0]], value_limit=1.0),
            lambda: self.plot(np.zeros((2, 2)), value_limit=0),
            lambda: self.plot(np.zeros((2, 2)), value_limit=1, labels=("one",)),
            lambda: self.plot(np.zeros((2, 2)), value_limit=1, node_colors={"a": "red"}),
            lambda: self.plot(
                np.zeros((2, 2)),
                value_limit=1,
                labels=("a", "b"),
                node_colors={"a": "red"},
            ),
            lambda: self.plot(np.zeros((2, 2)), value_limit=1, group_sizes=(1, 2)),
            lambda: self.plot(np.zeros((2, 2)), value_limit=1, group_sizes=(True, 1)),
            lambda: self.plot(
                np.zeros((2, 2)), value_limit=1, colorbar=False, colorbar_label="FC"
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                call()

    def test_exports_vector_and_raster_formats(self) -> None:
        axes = self.plot(
            [[0.0, -0.5], [0.5, 0.0]],
            value_limit=0.5,
            labels=("V1", "PUT"),
            node_colors=("blue", "purple"),
            colorbar_label="Fisher-z",
        )
        for format_name in ("svg", "pdf", "png"):
            with self.subTest(format_name=format_name):
                output = io.BytesIO()
                axes.figure.savefig(output, format=format_name)
                self.assertGreater(output.tell(), 100)

    def test_explicit_asymmetric_scale_keeps_zero_and_original_values(self) -> None:
        from matplotlib.colors import to_rgba

        from dfckit.plotting import NEGATIVE_COLOR, POSITIVE_COLOR

        matrix = np.array([[np.nan, -0.4], [0.8, np.nan]])
        axes = self.plot(matrix, value_limit=1.2, negative_limit=0.5, colorbar=False)
        image = axes.images[0]
        self.assertEqual((image.norm.vmin, image.norm.vmax), (-0.5, 1.2))
        self.assertAlmostEqual(image.norm(0.0), 0.5)
        np.testing.assert_array_equal(image.get_array().compressed(), (-0.4, 0.8))
        np.testing.assert_allclose(image.cmap(0.0), to_rgba(NEGATIVE_COLOR))
        np.testing.assert_allclose(image.cmap(1.0), to_rgba(POSITIVE_COLOR))
        with self.assertRaises(ValueError):
            self.plot(matrix, value_limit=1.2, negative_limit=0.3)


if __name__ == "__main__":
    unittest.main()
