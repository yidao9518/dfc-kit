from __future__ import annotations

import io
import unittest

import numpy as np


class RingNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import matplotlib
        except ImportError:
            raise unittest.SkipTest("matplotlib is not installed") from None
        matplotlib.use("Agg")
        from dfckit.plotting import plot_ring_network

        cls.plot_ring_network = staticmethod(plot_ring_network)

    def tearDown(self) -> None:
        from matplotlib import pyplot as plt

        plt.close("all")

    def test_draws_one_patch_per_node_and_edge(self) -> None:
        axes = self.plot_ring_network(
            ("V1", "V2", "PUT"),
            (("V1", "PUT"), ("V2", "PUT")),
            edge_values=(-0.4, 0.8),
            value_limit=1.0,
            node_colors={"V1": "blue", "V2": "blue", "PUT": "purple"},
        )
        self.assertEqual(len(axes.patches), 5)
        self.assertEqual(len(axes.lines), 3)
        self.assertEqual(axes.get_aspect(), 1.0)

    def test_preserves_node_order(self) -> None:
        axes = self.plot_ring_network(
            ("first", "second"),
            (),
            node_colors=("red", "blue"),
        )
        first, second = axes.patches
        self.assertAlmostEqual(first.theta1, 0.0)
        self.assertAlmostEqual(first.theta2, 180.0)
        self.assertAlmostEqual(second.theta1, 180.0)
        self.assertAlmostEqual(second.theta2, 360.0)

    def test_single_node_and_empty_edges(self) -> None:
        axes = self.plot_ring_network(("only",), (), labels=("Only",))
        self.assertEqual(len(axes.patches), 1)
        self.assertEqual(len(axes.lines), 1)
        self.assertEqual(len(axes.texts), 1)

    def test_rejects_ambiguous_or_unscaled_edges(self) -> None:
        invalid_calls = (
            lambda: self.plot_ring_network(("a", "a"), ()),
            lambda: self.plot_ring_network(("a", "b"), (("a", "missing"),)),
            lambda: self.plot_ring_network(("a", "b"), (("a", "a"),)),
            lambda: self.plot_ring_network(
                ("a", "b"), (("a", "b"), ("b", "a"))
            ),
            lambda: self.plot_ring_network(
                ("a", "b"), (("a", "b"),), edge_values=(0.5,)
            ),
            lambda: self.plot_ring_network(
                ("a", "b"), (("a", "b"),), edge_values=(1.1,), value_limit=1.0
            ),
            lambda: self.plot_ring_network(
                ("a", "b"),
                (("a", "b"),),
                edge_values=(np.nan,),
                value_limit=1.0,
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_requires_complete_colour_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "match node_names exactly"):
            self.plot_ring_network(("a", "b"), (), node_colors={"a": "red"})

    def test_same_system_rois_still_have_separate_segments(self) -> None:
        names = tuple(f"roi-{i}" for i in range(6))
        axes = self.plot_ring_network(names, (), node_colors=("blue",) * 6)
        self.assertEqual(len(axes.patches), 6)
        self.assertEqual(len(axes.lines), 6)
        for i, wedge in enumerate(axes.patches):
            self.assertAlmostEqual((wedge.theta1 + wedge.theta2) / 2, 90 + i * 60)

    def test_vector_and_raster_exports(self) -> None:
        axes = self.plot_ring_network(
            ("a", "b", "c"),
            (("a", "b"), ("a", "c")),
            edge_values=(-0.25, 0.5),
            value_limit=0.5,
            labels=("A", "B", "C"),
        )
        for format_name in ("svg", "pdf", "png"):
            with self.subTest(format_name=format_name):
                output = io.BytesIO()
                axes.figure.savefig(output, format=format_name)
                self.assertGreater(output.tell(), 100)

    def test_unconnected_roi_stays_in_frame_with_pale_colour(self) -> None:
        from matplotlib.colors import to_rgba

        axes = self.plot_ring_network(
            ("a", "b", "unused"), (("a", "b"),),
            node_colors=("red", "blue", "green"), labels=("A", "B", "Unused"),
            label_unconnected=False,
        )
        self.assertEqual(len(axes.patches), 4)
        np.testing.assert_allclose(axes.patches[0].get_facecolor(), to_rgba("red"))
        original = np.array(to_rgba("green"))
        expected = np.r_[1 - 0.28 * (1 - original[:3]), original[3]]
        np.testing.assert_allclose(axes.patches[2].get_facecolor(), expected)
        self.assertEqual([text.get_text() for text in axes.texts], ["A", "B"])

    def test_signed_edges_share_colour_and_width_scale(self) -> None:
        from matplotlib.colors import to_rgba

        from dfckit.plotting import NEGATIVE_COLOR, POSITIVE_COLOR

        axes = self.plot_ring_network(
            ("a", "b", "c"), (("a", "b"), ("b", "c")),
            edge_values=(-0.5, 1.0), value_limit=1.0,
        )
        negative, positive = axes.patches[-2:]
        np.testing.assert_allclose(negative.get_edgecolor()[:3], to_rgba(NEGATIVE_COLOR)[:3])
        np.testing.assert_allclose(positive.get_edgecolor()[:3], to_rgba(POSITIVE_COLOR)[:3])
        self.assertAlmostEqual(negative.get_linewidth(), 1.5)
        self.assertAlmostEqual(positive.get_linewidth(), 2.7)


class PublicationStyleTests(unittest.TestCase):
    def test_style_is_temporary_and_svg_text_is_editable(self) -> None:
        try:
            import matplotlib
        except ImportError:
            self.skipTest("matplotlib is not installed")
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        from dfckit.plotting import plot_association, publication_style

        keys = ("font.size", "axes.grid", "svg.fonttype", "pdf.fonttype")
        before = {key: matplotlib.rcParams[key] for key in keys}
        with publication_style(font_size=8):
            self.assertEqual(matplotlib.rcParams["font.size"], 8)
            self.assertFalse(matplotlib.rcParams["axes.grid"])
            self.assertEqual(matplotlib.rcParams["svg.fonttype"], "none")
            self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
            ax = plot_association((1, 2), (2, 3))
            ax.set_xlabel("Editable axis label")
            output = io.BytesIO()
            ax.figure.savefig(output, format="svg")
            self.assertIn(b"<text", output.getvalue())
            plt.close(ax.figure)
        self.assertEqual({key: matplotlib.rcParams[key] for key in keys}, before)


if __name__ == "__main__":
    unittest.main()
