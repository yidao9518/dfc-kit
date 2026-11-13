"""The plotting namespace remains usable in a NumPy-only installation."""

import subprocess
import sys
import textwrap
import unittest


class OptionalPlottingTests(unittest.TestCase):
    def test_import_with_matplotlib_blocked(self):
        code = textwrap.dedent("""
            import builtins
            import sys
            original_import = builtins.__import__
            def without_matplotlib(name, *args, **kwargs):
                if name == 'matplotlib' or name.startswith('matplotlib.'):
                    raise ImportError('Matplotlib intentionally unavailable')
                return original_import(name, *args, **kwargs)
            builtins.__import__ = without_matplotlib
            import dfckit.plotting as plotting
            assert len([name for name in plotting.__all__ if name.startswith('plot_')]) == 7
            assert callable(plotting.publication_style)
            assert 'matplotlib' not in sys.modules
            for call in (
                lambda: plotting.plot_ring_network(['a'], []),
                lambda: plotting.plot_connectivity_matrix([[0.0]], value_limit=1),
                lambda: plotting.plot_paired([[1, 2]], labels=['OFF', 'ON']),
                lambda: plotting.plot_effects([0.5], ['effect']),
                lambda: plotting.plot_association([1, 2], [3, 4]),
                lambda: plotting.plot_curve([1, 2], [3, 4]),
                lambda: plotting.plot_null_distribution([1, 2], 3),
                lambda: plotting.publication_style(),
            ):
                try:
                    call()
                except ImportError as error:
                    assert 'dfc-kit[plotting]' in str(error), str(error)
                else:
                    raise AssertionError('Plotting succeeded without Matplotlib')
        """)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
