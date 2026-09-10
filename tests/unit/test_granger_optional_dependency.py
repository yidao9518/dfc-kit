import subprocess
import sys
import textwrap
import unittest


class GrangerOptionalDependencyTests(unittest.TestCase):
    def test_connectivity_imports_without_scipy_and_granger_call_explains_extra(self):
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class BlockScipy(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "scipy" or fullname.startswith("scipy."):
                        raise ModuleNotFoundError("scipy blocked for optional-dependency test")
                    return None

            sys.meta_path.insert(0, BlockScipy())
            from dfckit.connectivity import conditional_granger
            from dfckit.data import TimeSeriesRun
            import numpy as np

            rng = np.random.default_rng(7)
            values = rng.normal(size=(40, 2))
            run = TimeSeriesRun(values, np.arange(40), ("source", "target"))
            try:
                conditional_granger(run, source=[0], target=[1], lag_order=1)
            except ModuleNotFoundError as error:
                print(error)
            else:
                raise AssertionError("conditional_granger unexpectedly succeeded without scipy")
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pip install 'dfc-kit[inference]'", result.stdout)


if __name__ == "__main__":
    unittest.main()
