import json
import unittest
from pathlib import Path

import numpy as np

from dfckit import TimeSeriesRun
from dfckit.connectivity import ETS, MTD, cross_block_mtd, within_block_mtd

FIXTURE = Path(__file__).parents[1] / "fixtures" / "audited_kernels.json"


class AuditedKernelRegressionTests(unittest.TestCase):
    def test_mtd_and_ets_match_frozen_audited_outputs(self):
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        values = np.vstack([np.asarray(segment, dtype=float) for segment in fixture["segments"]])
        run = TimeSeriesRun(
            values=values,
            original_indices=fixture["original_indices"],
            roi_names=("visual", "motor", "putamen"),
        )

        mtd = MTD().transform(run)
        np.testing.assert_allclose(
            mtd.standardized_derivatives,
            fixture["mtd"]["standardized_derivatives"],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            cross_block_mtd(mtd.standardized_derivatives, [0, 1], [2]),
            fixture["mtd"]["cross_block_visual_motor_to_putamen"],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            within_block_mtd(mtd.standardized_derivatives, [0, 1, 2]),
            fixture["mtd"]["within_all"],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            ETS().rss(run).rss,
            fixture["ets"]["rss"],
            rtol=1e-13,
            atol=1e-13,
        )


if __name__ == "__main__":
    unittest.main()
