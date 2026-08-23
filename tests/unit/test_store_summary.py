import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.states import FeatureSequence, FeatureSequenceDataset
from dfckit.storage import FeatureStore, summarize_store_statistics


class StoreSummaryTests(unittest.TestCase):
    def test_segments_are_combined_within_acquisition_by_sample_count(self):
        keys = (("visual", "motor"), ("visual", "putamen"))
        sequences = (
            FeatureSequence(
                values=np.asarray([[1.0, 3.0], [3.0, 5.0]]),
                sample_start_indices=[0, 1],
                sample_end_indices=[0, 1],
                feature_keys=keys,
                subject="sub-001",
                session="off",
                acquisition_id="run-1",
                segment_id=0,
                source_contract="mtd:test",
                sample_interval_seconds=1.0,
            ),
            FeatureSequence(
                values=np.asarray([[5.0, 7.0]]),
                sample_start_indices=[3],
                sample_end_indices=[3],
                feature_keys=keys,
                subject="sub-001",
                session="off",
                acquisition_id="run-1",
                segment_id=1,
                source_contract="mtd:test",
                sample_interval_seconds=1.0,
            ),
        )
        with TemporaryDirectory() as temporary:
            store = FeatureStore.create(
                Path(temporary) / "store",
                feature_keys=keys,
                source_contract="mtd:test",
                sample_interval_seconds=1.0,
            )
            store.append_dataset(FeatureSequenceDataset(sequences))
            payload = summarize_store_statistics(store)
        self.assertEqual(payload["n_acquisitions"], 1)
        self.assertEqual(payload["feature_type"], "edge")
        self.assertEqual(payload["rows"][0]["value"], 3.0)
        self.assertEqual(payload["rows"][1]["value"], 5.0)
        self.assertEqual(payload["rows"][0]["n_samples"], 3)

    def test_statistics_include_population_variance_and_extrema(self):
        keys = (("visual", "motor"),)
        sequences = tuple(
            FeatureSequence(
                values=values,
                sample_start_indices=indices,
                sample_end_indices=indices,
                feature_keys=keys,
                subject="sub-001",
                session="off",
                acquisition_id="run-1",
                segment_id=segment_id,
                source_contract="ets:test",
                sample_interval_seconds=1.0,
            )
            for segment_id, values, indices in (
                (0, np.asarray([[1.0], [3.0]]), [0, 1]),
                (1, np.asarray([[5.0]]), [3]),
            )
        )
        with TemporaryDirectory() as temporary:
            store = FeatureStore.create(
                Path(temporary) / "store",
                feature_keys=keys,
                source_contract="ets:test",
                sample_interval_seconds=1.0,
            )
            store.append_dataset(FeatureSequenceDataset(sequences), chunk_size=1)
            payload = summarize_store_statistics(
                store,
                ("mean", "variance", "standard_deviation", "minimum", "maximum"),
            )
        self.assertEqual(payload["format_version"], 2)
        self.assertEqual(
            payload["variance_definition"],
            "population variance across retained samples (ddof=0)",
        )
        observed = {row["statistic"]: row["value"] for row in payload["rows"]}
        self.assertEqual(observed["mean"], 3.0)
        self.assertAlmostEqual(observed["variance"], 8 / 3)
        self.assertAlmostEqual(observed["standard_deviation"], np.sqrt(8 / 3))
        self.assertEqual(observed["minimum"], 1.0)
        self.assertEqual(observed["maximum"], 5.0)

    def test_invalid_statistic_selection_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "sequence"):
            summarize_store_statistics(None, "mean")


if __name__ == "__main__":
    unittest.main()
