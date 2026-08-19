import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.states import FeatureSequence, FeatureSequenceDataset
from dfckit.storage import FeatureStore
from dfckit.store_summary import summarize_store_means


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
            payload = summarize_store_means(store)
        self.assertEqual(payload["n_acquisitions"], 1)
        self.assertEqual(payload["feature_type"], "edge")
        self.assertEqual(payload["rows"][0]["value"], 3.0)
        self.assertEqual(payload["rows"][1]["value"], 5.0)
        self.assertEqual(payload["rows"][0]["n_samples"], 3)


if __name__ == "__main__":
    unittest.main()
