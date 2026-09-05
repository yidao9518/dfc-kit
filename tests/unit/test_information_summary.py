import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit import TimeSeriesDataset, TimeSeriesRun
from dfckit.inference.endpoints import _write_paired_endpoint_inference
from dfckit.information import InformationGroups, compute_fixed_information, save_fixed_information
from dfckit.information.summary import summarize_information_artifact, write_information_summary


class InformationSummaryTests(unittest.TestCase):
    def test_endpoint_writers_preserve_no_overwrite_and_clean_up_failed_json(self):
        for writer in (write_information_summary, _write_paired_endpoint_inference):
            with self.subTest(writer=writer.__name__), TemporaryDirectory() as temporary:
                root = Path(temporary) / "nested"
                path = root / "result.json"
                with self.assertRaises(ValueError):
                    writer({"value": float("nan")}, path)
                self.assertFalse(path.exists())
                self.assertEqual(list(root.iterdir()), [])
                self.assertEqual(writer({"value": 3}, path), path)
                with self.assertRaisesRegex(FileExistsError, "output already exists"):
                    writer({"value": 4}, path)
                self.assertEqual(json.loads(path.read_text()), {"value": 3})
                link = root / "broken-link.json"
                link.symlink_to(root / "missing.json")
                with self.assertRaises(FileExistsError):
                    writer({"value": 5}, link)
                self.assertTrue(link.is_symlink())

    def test_mi_and_cmi_are_extracted_per_acquisition_and_length(self):
        run = TimeSeriesRun(
            values=np.random.default_rng(4).normal(size=(30, 3)),
            original_indices=np.arange(30),
            roi_names=("visual", "motor", "putamen"),
            subject="sub-001",
            session="off",
            acquisition_id="run-1",
            tr=1.0,
        )
        artifact = compute_fixed_information(
            TimeSeriesDataset((run,)),
            InformationGroups(
                left=("visual",),
                right=("putamen",),
                conditioning=("motor",),
            ),
            lengths=(20,),
            draws=2,
            sample_seed=3,
            k=2,
            jitter=1e-10,
            jitter_seed=5,
            standardize=True,
        )
        with TemporaryDirectory() as temporary:
            root = save_fixed_information(artifact, Path(temporary) / "information")
            payload = summarize_information_artifact(root)
        self.assertEqual(
            [row["endpoint"] for row in payload["rows"]],
            [
                "mean_mi.length_20",
                "mean_cmi.length_20",
            ],
        )
        self.assertEqual(payload["groups"]["conditioning"], ["motor"])

    def test_unavailable_cmi_is_retained_as_an_explicit_missing_endpoint(self):
        run = TimeSeriesRun(
            values=np.random.default_rng(5).normal(size=(24, 2)),
            original_indices=np.arange(24),
            roi_names=("visual", "putamen"),
            subject="sub-001",
            session="off",
            acquisition_id="run-1",
            tr=1.0,
        )
        artifact = compute_fixed_information(
            TimeSeriesDataset((run,)),
            InformationGroups(left=("visual",), right=("putamen",), conditioning=None),
            lengths=(16,),
            draws=2,
            sample_seed=3,
            k=2,
            jitter=1e-10,
            jitter_seed=5,
            standardize=True,
        )
        with TemporaryDirectory() as temporary:
            root = save_fixed_information(artifact, Path(temporary) / "information")
            payload = summarize_information_artifact(root)
        cmi = next(row for row in payload["rows"] if row["measure"] == "mean_cmi")
        self.assertIsNone(cmi["value"])


if __name__ == "__main__":
    unittest.main()
