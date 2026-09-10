import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit import TimeSeriesDataset, TimeSeriesRun
from dfckit.inference import infer_paired_endpoints
from dfckit.states import FeatureSequence, FeatureSequenceDataset
from dfckit.storage import (
    FeatureStore,
    summarize_static_fc_dataset,
    summarize_store_statistics,
)


class StoreSummaryTests(unittest.TestCase):
    def _store(self, root, sequences, keys, *, chunk_size=2, dtype="float64"):
        store = FeatureStore.create(
            root,
            feature_keys=keys,
            source_contract="ets:test",
            sample_interval_seconds=1.0,
            dtype=dtype,
        )
        store.append_dataset(FeatureSequenceDataset(tuple(sequences)), chunk_size=chunk_size)
        return store

    def test_static_fc_dataset_has_named_complete_edge_axis(self):
        runs = tuple(
            TimeSeriesRun(
                values=np.column_stack(
                    (
                        np.arange(6, dtype=float),
                        np.arange(6, dtype=float) + offset,
                        -np.arange(6, dtype=float),
                    )
                ),
                original_indices=np.arange(6),
                roi_names=("visual", "motor", "putamen"),
                subject=f"sub-{index:03d}",
                session="off",
                acquisition_id=f"run-{index}",
            )
            for index, offset in enumerate((0.0, 0.25), start=1)
        )

        payload = summarize_static_fc_dataset(TimeSeriesDataset(runs))

        self.assertEqual(payload["n_features"], 3)
        self.assertEqual(payload["n_acquisitions"], 2)
        self.assertEqual(len(payload["rows"]), 6)
        self.assertEqual(payload["rows"][0]["feature"], ["visual", "motor"])
        self.assertEqual(payload["rows"][0]["n_samples"], 6)

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

    def test_feature_mean_combines_rows_before_acquisition_statistics(self):
        keys = (("a", "b"), ("a", "c"))
        sequences = (
            FeatureSequence(
                values=np.asarray([[2.0, -2.0], [6.0, 2.0]]),
                sample_start_indices=[0, 1],
                sample_end_indices=[0, 1],
                feature_keys=keys,
                subject="sub-001",
                session="off",
                acquisition_id="run-1",
                segment_id=0,
                source_contract="ets:test",
                sample_interval_seconds=1.0,
            ),
            FeatureSequence(
                values=np.asarray([[-4.0, 2.0]]),
                sample_start_indices=[3],
                sample_end_indices=[3],
                feature_keys=keys,
                subject="sub-001",
                session="off",
                acquisition_id="run-1",
                segment_id=1,
                source_contract="ets:test",
                sample_interval_seconds=1.0,
            ),
            FeatureSequence(
                values=np.asarray([[10.0, 14.0]]),
                sample_start_indices=[0],
                sample_end_indices=[0],
                feature_keys=keys,
                subject="sub-001",
                session="on",
                acquisition_id="run-2",
                segment_id=0,
                source_contract="ets:test",
                sample_interval_seconds=1.0,
            ),
            FeatureSequence(
                values=np.asarray([[100.0, 100.0]]),
                sample_start_indices=[0],
                sample_end_indices=[0],
                feature_keys=keys,
                subject="sub-002",
                session="off",
                acquisition_id="run-1",
                segment_id=0,
                source_contract="ets:test",
                sample_interval_seconds=1.0,
            ),
            FeatureSequence(
                values=np.asarray([[-10.0, -14.0]]),
                sample_start_indices=[0],
                sample_end_indices=[0],
                feature_keys=keys,
                subject="sub-001",
                session="off",
                acquisition_id="repeat",
                segment_id=0,
                source_contract="ets:test",
                sample_interval_seconds=1.0,
            ),
        )
        with TemporaryDirectory() as temporary:
            store = self._store(Path(temporary) / "store", sequences, keys, chunk_size=1)
            payload = summarize_store_statistics(
                store,
                ("mean", "standard_deviation"),
                feature_mean="NBS24",
            )

        self.assertEqual(payload["feature_type"], "feature_set")
        self.assertEqual(payload["n_features"], 1)
        self.assertEqual(payload["source_feature_keys"], [["a", "b"], ["a", "c"]])
        self.assertEqual(payload["feature_aggregation"], "equal_weight_mean")
        off = {
            row["statistic"]: row
            for row in payload["rows"]
            if row["subject"] == "sub-001"
            and row["session"] == "off"
            and row["acquisition_id"] == "run-1"
        }
        self.assertEqual(off["mean"]["endpoint"], "NBS24.mean")
        self.assertEqual(off["mean"]["feature"], ["NBS24"])
        self.assertEqual(off["mean"]["n_samples"], 3)
        self.assertAlmostEqual(off["mean"]["value"], 1.0)
        self.assertAlmostEqual(off["standard_deviation"]["value"], np.sqrt(14 / 3))
        on = next(row for row in payload["rows"] if row["session"] == "on")
        self.assertEqual(on["value"], 12.0)
        other_subject = next(
            row
            for row in payload["rows"]
            if row["subject"] == "sub-002" and row["statistic"] == "mean"
        )
        self.assertEqual(other_subject["value"], 100.0)
        repeat = next(row for row in payload["rows"] if row["acquisition_id"] == "repeat")
        self.assertEqual(repeat["value"], -12.0)
        self.assertEqual(payload["n_acquisitions"], 4)

    def test_feature_mean_uses_float64_row_means_and_is_chunk_invariant(self):
        keys = (("a",), ("b",), ("c",))
        sequence = FeatureSequence(
            values=np.asarray(
                [[1e8, 1.0, -1e8], [3.0, 6.0, 9.0]],
                dtype=np.float32,
            ),
            sample_start_indices=[0, 1],
            sample_end_indices=[0, 1],
            feature_keys=keys,
            subject="sub-001",
            session="off",
            acquisition_id="run-1",
            segment_id=0,
            source_contract="ets:test",
            sample_interval_seconds=1.0,
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._store(root / "first", (sequence,), keys, chunk_size=1, dtype="float32")
            second = self._store(root / "second", (sequence,), keys, chunk_size=8, dtype="float32")
            payload_a = summarize_store_statistics(first, feature_mean="NBS24")
            payload_b = summarize_store_statistics(second, feature_mean="NBS24")

        for row_a, row_b in zip(payload_a["rows"], payload_b["rows"], strict=True):
            self.assertEqual(
                {key: value for key, value in row_a.items() if key != "value"},
                {key: value for key, value in row_b.items() if key != "value"},
            )
            np.testing.assert_allclose(row_a["value"], row_b["value"], rtol=1e-14, atol=1e-15)
        self.assertAlmostEqual(payload_a["rows"][0]["value"], (1 / 3 + 6) / 2)

    def test_single_selected_feature_mean_matches_existing_summary(self):
        keys = (("a", "b"), ("a", "c"))
        sequence = FeatureSequence(
            values=np.asarray([[1.0, 20.0], [3.0, 40.0]]),
            sample_start_indices=[0, 1],
            sample_end_indices=[0, 1],
            feature_keys=keys,
            subject="sub-001",
            session="off",
            acquisition_id="run-1",
            segment_id=0,
            source_contract="ets:test",
            sample_interval_seconds=1.0,
        )
        with TemporaryDirectory() as temporary:
            store = self._store(Path(temporary) / "store", (sequence,), keys)
            with self.assertRaises(KeyError):
                store.select_features(feature_keys=(("missing", "edge"),))
            selected = store.select_features(feature_keys=(("a", "b"),))
            ordinary = summarize_store_statistics(selected, ("mean", "standard_deviation"))
            combined = summarize_store_statistics(
                selected,
                ("mean", "standard_deviation"),
                feature_mean="NBS24",
            )

        self.assertEqual(
            [row["value"] for row in combined["rows"]],
            [row["value"] for row in ordinary["rows"]],
        )
        self.assertEqual(combined["source_feature_keys"], [["a", "b"]])

    def test_feature_mean_payload_is_usable_for_paired_endpoint_inference(self):
        keys = (("a", "b"), ("a", "c"))
        sequences = []
        for index in range(3):
            for session, shift in (("off", 0.0), ("on", 1.0)):
                sequences.append(
                    FeatureSequence(
                        values=np.asarray([[index + shift, index + shift + 2.0]]),
                        sample_start_indices=[0],
                        sample_end_indices=[0],
                        feature_keys=keys,
                        subject=f"sub-{index:03d}",
                        session=session,
                        acquisition_id=f"sub-{index:03d}_{session}",
                        segment_id=0,
                        source_contract="ets:test",
                        sample_interval_seconds=1.0,
                    )
                )
        with TemporaryDirectory() as temporary:
            store = self._store(Path(temporary) / "store", sequences, keys)
            payload = summarize_store_statistics(store, feature_mean="NBS24")
        result = infer_paired_endpoints(
            payload,
            condition_a="on",
            condition_b="off",
            fdr_family="NBS24 summary",
            exact=True,
            n_bootstrap=20,
            seed=4,
        )

        self.assertEqual(result["n_endpoints"], 1)
        self.assertEqual(result["results"][0]["endpoint"], "NBS24.mean")
        self.assertEqual(result["results"][0]["n"], 3)
        self.assertEqual(payload["feature_type"], "feature_set")

    def test_feature_mean_name_must_be_nonempty(self):
        for invalid in ("", "   ", 7):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                summarize_store_statistics(None, feature_mean=invalid)


if __name__ == "__main__":
    unittest.main()
