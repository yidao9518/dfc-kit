import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit import TimeSeriesRun
from dfckit.connectivity import ETS, MTD, SlidingWindowFC
from dfckit.states import FeatureSequence, FeatureSequenceDataset, window_fc_sequences
from dfckit.storage import (
    FeatureStore,
    append_mtd,
    append_window_fc,
    write_ets_store,
    write_mtd_store,
    write_window_fc_store,
)


class FeatureStoreTests(unittest.TestCase):
    def setUp(self):
        self.keys = (("a",), ("b",), ("c",))
        self.first = FeatureSequence(
            values=np.arange(30, dtype=float).reshape(10, 3),
            sample_start_indices=np.arange(0, 20, 2),
            sample_end_indices=np.arange(0, 20, 2) + 1,
            feature_keys=self.keys,
            subject="sub-001",
            session="off",
            segment_id=0,
            source_contract="synthetic:v1",
            sample_interval_seconds=1.6,
        )
        self.second = FeatureSequence(
            values=np.arange(18, dtype=float).reshape(6, 3) + 100.0,
            sample_start_indices=np.arange(30, 42, 2),
            sample_end_indices=np.arange(30, 42, 2) + 1,
            feature_keys=self.keys,
            subject="sub-002",
            session="on",
            segment_id=1,
            source_contract="synthetic:v1",
            sample_interval_seconds=1.6,
        )

    def test_append_reopen_mmap_slice_and_dataset_roundtrip(self):
        with TemporaryDirectory() as temporary:
            store = FeatureStore.create(
                Path(temporary) / "features",
                feature_keys=self.keys,
                source_contract="synthetic:v1",
                sample_interval_seconds=1.6,
            )
            store.append_dataset(
                FeatureSequenceDataset((self.first, self.second)),
                chunk_size=4,
            )
            reopened = FeatureStore.open(store.root)

            self.assertEqual(reopened.n_samples, 16)
            self.assertEqual(reopened.n_sequences, 2)
            self.assertEqual(reopened.n_chunks, 5)
            self.assertEqual(reopened.subjects, ("sub-001", "sub-002"))
            self.assertEqual(
                reopened.sequence_sample_counts,
                ((('sub-001', 'off', None, 0), 10), (('sub-002', 'on', None, 1), 6)),
            )
            chunks = list(reopened.iter_chunks(subjects=("sub-001",)))
            self.assertEqual([len(chunk.values) for chunk in chunks], [4, 4, 2])
            self.assertTrue(all(isinstance(chunk.values, np.memmap) for chunk in chunks))

            sliced = reopened.read_sequence("sub-001", "off", 0, sample_slice=slice(3, 8))
            np.testing.assert_array_equal(sliced.values, self.first.values[3:8])
            np.testing.assert_array_equal(
                sliced.sample_start_indices, self.first.sample_start_indices[3:8]
            )
            selected = reopened.read_dataset(subjects=("sub-002",))
            self.assertEqual(selected.subjects, ("sub-002",))
            np.testing.assert_array_equal(selected.sequences[0].values, self.second.values)

    def test_same_subject_session_segments_can_be_separated_by_acquisition_id(self):
        second_acquisition = FeatureSequence(
            values=self.first.values + 1000.0,
            sample_start_indices=self.first.sample_start_indices,
            sample_end_indices=self.first.sample_end_indices,
            feature_keys=self.keys,
            subject="sub-001",
            session="off",
            acquisition_id="task-rest_run-2",
            segment_id=0,
            source_contract="synthetic:v1",
            sample_interval_seconds=1.6,
        )
        first_acquisition = FeatureSequence(
            values=self.first.values,
            sample_start_indices=self.first.sample_start_indices,
            sample_end_indices=self.first.sample_end_indices,
            feature_keys=self.keys,
            subject="sub-001",
            session="off",
            acquisition_id="task-rest_run-1",
            segment_id=0,
            source_contract="synthetic:v1",
            sample_interval_seconds=1.6,
        )
        with TemporaryDirectory() as temporary:
            store = FeatureStore.create(
                Path(temporary) / "features",
                feature_keys=self.keys,
                source_contract="synthetic:v1",
                sample_interval_seconds=1.6,
            )
            store.append_dataset(
                FeatureSequenceDataset((first_acquisition, second_acquisition)),
            )
            self.assertEqual(store.n_sequences, 2)
            self.assertEqual(
                store.sequence_identities,
                (
                    ("sub-001", "off", "task-rest_run-1", 0),
                    ("sub-001", "off", "task-rest_run-2", 0),
                ),
            )
            loaded = store.read_sequence(
                "sub-001", "off", 0, acquisition_id="task-rest_run-2"
            )
            np.testing.assert_array_equal(loaded.values, second_acquisition.values)

    def test_v1_manifest_reads_and_upgrades_on_append(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "features"
            store = FeatureStore.create(
                root,
                feature_keys=self.keys,
                source_contract="synthetic:v1",
                sample_interval_seconds=1.6,
            )
            store.append_sequence(self.first, chunk_size=4)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            manifest["format_version"] = 1
            for sequence in manifest["sequences"]:
                sequence.pop("acquisition_id")
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            legacy = FeatureStore.open(root)
            self.assertEqual(
                legacy.sequence_identities,
                (("sub-001", "off", None, 0),),
            )
            upgraded_sequence = FeatureSequence(
                values=self.first.values + 2000.0,
                sample_start_indices=self.first.sample_start_indices,
                sample_end_indices=self.first.sample_end_indices,
                feature_keys=self.keys,
                subject="sub-001",
                session="off",
                acquisition_id="task-rest_run-2",
                segment_id=0,
                source_contract="synthetic:v1",
                sample_interval_seconds=1.6,
            )
            legacy.append_sequence(upgraded_sequence, chunk_size=4)
            upgraded_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(upgraded_manifest["format_version"], 2)
            self.assertTrue(all("acquisition_id" in item for item in upgraded_manifest["sequences"]))

    def test_append_is_atomic_for_invalid_parts_and_rejects_duplicates(self):
        with TemporaryDirectory() as temporary:
            store = FeatureStore.create(
                Path(temporary) / "features",
                feature_keys=self.keys,
                source_contract="synthetic:v1",
                sample_interval_seconds=1.6,
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                store.append_sequence_parts(
                    [(np.full((2, 3), np.nan), [0, 1], [0, 1])],
                    subject="sub-bad",
                    session=None,
                    segment_id=0,
                )
            self.assertEqual(store.n_chunks, 0)
            self.assertEqual(list((store.root / "chunks").iterdir()), [])

            store.append_sequence(self.first, chunk_size=4)
            with self.assertRaisesRegex(ValueError, "already contains"):
                store.append_sequence(self.first, chunk_size=4)
            self.assertEqual(store.n_sequences, 1)

    def test_manifest_and_contract_corruption_are_detected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "features"
            store = FeatureStore.create(
                root,
                feature_keys=self.keys,
                source_contract="synthetic:v1",
                sample_interval_seconds=None,
            )
            with self.assertRaisesRegex(ValueError, "source contracts"):
                store.require_contract(
                    feature_keys=self.keys,
                    source_contract="other",
                    sample_interval_seconds=None,
                )
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            manifest["format_version"] = 999
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported"):
                FeatureStore.open(root)

    def test_data_fingerprint_is_chunk_invariant_and_content_sensitive(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            stores = []
            for name, chunk_size in (("left", 3), ("right", 4)):
                store = FeatureStore.create(
                    root / name,
                    feature_keys=self.keys,
                    source_contract="synthetic:v1",
                    sample_interval_seconds=1.6,
                )
                store.append_dataset(
                    FeatureSequenceDataset((self.first, self.second)),
                    chunk_size=chunk_size,
                )
                stores.append(store)
            left, right = stores
            self.assertEqual(left.data_fingerprint(), right.data_fingerprint())
            self.assertEqual(
                left.feature_contract_fingerprint(),
                right.feature_contract_fingerprint(),
            )
            self.assertEqual(len(left.feature_contract_fingerprint()), 64)
            self.assertEqual(len(left.data_fingerprint()), 64)
            self.assertEqual(
                left.data_fingerprint(subjects=("sub-001",)),
                left.data_fingerprint(minimum_sequence_length=8),
            )
            changed = FeatureStore.create(
                root / "changed",
                feature_keys=self.keys,
                source_contract="synthetic:v1",
                sample_interval_seconds=1.6,
            )
            changed_first = FeatureSequence(
                values=self.first.values + 0.01,
                sample_start_indices=self.first.sample_start_indices,
                sample_end_indices=self.first.sample_end_indices,
                feature_keys=self.keys,
                subject="sub-001",
                session="off",
                segment_id=0,
                source_contract="synthetic:v1",
                sample_interval_seconds=1.6,
            )
            changed.append_dataset(
                FeatureSequenceDataset((changed_first, self.second)),
                chunk_size=3,
            )
            self.assertNotEqual(left.data_fingerprint(), changed.data_fingerprint())
            self.assertEqual(
                left.feature_contract_fingerprint(),
                changed.feature_contract_fingerprint(),
            )
            different_contract = FeatureStore.create(
                root / "different-contract",
                feature_keys=self.keys,
                source_contract="synthetic:v2",
                sample_interval_seconds=1.6,
            )
            self.assertNotEqual(
                left.feature_contract_fingerprint(),
                different_contract.feature_contract_fingerprint(),
            )


class StreamingEstimatorStoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(73)
        self.run = TimeSeriesRun(
            values=rng.normal(size=(24, 5)),
            original_indices=np.r_[np.arange(12), np.arange(16, 28)],
            roi_names=("a", "b", "c", "d", "e"),
            subject="sub-073",
            session="off",
            tr=0.8,
        )

    def test_streamed_window_fc_matches_materialized_result(self):
        estimator = SlidingWindowFC(length=6, step=2)
        expected = estimator.transform(self.run)
        with TemporaryDirectory() as temporary:
            store = write_window_fc_store(
                Path(temporary) / "window-fc",
                (self.run,),
                estimator,
                chunk_size=2,
            )
            observed = store.read_dataset()
            reconstructed = window_fc_sequences((expected,))

            self.assertEqual(store.n_chunks, 4)
            self.assertEqual(observed.source_contract, reconstructed.source_contract)
            self.assertEqual(len(observed.sequences), 2)
            for left, right in zip(
                observed.sequences, reconstructed.sequences, strict=True
            ):
                np.testing.assert_array_equal(left.values, right.values)
                np.testing.assert_array_equal(
                    left.sample_start_indices, right.sample_start_indices
                )
                np.testing.assert_array_equal(left.sample_end_indices, right.sample_end_indices)

            with self.assertRaisesRegex(ValueError, "already contains"):
                append_window_fc(store, self.run, estimator, chunk_size=2)

    def test_streamed_ets_matches_materialized_result(self):
        expected = ETS().transform(self.run)
        with TemporaryDirectory() as temporary:
            store = write_ets_store(
                Path(temporary) / "ets",
                (self.run,),
                chunk_size=5,
            )
            observed = store.read_dataset()
            features = np.concatenate([sequence.values for sequence in observed.sequences])
            starts = np.concatenate(
                [sequence.sample_start_indices for sequence in observed.sequences]
            )

            np.testing.assert_array_equal(features, expected.features)
            np.testing.assert_array_equal(starts, expected.original_indices)
            self.assertTrue(all(chunk.values.shape[0] <= 5 for chunk in store.iter_chunks()))

    def test_streamed_mtd_matches_materialized_result_and_preserves_gap_segments(self):
        expected = MTD().transform(self.run)
        with TemporaryDirectory() as temporary:
            store = write_mtd_store(
                Path(temporary) / "mtd",
                (self.run,),
                chunk_size=5,
            )
            observed = store.read_dataset()
            features = np.concatenate([sequence.values for sequence in observed.sequences])
            starts = np.concatenate(
                [sequence.sample_start_indices for sequence in observed.sequences]
            )
            ends = np.concatenate(
                [sequence.sample_end_indices for sequence in observed.sequences]
            )

            np.testing.assert_allclose(features, expected.features)
            np.testing.assert_array_equal(starts, expected.start_frames)
            np.testing.assert_array_equal(ends, expected.end_frames)
            self.assertEqual(store.source_contract, "mtd:difference=within-segment;normalization=run")
            self.assertEqual(store.sequence_identities[0][3], 0)
            self.assertEqual(store.sequence_identities[1][3], 1)
            self.assertTrue(all(chunk.values.shape[0] <= 5 for chunk in store.iter_chunks()))

            with self.assertRaisesRegex(ValueError, "already contains"):
                append_mtd(store, self.run, chunk_size=5)


if __name__ == "__main__":
    unittest.main()
