import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.artifacts._fields import (
    artifact_finite_float,
    artifact_integer,
    artifact_integer_grid,
    sample_intervals_match,
)
from dfckit.artifacts._json import (
    load_json_object,
    nonstandard_constant_hook,
    strict_object_hook,
    write_json_atomic,
)
from dfckit.artifacts._numpy import load_numpy_artifact, write_numpy_artifact


class ArtifactFieldValidationTests(unittest.TestCase):
    def test_integer_and_grid_reject_boolean_or_unsorted_values(self):
        self.assertEqual(artifact_integer(np.int64(2), "count", minimum=1), 2)
        with self.assertRaisesRegex(ValueError, "integer of at least 1"):
            artifact_integer(True, "count", minimum=1)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            artifact_integer_grid([2, 1], "states", minimum=1, minimum_count=2)

    def test_float_and_interval_validation(self):
        self.assertEqual(artifact_finite_float(0.8, "TR", positive=True), 0.8)
        self.assertTrue(sample_intervals_match(0.8, 0.8))
        self.assertFalse(sample_intervals_match(float("nan"), 0.8))

class StrictJSONHookTests(unittest.TestCase):
    def test_duplicate_fields_and_nonstandard_constants_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON field in test artifact"):
            json.loads('{"x": 1, "x": 2}', object_pairs_hook=strict_object_hook("test artifact"))
        with self.assertRaisesRegex(ValueError, "non-standard JSON constant"):
            json.loads("NaN", parse_constant=nonstandard_constant_hook("test artifact"))

    def test_atomic_writer_and_strict_loader_roundtrip(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            self.assertEqual(write_json_atomic(path, {"value": 3}), path)
            self.assertEqual(load_json_object(path, context="test artifact"), {"value": 3})
            with self.assertRaises(FileExistsError):
                write_json_atomic(path, {"value": 4})
            write_json_atomic(path, {"value": 4}, overwrite=True)
            self.assertEqual(load_json_object(path, context="test artifact"), {"value": 4})


class NumPyArtifactTests(unittest.TestCase):
    def test_common_reader_preserves_array_dtype(self):
        with TemporaryDirectory() as temporary:
            target = write_numpy_artifact(
                Path(temporary) / "result",
                {"array_names": ["indices"], "format": "test", "format_version": 1},
                {"indices": np.asarray([1, 2], dtype=np.int64)},
                label="test",
            )
            manifest, arrays = load_numpy_artifact(
                target,
                label="test",
                manifest_fields={"array_names", "format", "format_version"},
            )
            self.assertEqual(manifest["format"], "test")
            self.assertEqual(arrays["indices"].dtype, np.dtype(np.int64))

    def test_common_reader_rejects_archive_manifest_mismatch(self):
        with TemporaryDirectory() as temporary:
            target = write_numpy_artifact(
                Path(temporary) / "result",
                {"array_names": ["values"], "format": "test"},
                {"values": np.asarray([1.0])},
                label="test",
            )
            np.savez(target / "arrays.npz", other=np.asarray([1.0]))
            with self.assertRaisesRegex(ValueError, "arrays do not match"):
                load_numpy_artifact(
                    target,
                    label="test",
                    manifest_fields={"array_names", "format"},
                )


if __name__ == "__main__":
    unittest.main()
