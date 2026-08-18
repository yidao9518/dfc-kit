import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dfckit.io.state_cross_validation_checkpoint import (
    load_state_count_checkpoint,
    write_state_count_checkpoint,
)


def _payload() -> dict[str, object]:
    return {
        "format": "dfckit-state-count-cross-validation-checkpoint",
        "format_version": 1,
        "method": "kmeans",
        "model_kind": "kmeans-state",
        "source_contract": "state-count-checkpoint-test:v1",
        "sample_interval_seconds": 0.8,
        "feature_contract_fingerprint": "a" * 64,
        "development_data_fingerprint": "b" * 64,
        "development_subjects": ["sub-001", "sub-002", "sub-003"],
        "candidate_n_states": [2, 3],
        "model_seeds": [17, 29],
        "split": {
            "algorithm": "sha256-seed-subject-balanced-v1",
            "seed": 101,
            "n_folds": 2,
            "folds": [],
        },
        "fit_configuration": {
            "batch_size": 16,
            "init_sample_size": None,
            "max_iter": 1,
            "n_init": 1,
            "reassignment_ratio": 0.01,
            "standardize_features": True,
        },
    }


class StateCountCheckpointIOTests(unittest.TestCase):
    def test_roundtrip_and_overwrite_protection(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            write_state_count_checkpoint(_payload(), path)
            self.assertEqual(load_state_count_checkpoint(path), _payload())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_state_count_checkpoint(_payload(), path)

    def test_rejects_schema_duplicate_and_nonfinite_values(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            changed = _payload()
            changed["unexpected"] = True
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_state_count_checkpoint(path)

            path.write_text(
                json.dumps(_payload()).replace(
                    "{",
                    '{"format": "duplicate",',
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                load_state_count_checkpoint(path)

            path.write_text(
                json.dumps(_payload()).replace("0.8", "NaN", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-standard JSON"):
                load_state_count_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
