import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dfckit.io.state_nested_checkpoint import (
    load_nested_state_count_checkpoint,
    write_nested_state_count_checkpoint,
)


def _payload() -> dict[str, object]:
    return {
        "format": "dfckit-nested-state-count-checkpoint",
        "format_version": 1,
        "method": "kmeans",
        "model_kind": "kmeans-state",
        "source_contract": "checkpoint-test:v1",
        "sample_interval_seconds": 0.8,
        "feature_contract_fingerprint": "a" * 64,
        "cohort_data_fingerprint": "b" * 64,
        "subjects": ["sub-001", "sub-002", "sub-003"],
        "candidate_n_states": [2, 3],
        "model_seeds": [17, 29],
        "selection_policy": "one-standard-error",
        "fit_configuration": {
            "batch_size": 16,
            "init_sample_size": None,
            "max_iter": 1,
            "n_init": 1,
            "reassignment_ratio": 0.01,
            "standardize_features": True,
        },
        "outer_split": {
            "algorithm": "sha256-seed-subject-balanced-v1",
            "n_folds": 3,
            "seed": 303,
        },
        "inner_validation": {"n_folds": 2, "split_seed": 101},
    }


class NestedCheckpointIOTests(unittest.TestCase):
    def test_roundtrip_and_overwrite_protection(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            write_nested_state_count_checkpoint(_payload(), path)
            self.assertEqual(load_nested_state_count_checkpoint(path), _payload())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_nested_state_count_checkpoint(_payload(), path)

    def test_rejects_schema_duplicate_and_nonfinite_values(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            changed = _payload()
            changed["unexpected"] = True
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_nested_state_count_checkpoint(path)

            path.write_text(
                json.dumps(_payload()).replace(
                    "{",
                    '{"format": "duplicate",',
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                load_nested_state_count_checkpoint(path)

            path.write_text(
                json.dumps(_payload()).replace("0.8", "NaN", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-standard JSON"):
                load_nested_state_count_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
