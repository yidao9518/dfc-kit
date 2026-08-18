import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.cli import main
from dfckit.io import load_state_count_cross_validation
from dfckit.states import FeatureSequence, FeatureSequenceDataset
from dfckit.storage import FeatureStore


def _write_store(root: Path) -> FeatureStore:
    feature_keys = tuple((f"feature-{index}",) for index in range(4))
    means = np.asarray([[-2.0, -1.0, 0.5, 1.0], [2.0, 1.0, -0.5, -1.0]])
    latent = np.tile(np.repeat([0, 1], 10), 2)
    sequences = []
    for index in range(4):
        random = np.random.default_rng(900 + index)
        values = means[latent] + random.normal(scale=0.2, size=(len(latent), 4))
        sequences.append(
            FeatureSequence(
                values=values,
                sample_start_indices=np.arange(len(values)),
                sample_end_indices=np.arange(len(values)),
                feature_keys=feature_keys,
                subject=f"sub-{index:03d}",
                session="off",
                acquisition_id=f"sub-{index:03d}_ses-off_task-rest_run-1",
                segment_id=0,
                source_contract="cross-validation-loader-test:v1",
                sample_interval_seconds=0.8,
            )
        )
    store = FeatureStore.create(
        root,
        feature_keys=feature_keys,
        source_contract="cross-validation-loader-test:v1",
        sample_interval_seconds=0.8,
    )
    store.append_dataset(FeatureSequenceDataset(tuple(sequences)), chunk_size=7)
    return store


def _write_workflow(root: Path) -> tuple[FeatureStore, Path]:
    store = _write_store(root / "features.store")
    output = root / "selection"
    status = main(
        [
            "cross-validate-state-counts",
            str(store.root),
            str(output),
            "--method",
            "kmeans",
            "--n-states",
            "2",
            "--n-states",
            "3",
            "--seed",
            "17",
            "--seed",
            "29",
            "--n-folds",
            "2",
            "--split-seed",
            "101",
            "--n-init",
            "1",
            "--max-iter",
            "1",
            "--batch-size",
            "16",
        ]
    )
    if status != 0:
        raise RuntimeError("test cross-validation workflow could not be created")
    return store, output


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class StateCrossValidationIOTests(unittest.TestCase):
    def test_loads_complete_workflow_after_directory_move(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, workflow = _write_workflow(root)
            loaded = load_state_count_cross_validation(workflow)
            self.assertEqual(loaded.method, "kmeans")
            self.assertEqual(loaded.development_subjects, store.subjects)
            self.assertEqual(loaded.candidate_n_states, (2, 3))
            self.assertEqual(loaded.model_seeds, (17, 29))
            self.assertEqual(len(loaded.workflow_fingerprint), 64)

            moved = root / "moved-selection"
            workflow.rename(moved)
            restored = load_state_count_cross_validation(moved)
            self.assertEqual(restored.workflow_fingerprint, loaded.workflow_fingerprint)
            self.assertEqual(restored.selected_n_states(), loaded.selected_n_states())
            self.assertEqual(
                restored.selected_n_states("best"),
                loaded.selected_n_states("best"),
            )

    def test_rejects_workflow_and_comparison_tampering(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, workflow = _write_workflow(root)

            changed = _read_json(workflow / "workflow.json")
            changed["best_n_states"] = 999
            _write_json(workflow / "workflow.json", changed)
            with self.assertRaisesRegex(ValueError, "metadata disagree"):
                load_state_count_cross_validation(workflow)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, workflow = _write_workflow(root)
            changed = _read_json(workflow / "comparison.json")
            changed["candidates"][0]["mean_score"] += 1.0
            _write_json(workflow / "comparison.json", changed)
            with self.assertRaisesRegex(ValueError, "cannot be rebuilt"):
                load_state_count_cross_validation(workflow)

    def test_rejects_model_score_identity_and_grid_misalignment(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, workflow = _write_workflow(root)
            manifest = _read_json(workflow / "workflow.json")
            manifest["fits"][0]["model_fingerprint"] = "0" * 64
            _write_json(workflow / "workflow.json", manifest)
            with self.assertRaisesRegex(ValueError, "identities disagree"):
                load_state_count_cross_validation(workflow)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, workflow = _write_workflow(root)
            manifest = _read_json(workflow / "workflow.json")
            manifest["fits"][0]["n_states"] = 3
            _write_json(workflow / "workflow.json", manifest)
            with self.assertRaisesRegex(ValueError, "grid is incomplete or misordered"):
                load_state_count_cross_validation(workflow)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, workflow = _write_workflow(root)
            manifest = _read_json(workflow / "workflow.json")
            manifest["fits"][0]["score_artifact"] = manifest["fits"][1][
                "score_artifact"
            ]
            _write_json(workflow / "workflow.json", manifest)
            with self.assertRaisesRegex(ValueError, "path is misaligned"):
                load_state_count_cross_validation(workflow)

    def test_rejects_paths_outside_workflow(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, workflow = _write_workflow(root)
            manifest = _read_json(workflow / "workflow.json")
            manifest["comparison_artifact"] = "../comparison.json"
            _write_json(workflow / "workflow.json", manifest)
            with self.assertRaisesRegex(ValueError, "inside the workflow"):
                load_state_count_cross_validation(workflow)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, workflow = _write_workflow(root)
            outside = root / "outside-score.json"
            shutil.copy2(workflow / "scores" / next((workflow / "scores").iterdir()).name, outside)
            manifest = _read_json(workflow / "workflow.json")
            manifest["fits"][0]["score_artifact"] = "../outside-score.json"
            _write_json(workflow / "workflow.json", manifest)
            with self.assertRaisesRegex(ValueError, "inside the workflow"):
                load_state_count_cross_validation(workflow)

    def test_rejects_nonstandard_and_duplicate_json_fields(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, workflow = _write_workflow(root)
            path = workflow / "workflow.json"
            path.write_text(
                path.read_text(encoding="utf-8").replace('"n_fits": 8', '"n_fits": NaN'),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-standard JSON"):
                load_state_count_cross_validation(workflow)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, workflow = _write_workflow(root)
            path = workflow / "workflow.json"
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("{", '{"format": "duplicate",', 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                load_state_count_cross_validation(workflow)


if __name__ == "__main__":
    unittest.main()
