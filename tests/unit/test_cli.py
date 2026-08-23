import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfckit.artifacts import load_fitted_model, load_state_model_scores, load_state_predictions
from dfckit.cli import main
from dfckit.states import FeatureSequence, FeatureSequenceDataset, KMeansStateModel
from dfckit.storage import FeatureStore


def _store(path: Path) -> FeatureStore:
    rng = np.random.default_rng(31)
    sequences = []
    for index in range(4):
        values = np.r_[
            rng.normal(-1.0, 0.1, size=(10, 3)),
            rng.normal(1.0, 0.1, size=(10, 3)),
        ]
        sequences.append(
            FeatureSequence(
                values=values,
                sample_start_indices=np.arange(20),
                sample_end_indices=np.arange(20),
                feature_keys=(("a",), ("b",), ("c",)),
                subject=f"sub-{index:03d}",
                session="off",
                segment_id=0,
                source_contract="cli-test:v1",
                sample_interval_seconds=0.8,
            )
        )
    store = FeatureStore.create(
        path,
        feature_keys=sequences[0].feature_keys,
        source_contract="cli-test:v1",
        sample_interval_seconds=0.8,
    )
    store.append_dataset(FeatureSequenceDataset(tuple(sequences)))
    return store


def _run(arguments: list[str]) -> dict[str, object]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = main(arguments)
    if status != 0:
        raise AssertionError(f"CLI returned {status}: {arguments}")
    return json.loads(output.getvalue())


class CLITests(unittest.TestCase):
    def test_summarize_store_selects_requested_statistics(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = _store(root / "features")
            output = root / "summary.json"
            summary = _run(
                [
                    "summarize-store",
                    str(store.root),
                    str(output),
                    "--statistic",
                    "variance",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(summary["statistics"], ["variance"])
        self.assertEqual(payload["statistics"], ["variance"])
        self.assertTrue(all(row["statistic"] == "variance" for row in payload["rows"]))

    def test_fit_predict_and_score_kmeans(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = _store(root / "features")
            model_path = root / "model"
            _run(
                [
                    "fit-states",
                    str(store.root),
                    str(model_path),
                    "--method",
                    "kmeans",
                    "--n-states",
                    "2",
                    "--seed",
                    "17",
                    "--max-iter",
                    "5",
                ]
            )
            self.assertIsInstance(load_fitted_model(model_path), KMeansStateModel)

            predictions_path = root / "predictions"
            _run(
                [
                    "predict-states",
                    str(store.root),
                    str(model_path),
                    str(predictions_path),
                    "--allow-fit-subjects",
                ]
            )
            self.assertEqual(load_state_predictions(predictions_path).n_samples, 80)

            score_path = root / "scores.json"
            _run(
                [
                    "score-states",
                    str(store.root),
                    str(model_path),
                    str(score_path),
                    "--allow-fit-subjects",
                ]
            )
            self.assertEqual(load_state_model_scores(score_path).n_samples, 80)

    def test_align_and_stability_use_seeds_without_content_hashes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = _store(root / "features")
            models = []
            for seed in (17, 29):
                model = root / f"model-{seed}"
                _run(
                    [
                        "fit-states",
                        str(store.root),
                        str(model),
                        "--method",
                        "kmeans",
                        "--n-states",
                        "2",
                        "--seed",
                        str(seed),
                        "--max-iter",
                        "5",
                    ]
                )
                models.append(model)

            alignment = _run(
                [
                    "align-states",
                    str(models[0]),
                    str(models[1]),
                    str(root / "aligned-model"),
                    str(root / "alignment"),
                ]
            )
            self.assertEqual(alignment["reference_seed"], 17)

            stability = _run(
                [
                    "summarize-stability",
                    str(store.root),
                    str(models[0]),
                    str(root / "stability.json"),
                    "--candidate-model",
                    str(models[1]),
                    "--allow-fit-subjects",
                ]
            )
            self.assertEqual(stability["candidate_seeds"], [29])


if __name__ == "__main__":
    unittest.main()
