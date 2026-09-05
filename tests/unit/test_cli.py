import argparse
import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

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


def _edge_store(path: Path) -> FeatureStore:
    rng = np.random.default_rng(17)
    keys = (("a", "b"), ("a", "c"), ("b", "c"))
    sequences = []
    for index in range(8):
        subject = f"sub-{index:03d}"
        off = rng.normal(0.0, 0.2, size=(6, 3))
        on = off + rng.normal((0.5, 0.5, 0.0), 0.08, size=(6, 3))
        for session, values in (("off", off), ("on", on)):
            sequences.append(
                FeatureSequence(
                    values=values,
                    sample_start_indices=np.arange(6),
                    sample_end_indices=np.arange(6),
                    feature_keys=keys,
                    subject=subject,
                    session=session,
                    segment_id=0,
                    source_contract="nbs-cli-test:v1",
                    sample_interval_seconds=0.8,
                    acquisition_id=f"{subject}_{session}",
                )
            )
    store = FeatureStore.create(
        path,
        feature_keys=keys,
        source_contract="nbs-cli-test:v1",
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
    def test_every_parser_command_has_exactly_one_handler(self):
        parser = cli.build_parser()
        commands = next(
            action.choices
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(set(commands), set(cli._HANDLERS))

    def test_dispatch_calls_only_the_selected_handler(self):
        for command in cli._HANDLERS:
            with self.subTest(command=command):
                namespace = argparse.Namespace(command=command)
                handlers = {name: Mock(return_value={"command": name}) for name in cli._HANDLERS}
                with (
                    patch.object(cli, "build_parser") as parser,
                    patch.dict(cli._HANDLERS, handlers),
                ):
                    parser.return_value.parse_args.return_value = namespace
                    self.assertEqual(_run([command]), {"command": command})
                for name, handler in handlers.items():
                    if name == command:
                        handler.assert_called_once_with(namespace)
                    else:
                        handler.assert_not_called()

    def test_infer_paired_endpoints_selects_a_declared_family(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            endpoints = root / "endpoints.json"
            rows = []
            for index in range(6):
                subject = f"sub-{index:03d}"
                for endpoint in ("homogeneity", "adjacency_excess", "descriptive"):
                    difference = 1.0 if endpoint != "descriptive" else -0.5
                    rows.extend(
                        [
                            {
                                "subject": subject,
                                "session": "off",
                                "endpoint": endpoint,
                                "value": float(index),
                            },
                            {
                                "subject": subject,
                                "session": "on",
                                "endpoint": endpoint,
                                "value": float(index) + difference,
                            },
                        ]
                    )
            endpoints.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            output = root / "inference.json"
            summary = _run(
                [
                    "infer-paired-endpoints",
                    str(endpoints),
                    str(output),
                    "--condition-a",
                    "on",
                    "--condition-b",
                    "off",
                    "--fdr-family",
                    "temporal homogeneity",
                    "--endpoint",
                    "homogeneity",
                    "--endpoint",
                    "adjacency_excess",
                    "--bootstrap",
                    "50",
                    "--seed",
                    "9",
                    "--exact",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(summary["n_endpoints"], 2)
        self.assertEqual(
            payload["endpoint_selection"]["selected_endpoint_names"],
            ["adjacency_excess", "homogeneity"],
        )
        self.assertEqual(
            [result["endpoint"] for result in payload["results"]],
            ["adjacency_excess", "homogeneity"],
        )

    def test_infer_independent_endpoints_with_covariates(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for group, shift, residuals in (
                ("PD2", 2.0, (0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1)),
                ("HC", 0.0, (-0.1, 0.1, -0.2, 0.2, -0.1, 0.3, 0.0, -0.2)),
            ):
                path = root / f"{group}.json"
                path.write_text(
                    json.dumps(
                        {
                            "format": "test-endpoints",
                            "source_contract": "test:v1",
                            "rows": [
                                {
                                    "subject": f"sub-{index:03d}",
                                    "session": "ses-01",
                                    "endpoint": "feature_0",
                                    "value": float(index) + shift + residuals[index],
                                }
                                for index in range(8)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            covariates = root / "covariates.tsv"
            covariates.write_text(
                "group\tsubject\tage\tmale\n"
                + "".join(
                    f"{group}\tsub-{index:03d}\t{50 + index}\t{index % 2}\n"
                    for group in ("PD2", "HC")
                    for index in range(8)
                ),
                encoding="utf-8",
            )
            output = root / "inference.json"
            summary = _run(
                [
                    "infer-independent-endpoints",
                    str(paths[0]),
                    str(paths[1]),
                    str(output),
                    "--group-a",
                    "PD2",
                    "--group-b",
                    "HC",
                    "--fdr-family",
                    "features",
                    "--covariates",
                    str(covariates),
                    "--covariate",
                    "age",
                    "--covariate",
                    "male",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(summary["n_tested"], 1)
        self.assertEqual(payload["contrast"], "PD2 - HC")
        self.assertEqual(payload["results"][0]["n_group_a"], 8)
        self.assertAlmostEqual(payload["results"][0]["estimate"], 2.0, places=1)

    def test_summarize_then_infer_paired_nbs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = _edge_store(root / "edges")
            endpoints = root / "endpoints.json"
            _run(["summarize-store", str(store.root), str(endpoints)])
            output = root / "nbs.json"
            summary = _run(
                [
                    "infer-paired-nbs",
                    str(endpoints),
                    str(output),
                    "--condition-a",
                    "on",
                    "--condition-b",
                    "off",
                    "--threshold",
                    "2.0",
                    "--permutations",
                    "50",
                    "--seed",
                    "9",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(summary["n_subjects"], 8)
        self.assertEqual(summary["n_edges"], 3)
        self.assertEqual(payload["format"], "dfc-kit-paired-nbs")
        self.assertTrue(payload["results"])

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
            fit_summary = _run(
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
                    "20",
                    "--streaming-tol",
                    "0.001",
                    "--streaming-patience",
                    "2",
                    "--streaming-min-passes",
                    "2",
                ]
            )
            self.assertIsInstance(load_fitted_model(model_path), KMeansStateModel)
            self.assertTrue(fit_summary["converged"])
            self.assertLess(fit_summary["passes_completed"], 20)
            self.assertEqual(len(fit_summary["initialization_passes"]), 10)

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
from dfckit import cli
