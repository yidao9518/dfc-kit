import contextlib
import csv
import importlib.util
import io
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import dfckit.cli as cli_module
from dfckit.cli import main
from dfckit.io import (
    fitted_model_fingerprint,
    load_fitted_model,
    load_nested_state_count_cross_validation,
    load_selected_state_count_evaluation,
    load_state_alignment,
    load_state_model_scores,
    load_state_predictions,
)
from dfckit.io.state_nested_lock import (
    acquire_nested_checkpoint_lock,
    nested_checkpoint_lock_path,
)
from dfckit.states import FeatureSequence, FeatureSequenceDataset
from dfckit.states.hmm import GaussianHMMStateModel
from dfckit.states.kmeans import KMeansStateModel
from dfckit.storage import FeatureStore

HAS_HMM_EXTRA = (
    importlib.util.find_spec("hmmlearn") is not None
    and importlib.util.find_spec("sklearn") is not None
)


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


class CLITests(unittest.TestCase):
    def _write_run(self, root: Path, run: int, *, subject: str = "sub-001") -> None:
        func = root / subject / "ses-off" / "func"
        stem = f"{subject}_ses-off_task-rest_run-{run}"
        spatial = f"{stem}_space-MNI152NLin2009cAsym"
        values = np.column_stack(
            (
                np.arange(6, dtype=float) ** 2 + run,
                np.asarray([0.0, 1.0, 0.0, -1.0, 0.0, 1.0]) + run,
            )
        )
        write_tsv(
            func / f"{spatial}_atlas-Example_stat-mean_timeseries.tsv",
            ["visual", "motor"],
            values.tolist(),
        )
        write_tsv(
            func / f"{spatial}_atlas-Example_stat-coverage_bold.tsv",
            ["Node", "coverage"],
            [["visual", 1.0], ["motor", 1.0]],
        )
        write_tsv(
            func / f"{stem}_outliers.tsv",
            ["outlier"],
            [[0], [0], [0], [0], [0], [0]],
        )

    def _write_state_store(self, root: Path, *, n_subjects: int = 4) -> FeatureStore:
        keys = tuple((f"feature-{index}",) for index in range(4))
        means = np.asarray([[-2.0, -1.0, 0.5, 1.0], [2.0, 1.0, -0.5, -1.0]])
        latent = np.tile(np.repeat([0, 1], 10), 2)
        sequences = []
        for index in range(n_subjects):
            rng = np.random.default_rng(900 + index)
            values = means[latent] + rng.normal(scale=0.2, size=(len(latent), 4))
            sequences.append(
                FeatureSequence(
                    values=values,
                    sample_start_indices=np.arange(len(values)),
                    sample_end_indices=np.arange(len(values)),
                    feature_keys=keys,
                    subject=f"sub-{index:03d}",
                    session="off",
                    acquisition_id=f"sub-{index:03d}_ses-off_task-rest_run-1",
                    segment_id=0,
                    source_contract="cli-state-fit:v1",
                    sample_interval_seconds=0.8,
                )
            )
        store = FeatureStore.create(
            root,
            feature_keys=keys,
            source_contract="cli-state-fit:v1",
            sample_interval_seconds=0.8,
        )
        store.append_dataset(FeatureSequenceDataset(tuple(sequences)), chunk_size=7)
        return store

    def test_inspect_and_build_window_store(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "xcpd"
            self._write_run(root, 1)
            self._write_run(root, 2)
            output = Path(temporary) / "window.store"
            roi_file = Path(temporary) / "rois.json"
            roi_file.write_text(json.dumps({"Example": ["visual", "motor"]}), encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "inspect-xcpd",
                        str(root),
                        "--atlas",
                        "Example",
                        "--space",
                        "MNI152NLin2009cAsym",
                    ]
                )
            self.assertEqual(status, 0)
            inspected = json.loads(stdout.getvalue())
            self.assertEqual(inspected["n_acquisitions"], 2)
            self.assertEqual(
                [item["acquisition_id"] for item in inspected["acquisitions"]],
                [
                    "sub-001_ses-off_task-rest_run-1",
                    "sub-001_ses-off_task-rest_run-2",
                ],
            )
            self.assertEqual(
                set(inspected["acquisitions"][0]["files"]),
                {"Example"},
            )
            self.assertTrue(
                inspected["acquisitions"][0]["files"]["Example"]["timeseries"].endswith(
                    "_atlas-Example_stat-mean_timeseries.tsv"
                )
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "build-store",
                        str(root),
                        str(output),
                        "--atlas",
                        "Example",
                        "--space",
                        "MNI152NLin2009cAsym",
                        "--roi-selection",
                        str(roi_file),
                        "--method",
                        "window-fc",
                        "--window-length",
                        "4",
                        "--window-step",
                        "2",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["n_runs"], 2)
            self.assertEqual(summary["n_sequences"], 2)
            self.assertEqual(summary["format_version"], 2)
            store = FeatureStore.open(output)
            self.assertEqual(
                {identity[2] for identity in store.sequence_identities},
                {
                    "sub-001_ses-off_task-rest_run-1",
                    "sub-001_ses-off_task-rest_run-2",
                },
            )

            ets_output = Path(temporary) / "ets.store"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "build-store",
                        str(root),
                        str(ets_output),
                        "--atlas",
                        "Example",
                        "--space",
                        "MNI152NLin2009cAsym",
                        "--roi-selection",
                        str(roi_file),
                        "--method",
                        "ets",
                    ]
                )
            self.assertEqual(status, 0)
            ets_summary = json.loads(stdout.getvalue())
            self.assertEqual(ets_summary["method"], "ets")
            self.assertEqual(ets_summary["n_runs"], 2)
            self.assertEqual(FeatureStore.open(ets_output).n_sequences, 2)

            mtd_output = Path(temporary) / "mtd.store"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "build-store",
                        str(root),
                        str(mtd_output),
                        "--atlas",
                        "Example",
                        "--space",
                        "MNI152NLin2009cAsym",
                        "--roi-selection",
                        str(roi_file),
                        "--method",
                        "mtd",
                        "--chunk-size",
                        "2",
                    ]
                )
            self.assertEqual(status, 0)
            mtd_summary = json.loads(stdout.getvalue())
            self.assertEqual(mtd_summary["method"], "mtd")
            self.assertEqual(mtd_summary["n_runs"], 2)
            self.assertEqual(mtd_summary["n_samples"], 10)
            self.assertEqual(
                FeatureStore.open(mtd_output).source_contract,
                "mtd:difference=within-segment;normalization=run",
            )

            self._write_run(root, 1, subject="sub-002")
            selected_output = Path(temporary) / "selected.store"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "inspect-xcpd",
                        str(root),
                        "--atlas",
                        "Example",
                        "--space",
                        "MNI152NLin2009cAsym",
                        "--subject",
                        "sub-001",
                        "--subject",
                        "sub-002",
                    ]
                )
            self.assertEqual(status, 0)
            selected = json.loads(stdout.getvalue())
            self.assertEqual(selected["n_acquisitions"], 3)
            self.assertEqual(
                {item["subject"] for item in selected["acquisitions"]},
                {"001", "002"},
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "build-store",
                        str(root),
                        str(selected_output),
                        "--atlas",
                        "Example",
                        "--space",
                        "MNI152NLin2009cAsym",
                        "--subject",
                        "sub-001",
                        "--subject",
                        "sub-002",
                        "--roi-selection",
                        str(roi_file),
                        "--method",
                        "window-fc",
                        "--window-length",
                        "4",
                        "--window-step",
                        "2",
                    ]
                )
            self.assertEqual(status, 0)
            selected_summary = json.loads(stdout.getvalue())
            self.assertEqual(selected_summary["subjects"], ["sub-001", "sub-002"])

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "inspect-xcpd",
                        str(root),
                        "--atlas",
                        "Example",
                        "--subject",
                        "sub-001",
                        "--subject",
                        "001",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("must be unique", stderr.getvalue())

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "inspect-xcpd",
                        str(root),
                        "--atlas",
                        "Example",
                        "--subject",
                        "",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("must be non-empty", stderr.getvalue())

    def test_invalid_roi_json_returns_an_error_status(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text("[]", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "build-store",
                        temporary,
                        str(Path(temporary) / "out"),
                        "--atlas",
                        "Example",
                        "--roi-selection",
                        str(path),
                        "--method",
                        "ets",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("ROI selection", stderr.getvalue())

    def test_fit_states_kmeans_writes_roundtrippable_artifact(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            output = root / "kmeans.model"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "fit-states",
                        str(store.root),
                        str(output),
                        "--method",
                        "kmeans",
                        "--n-states",
                        "2",
                        "--seed",
                        "17",
                        "--subject",
                        "sub-000",
                        "--subject",
                        "sub-001",
                        "--n-init",
                        "2",
                        "--max-iter",
                        "2",
                        "--batch-size",
                        "16",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["model_kind"], "kmeans-state")
            self.assertEqual(summary["fit_subjects"], ["sub-000", "sub-001"])
            self.assertEqual(summary["fit_sample_count"], 80)
            self.assertEqual(summary["fit_sequence_count"], 2)
            self.assertEqual(summary["n_states"], 2)
            self.assertEqual(summary["init_sample_size"], 80)
            self.assertEqual(len(summary["training_data_fingerprint"]), 64)
            self.assertIsNone(summary["converged"])
            self.assertIsNone(summary["log_likelihood"])
            restored = load_fitted_model(output)
            self.assertIsInstance(restored, KMeansStateModel)
            self.assertEqual(restored.fit_subjects, ("sub-000", "sub-001"))

            prediction_output = root / "heldout.labels"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "predict-states",
                        str(store.root),
                        str(output),
                        str(prediction_output),
                        "--subject",
                        "sub-002",
                    ]
                )
            self.assertEqual(status, 0)
            prediction_summary = json.loads(stdout.getvalue())
            self.assertEqual(prediction_summary["subjects"], ["sub-002"])
            self.assertEqual(prediction_summary["n_sequences"], 1)

            metrics_output = root / "heldout.metrics.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "summarize-states",
                        str(prediction_output),
                        str(metrics_output),
                    ]
                )
            self.assertEqual(status, 0)
            metrics_summary = json.loads(stdout.getvalue())
            self.assertEqual(metrics_summary["n_runs"], 1)
            metrics = json.loads(metrics_output.read_text(encoding="utf-8"))
            self.assertEqual(metrics["runs"][0]["subject"], "sub-002")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "predict-states",
                        str(store.root),
                        str(output),
                        str(root / "overlap.labels"),
                        "--subject",
                        "sub-000",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("overlap", stderr.getvalue())

    @unittest.skipUnless(HAS_HMM_EXTRA, "requires dfc-kit[hmm]")
    def test_fit_states_hmm_writes_roundtrippable_artifact(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            output = root / "hmm.model"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "fit-states",
                        str(store.root),
                        str(output),
                        "--method",
                        "hmm",
                        "--n-states",
                        "2",
                        "--seed",
                        "19",
                        "--subject",
                        "sub-000",
                        "--subject",
                        "sub-001",
                        "--n-pca-components",
                        "2",
                        "--n-init",
                        "1",
                        "--n-iter",
                        "20",
                        "--pca-batch-size",
                        "16",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["model_kind"], "gaussian-hmm-state")
            self.assertEqual(summary["fit_subjects"], ["sub-000", "sub-001"])
            self.assertEqual(summary["fit_sample_count"], 80)
            self.assertEqual(summary["fit_sequence_count"], 2)
            self.assertEqual(summary["n_pca_components"], 2)
            self.assertEqual(summary["pca_batch_size"], 16)
            self.assertEqual(len(summary["training_data_fingerprint"]), 64)
            self.assertIsInstance(summary["converged"], bool)
            self.assertIsInstance(summary["log_likelihood"], float)
            restored = load_fitted_model(output)
            self.assertIsInstance(restored, GaussianHMMStateModel)
            self.assertEqual(restored.fit_subjects, ("sub-000", "sub-001"))

            prediction_output = root / "heldout-hmm.labels"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "predict-states",
                        str(store.root),
                        str(output),
                        str(prediction_output),
                        "--subject",
                        "sub-002",
                    ]
                )
            self.assertEqual(status, 0)
            prediction_summary = json.loads(stdout.getvalue())
            self.assertEqual(prediction_summary["model_kind"], "gaussian-hmm-state")
            self.assertIsInstance(prediction_summary["log_likelihood"], float)
            predictions = load_state_predictions(prediction_output)
            self.assertEqual(predictions.subjects, ("sub-002",))
            self.assertIsNotNone(predictions.posterior_probabilities)
            assert predictions.posterior_probabilities is not None
            self.assertEqual(predictions.posterior_probabilities[0].shape, (40, 2))

    def test_fit_states_rejects_missing_hmm_pca_and_existing_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "fit-states",
                        str(store.root),
                        str(root / "hmm.model"),
                        "--method",
                        "hmm",
                        "--n-states",
                        "2",
                        "--seed",
                        "19",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("--n-pca-components is required", stderr.getvalue())

            existing = root / "existing.model"
            existing.mkdir()
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "fit-states",
                        str(store.root),
                        str(existing),
                        "--method",
                        "kmeans",
                        "--n-states",
                        "2",
                        "--seed",
                        "17",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("already exists", stderr.getvalue())

    def test_fit_states_rejects_an_unknown_explicit_subject(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            output = root / "kmeans.model"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "fit-states",
                        str(store.root),
                        str(output),
                        "--method",
                        "kmeans",
                        "--n-states",
                        "2",
                        "--seed",
                        "17",
                        "--subject",
                        "sub-missing",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("sub-missing", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_cross_validate_state_counts_runs_complete_kmeans_grid(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            outputs = (root / "selection-a", root / "selection-b")
            summaries = []
            for output in outputs:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
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
                self.assertEqual(status, 0)
                summaries.append(json.loads(stdout.getvalue()))

            self.assertEqual(summaries[0]["n_fits"], 8)
            self.assertEqual(summaries[0]["n_folds"], 2)
            self.assertEqual(summaries[0]["candidate_n_states"], [2, 3])
            self.assertEqual(summaries[0]["model_seeds"], [17, 29])
            self.assertEqual(summaries[0]["n_subjects"], 4)
            self.assertEqual(
                summaries[0]["best_n_states"],
                summaries[1]["best_n_states"],
            )
            self.assertEqual(
                summaries[0]["one_standard_error_n_states"],
                summaries[1]["one_standard_error_n_states"],
            )

            workflow = json.loads(
                (outputs[0] / "workflow.json").read_text(encoding="utf-8")
            )
            comparison = json.loads(
                (outputs[0] / "comparison.json").read_text(encoding="utf-8")
            )
            repeated_comparison = json.loads(
                (outputs[1] / "comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(workflow["format_version"], 1)
            self.assertEqual(workflow["n_fits"], 8)
            self.assertEqual(workflow["development_subjects"], list(store.subjects))
            self.assertEqual(
                workflow["split"]["algorithm"],
                "sha256-seed-subject-balanced-v1",
            )
            evaluated = [
                subject
                for fold in workflow["split"]["folds"]
                for subject in fold["evaluation_subjects"]
            ]
            self.assertEqual(sorted(evaluated), sorted(store.subjects))
            self.assertEqual(len(evaluated), len(set(evaluated)))
            self.assertEqual(comparison, repeated_comparison)
            self.assertEqual(comparison["format_version"], 2)
            self.assertEqual(comparison["n_seeds"], 2)
            self.assertEqual(len(tuple((outputs[0] / "models").iterdir())), 8)
            self.assertEqual(len(tuple((outputs[0] / "scores").iterdir())), 8)
            for fit in workflow["fits"]:
                self.assertTrue((outputs[0] / fit["model_artifact"]).is_dir())
                self.assertTrue((outputs[0] / fit["score_artifact"]).is_file())
                self.assertEqual(len(fit["model_fingerprint"]), 64)
            self.assertNotIn(
                str(outputs[0]),
                (outputs[0] / "workflow.json").read_text(encoding="utf-8"),
            )

            moved = root / "moved-selection"
            outputs[0].rename(moved)
            moved_workflow = json.loads(
                (moved / "workflow.json").read_text(encoding="utf-8")
            )
            for fit in moved_workflow["fits"]:
                model = load_fitted_model(moved / fit["model_artifact"])
                score = load_state_model_scores(moved / fit["score_artifact"])
                self.assertEqual(fitted_model_fingerprint(model), fit["model_fingerprint"])
                self.assertEqual(score.model_fingerprint, fit["model_fingerprint"])

    def test_cross_validate_state_counts_rejects_bad_grid_and_existing_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "cross-validate-state-counts",
                        str(store.root),
                        str(root / "selection"),
                        "--method",
                        "kmeans",
                        "--n-states",
                        "2",
                        "--seed",
                        "17",
                        "--n-folds",
                        "2",
                        "--split-seed",
                        "101",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("at least 2", stderr.getvalue())
            self.assertFalse((root / "selection").exists())

            existing = root / "existing"
            existing.mkdir()
            marker = existing / "preserve.txt"
            marker.write_text("preserve", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "cross-validate-state-counts",
                        str(root / "missing.store"),
                        str(existing),
                        "--method",
                        "kmeans",
                        "--n-states",
                        "2",
                        "--n-states",
                        "3",
                        "--seed",
                        "17",
                        "--n-folds",
                        "2",
                        "--split-seed",
                        "101",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("already exists", stderr.getvalue())
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

            failed = root / "failed-selection"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "cross-validate-state-counts",
                        str(store.root),
                        str(failed),
                        "--method",
                        "kmeans",
                        "--n-states",
                        "2",
                        "--n-states",
                        "1000",
                        "--seed",
                        "17",
                        "--n-folds",
                        "2",
                        "--split-seed",
                        "101",
                        "--n-init",
                        "1",
                        "--max-iter",
                        "1",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("cannot exceed", stderr.getvalue())
            self.assertFalse(failed.exists())
            self.assertEqual(tuple(root.glob(".failed-selection.tmp-*")), ())

    def test_evaluate_selected_state_count_rejects_development_data_drift(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            selection = root / "selection"
            arguments = [
                "cross-validate-state-counts",
                str(store.root),
                str(selection),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
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
            for subject in ("sub-000", "sub-001", "sub-002", "sub-003"):
                arguments.extend(("--subject", subject))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)

            values_path = store.root / "chunks" / "00000000.values.npy"
            values = np.load(values_path, allow_pickle=False)
            changed = values.copy()
            changed[0, 0] += 0.5
            np.save(values_path, changed, allow_pickle=False)

            output = root / "drifted-evaluation"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    main(
                        [
                            "evaluate-selected-state-count",
                            str(store.root),
                            str(selection),
                            str(output),
                            "--test-subject",
                            "sub-004",
                        ]
                    ),
                    2,
                )
            self.assertIn("changed after inner", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_evaluate_selected_state_count_checks_fold_evidence_after_manifest_edit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            selection = root / "selection"
            arguments = [
                "cross-validate-state-counts",
                str(store.root),
                str(selection),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
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
            development = ("sub-000", "sub-001", "sub-002", "sub-003")
            for subject in development:
                arguments.extend(("--subject", subject))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)

            values_path = store.root / "chunks" / "00000000.values.npy"
            values = np.load(values_path, allow_pickle=False)
            changed = values.copy()
            changed[0, 0] += 0.5
            np.save(values_path, changed, allow_pickle=False)
            workflow_path = selection / "workflow.json"
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            workflow["development_data_fingerprint"] = store.data_fingerprint(
                subjects=development
            )
            workflow_path.write_text(
                json.dumps(workflow, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            output = root / "forged-manifest-evaluation"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    main(
                        [
                            "evaluate-selected-state-count",
                            str(store.root),
                            str(selection),
                            str(output),
                            "--test-subject",
                            "sub-004",
                        ]
                    ),
                    2,
                )
            self.assertIn("fold data changed", stderr.getvalue())
            self.assertFalse(output.exists())

    @unittest.skipUnless(HAS_HMM_EXTRA, "requires dfc-kit[hmm]")
    def test_evaluate_selected_state_count_runs_hmm_outer_test(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=6)
            selection = root / "hmm-selection"
            development = ("sub-000", "sub-001", "sub-002", "sub-003")
            arguments = [
                "cross-validate-state-counts",
                str(store.root),
                str(selection),
                "--method",
                "hmm",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--n-folds",
                "2",
                "--split-seed",
                "101",
                "--n-pca-components",
                "2",
                "--n-init",
                "1",
                "--n-iter",
                "20",
                "--pca-batch-size",
                "16",
            ]
            for subject in development:
                arguments.extend(("--subject", subject))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)

            output = root / "hmm-evaluation"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "evaluate-selected-state-count",
                            str(store.root),
                            str(selection),
                            str(output),
                            "--selection",
                            "best",
                            "--test-subject",
                            "sub-004",
                            "--test-subject",
                            "sub-005",
                        ]
                    ),
                    0,
                )
            evaluation = json.loads(
                (output / "evaluation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(evaluation["method"], "hmm")
            self.assertEqual(evaluation["selection_policy"], "best")
            self.assertEqual(evaluation["selection_direction"], "higher is better")
            self.assertEqual(evaluation["test_subjects"], ["sub-004", "sub-005"])
            model = load_fitted_model(output / evaluation["seeds"][0]["model_artifact"])
            report = load_state_model_scores(
                output / evaluation["seeds"][0]["score_artifact"]
            )
            self.assertIsInstance(model, GaussianHMMStateModel)
            self.assertEqual(model.fit_subjects, development)
            self.assertEqual(report.subjects, ("sub-004", "sub-005"))

    def test_cross_validate_state_counts_respects_outer_training_subjects(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            output = root / "outer-training-selection"
            selected = ("sub-000", "sub-001", "sub-002")
            arguments = [
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
            for subject in selected:
                arguments.extend(("--subject", subject))

            self.assertEqual(main(arguments), 0)
            workflow = json.loads(
                (output / "workflow.json").read_text(encoding="utf-8")
            )
            self.assertEqual(workflow["development_subjects"], list(selected))
            self.assertNotIn("sub-003", json.dumps(workflow, sort_keys=True))
            for fold in workflow["split"]["folds"]:
                self.assertEqual(
                    set(fold["fit_subjects"]).union(fold["evaluation_subjects"]),
                    set(selected),
                )

    def test_evaluate_selected_state_count_refits_and_scores_outer_test_subjects(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=6)
            selection = root / "selection"
            development = ("sub-000", "sub-001", "sub-002", "sub-003")
            cross_validation_arguments = [
                "cross-validate-state-counts",
                str(store.root),
                str(selection),
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
            for subject in development:
                cross_validation_arguments.extend(("--subject", subject))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(cross_validation_arguments), 0)

            output = root / "outer-evaluation"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "evaluate-selected-state-count",
                        str(store.root),
                        str(selection),
                        str(output),
                        "--test-subject",
                        "sub-004",
                        "--test-subject",
                        "sub-005",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            evaluation = json.loads(
                (output / "evaluation.json").read_text(encoding="utf-8")
            )
            workflow = json.loads(
                (selection / "workflow.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["selected_n_states"], workflow["one_standard_error_n_states"])
            self.assertEqual(summary["test_subjects"], ["sub-004", "sub-005"])
            self.assertEqual(evaluation["development_subjects"], list(development))
            self.assertEqual(evaluation["test_subjects"], ["sub-004", "sub-005"])
            self.assertEqual(evaluation["model_seeds"], [17, 29])
            self.assertEqual(evaluation["n_models"], 2)
            self.assertNotIn(str(output), json.dumps(evaluation, sort_keys=True))
            expected_subject_scores = np.mean(
                [
                    [record["score"] for record in seed["subject_scores"]]
                    for seed in evaluation["seeds"]
                ],
                axis=0,
            )
            np.testing.assert_allclose(
                [record["score"] for record in evaluation["subject_scores"]],
                expected_subject_scores,
            )
            self.assertAlmostEqual(
                evaluation["cohort_score"],
                float(np.mean(expected_subject_scores)),
            )
            for seed in evaluation["seeds"]:
                model = load_fitted_model(output / seed["model_artifact"])
                report = load_state_model_scores(output / seed["score_artifact"])
                self.assertEqual(model.fit_subjects, development)
                self.assertEqual(report.fit_subjects, development)
                self.assertEqual(report.subjects, ("sub-004", "sub-005"))
                self.assertFalse(set(model.fit_subjects).intersection(report.subjects))
                self.assertEqual(
                    fitted_model_fingerprint(model),
                    seed["model_fingerprint"],
                )
            loaded = load_selected_state_count_evaluation(output)
            self.assertEqual(loaded.development_subjects, development)
            self.assertEqual(loaded.test_subjects, ("sub-004", "sub-005"))
            self.assertEqual(loaded.model_seeds, (17, 29))
            self.assertAlmostEqual(loaded.cohort_score, evaluation["cohort_score"])
            self.assertEqual(len(loaded.artifact_fingerprint), 64)

            evaluation_path = output / "evaluation.json"
            original = evaluation_path.read_text(encoding="utf-8")
            escaped = json.loads(original)
            escaped["seeds"][0]["model_artifact"] = "../model"
            evaluation_path.write_text(
                json.dumps(escaped, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inside the evaluation"):
                load_selected_state_count_evaluation(output)

            evaluation_path.write_text(original, encoding="utf-8")
            evaluation["cohort_score"] += 1.0
            evaluation_path.write_text(
                json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot be rebuilt"):
                load_selected_state_count_evaluation(output)

    def test_evaluate_selected_state_count_rejects_leakage_and_is_atomic(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=6)
            selection = root / "selection"
            arguments = [
                "cross-validate-state-counts",
                str(store.root),
                str(selection),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
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
            for subject in ("sub-000", "sub-001", "sub-002", "sub-003"):
                arguments.extend(("--subject", subject))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)

            for test_subjects, message in (
                (("sub-003",), "overlap"),
                (("sub-004", "sub-004"), "unique"),
                (("sub-missing",), "absent"),
            ):
                output = root / f"invalid-{message}"
                command = [
                    "evaluate-selected-state-count",
                    str(store.root),
                    str(selection),
                    str(output),
                ]
                for subject in test_subjects:
                    command.extend(("--test-subject", subject))
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(command), 2)
                self.assertIn(message, stderr.getvalue())
                self.assertFalse(output.exists())
                self.assertEqual(tuple(root.glob(f".{output.name}.tmp-*")), ())

            existing = root / "existing-evaluation"
            existing.mkdir()
            marker = existing / "preserve.txt"
            marker.write_text("preserve", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    main(
                        [
                            "evaluate-selected-state-count",
                            str(store.root),
                            str(selection),
                            str(existing),
                            "--test-subject",
                            "sub-004",
                        ]
                    ),
                    2,
                )
            self.assertIn("already exists", stderr.getvalue())
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

            failed = root / "failed-evaluation"
            stderr = io.StringIO()
            with patch(
                "dfckit.cli._fit_selected_state_count_model",
                side_effect=RuntimeError("injected fit failure"),
            ), contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    main(
                        [
                            "evaluate-selected-state-count",
                            str(store.root),
                            str(selection),
                            str(failed),
                            "--test-subject",
                            "sub-004",
                        ]
                    ),
                    2,
                )
            self.assertIn("injected fit failure", stderr.getvalue())
            self.assertFalse(failed.exists())
            self.assertEqual(tuple(root.glob(".failed-evaluation.tmp-*")), ())

    def test_nested_cross_validation_scores_every_subject_once(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "nested-cross-validate-state-counts",
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
                        "--outer-n-folds",
                        "2",
                        "--outer-split-seed",
                        "303",
                        "--inner-n-folds",
                        "2",
                        "--inner-split-seed",
                        "101",
                        "--n-init",
                        "1",
                        "--max-iter",
                        "1",
                        "--batch-size",
                        "16",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            nested = load_nested_state_count_cross_validation(output)
            self.assertEqual(summary["n_outer_folds"], 2)
            self.assertEqual(summary["n_subjects"], 5)
            self.assertEqual(summary["n_inner_fits"], 8)
            self.assertEqual(summary["n_outer_models"], 2)
            self.assertEqual(nested.subjects, store.subjects)
            self.assertEqual(len(nested.folds), 2)
            self.assertEqual(len(nested.selected_state_counts), 2)
            self.assertEqual(
                sorted(
                    subject
                    for fold in nested.folds
                    for subject in fold.evaluation.test_subjects
                ),
                sorted(store.subjects),
            )
            self.assertEqual(
                len(
                    {
                        subject
                        for fold in nested.folds
                        for subject in fold.evaluation.test_subjects
                    }
                ),
                len(store.subjects),
            )
            self.assertAlmostEqual(nested.cohort_score, float(np.mean(nested.subject_scores)))
            for fold in nested.folds:
                self.assertFalse(
                    set(fold.selection.development_subjects).intersection(
                        fold.evaluation.test_subjects
                    )
                )
                self.assertEqual(
                    fold.evaluation.selected_n_states,
                    fold.selection.selected_n_states(),
                )

            moved = root / "moved-nested"
            output.rename(moved)
            nested = load_nested_state_count_cross_validation(moved)
            self.assertEqual(nested.subjects, store.subjects)

            manifest_path = moved / "nested_evaluation.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cohort_score"] += 1.0
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot be rebuilt"):
                load_nested_state_count_cross_validation(moved)

    def test_nested_cross_validation_rejects_invalid_inner_split_and_is_atomic(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "invalid-nested"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "nested-cross-validate-state-counts",
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
                        "--outer-n-folds",
                        "2",
                        "--outer-split-seed",
                        "303",
                        "--inner-n-folds",
                        "4",
                        "--inner-split-seed",
                        "101",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("development subjects", stderr.getvalue())
            self.assertFalse(output.exists())
            self.assertEqual(tuple(root.glob(".invalid-nested.tmp-*")), ())

            failed = root / "failed-midway"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(failed),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            stderr = io.StringIO()
            with patch(
                "dfckit.cli._evaluate_selected_state_count",
                side_effect=RuntimeError("injected outer evaluation failure"),
            ), contextlib.redirect_stderr(stderr):
                self.assertEqual(main(command), 2)
            self.assertIn("injected outer evaluation failure", stderr.getvalue())
            self.assertFalse(failed.exists())
            self.assertEqual(tuple(root.glob(".failed-midway.tmp-*")), ())

    def test_nested_loader_rejects_path_and_child_identity_tampering(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            command = [
                "nested-cross-validate-state-counts",
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
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(command), 0)
            manifest_path = output / "nested_evaluation.json"
            original = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(original)
            manifest["folds"][0]["selection_artifact"] = "../selection"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inside the nested"):
                load_nested_state_count_cross_validation(output)

            manifest_path.write_text(original, encoding="utf-8")
            manifest = json.loads(original)
            manifest["folds"][0]["evaluation_fingerprint"] = "0" * 64
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "child evidence"):
                load_nested_state_count_cross_validation(output)

            manifest_path.write_text(
                original.replace("{", '{"format": "duplicate",', 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                load_nested_state_count_cross_validation(output)

    def test_nested_checkpoint_resumes_only_missing_outer_evaluation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            original_evaluate = cli_module._evaluate_selected_state_count
            evaluation_calls = 0

            def fail_second_evaluation(namespace):
                nonlocal evaluation_calls
                evaluation_calls += 1
                if evaluation_calls == 2:
                    raise RuntimeError("injected second-fold failure")
                return original_evaluate(namespace)

            stderr = io.StringIO()
            with patch(
                "dfckit.cli._evaluate_selected_state_count",
                side_effect=fail_second_evaluation,
            ), contextlib.redirect_stderr(stderr):
                self.assertEqual(main(command), 2)
            self.assertIn("injected second-fold failure", stderr.getvalue())
            self.assertFalse(output.exists())
            self.assertTrue((checkpoint / "checkpoint.json").is_file())
            self.assertEqual(
                len(tuple(checkpoint.glob("outer-folds/fold-*/selection"))),
                2,
            )
            self.assertEqual(
                len(tuple(checkpoint.glob("outer-folds/fold-*/evaluation"))),
                1,
            )

            stdout = io.StringIO()
            with patch(
                "dfckit.cli._cross_validate_state_counts",
                wraps=cli_module._cross_validate_state_counts,
            ) as cross_validate, patch(
                "dfckit.cli._evaluate_selected_state_count",
                wraps=cli_module._evaluate_selected_state_count,
            ) as evaluate, contextlib.redirect_stdout(stdout):
                self.assertEqual(main(command), 0)
            self.assertEqual(cross_validate.call_count, 0)
            self.assertEqual(evaluate.call_count, 1)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["reused_selections"], 2)
            self.assertEqual(summary["reused_evaluations"], 1)
            self.assertFalse(checkpoint.exists())
            self.assertFalse((output / "checkpoint.json").exists())
            self.assertEqual(
                load_nested_state_count_cross_validation(output).subjects,
                store.subjects,
            )

    def test_nested_checkpoint_resumes_partial_inner_grid_cells(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            original_scores = cli_module._cross_validation_scores
            score_calls = 0

            def fail_second_score(store, model, subjects):
                nonlocal score_calls
                score_calls += 1
                if score_calls == 2:
                    raise RuntimeError("injected partial inner-grid failure")
                return original_scores(store, model, subjects)

            stderr = io.StringIO()
            with patch(
                "dfckit.cli._cross_validation_scores",
                side_effect=fail_second_score,
            ), contextlib.redirect_stderr(stderr):
                self.assertEqual(main(command), 2)
            self.assertIn("partial inner-grid failure", stderr.getvalue())
            selection_checkpoint = (
                checkpoint / "outer-folds" / "fold-001" / "selection"
            )
            self.assertTrue((selection_checkpoint / "checkpoint.json").is_file())
            self.assertEqual(
                len(tuple((selection_checkpoint / "models").glob("*.model"))),
                2,
            )
            self.assertEqual(
                len(tuple((selection_checkpoint / "scores").glob("*.json"))),
                1,
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "inspect-nested-state-counts",
                            str(store.root),
                            str(checkpoint),
                        ]
                    ),
                    0,
                )
            progress = json.loads(stdout.getvalue())
            self.assertEqual(progress["completed_inner_fits"], 2)
            self.assertEqual(progress["completed_inner_scores"], 1)
            self.assertEqual(progress["total_inner_fits"], 8)
            self.assertEqual(progress["total_inner_scores"], 8)
            self.assertEqual(progress["completed_fit_steps"], 2)
            self.assertEqual(progress["total_fit_steps"], 10)
            self.assertEqual(progress["fit_completion_fraction"], 0.2)
            self.assertEqual(progress["folds"][0]["status"], "selection-in-progress")
            self.assertEqual(progress["folds"][0]["completed_inner_models"], 2)
            self.assertEqual(progress["folds"][0]["completed_inner_scores"], 1)

            stdout = io.StringIO()
            with patch(
                "dfckit.cli._fit_cross_validation_model",
                wraps=cli_module._fit_cross_validation_model,
            ) as fit_model, contextlib.redirect_stdout(stdout):
                self.assertEqual(main(command), 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(fit_model.call_count, 6)
            self.assertEqual(summary["reused_inner_models"], 2)
            self.assertEqual(summary["reused_inner_scores"], 1)
            self.assertEqual(summary["reused_selections"], 0)
            self.assertFalse(checkpoint.exists())
            self.assertTrue(output.is_dir())
            self.assertEqual(tuple(output.rglob("checkpoint.json")), ())
            self.assertEqual(
                load_nested_state_count_cross_validation(output).subjects,
                store.subjects,
            )

    @unittest.skipUnless(HAS_HMM_EXTRA, "requires dfc-kit[hmm]")
    def test_nested_hmm_checkpoint_resumes_partial_inner_grid_cells(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=6)
            output = root / "nested-hmm"
            checkpoint = root / "nested-hmm-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "hmm",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-pca-components",
                "2",
                "--n-init",
                "1",
                "--n-iter",
                "3",
                "--pca-batch-size",
                "16",
            ]
            original_scores = cli_module._cross_validation_scores
            score_calls = 0

            def fail_second_score(store, model, subjects):
                nonlocal score_calls
                score_calls += 1
                if score_calls == 2:
                    raise RuntimeError("injected partial HMM inner-grid failure")
                return original_scores(store, model, subjects)

            with patch(
                "dfckit.cli._cross_validation_scores",
                side_effect=fail_second_score,
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 2)

            selection_checkpoint = (
                checkpoint / "outer-folds" / "fold-001" / "selection"
            )
            self.assertEqual(
                len(tuple((selection_checkpoint / "models").glob("*.model"))),
                2,
            )
            self.assertEqual(
                len(tuple((selection_checkpoint / "scores").glob("*.json"))),
                1,
            )

            stdout = io.StringIO()
            with patch(
                "dfckit.cli._fit_cross_validation_model",
                wraps=cli_module._fit_cross_validation_model,
            ) as fit_model, contextlib.redirect_stdout(stdout):
                self.assertEqual(main(command), 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(fit_model.call_count, 6)
            self.assertEqual(summary["reused_inner_models"], 2)
            self.assertEqual(summary["reused_inner_scores"], 1)
            nested = load_nested_state_count_cross_validation(output)
            self.assertEqual(nested.model_kind, "gaussian-hmm-state")
            self.assertEqual(nested.subjects, store.subjects)

    def test_nested_checkpoint_rejects_tampered_partial_inner_score_before_refit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            original_scores = cli_module._cross_validation_scores
            score_calls = 0

            def fail_second_score(store, model, subjects):
                nonlocal score_calls
                score_calls += 1
                if score_calls == 2:
                    raise RuntimeError("leave partial score checkpoint")
                return original_scores(store, model, subjects)

            with patch(
                "dfckit.cli._cross_validation_scores",
                side_effect=fail_second_score,
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 2)

            score_path = next(
                (
                    checkpoint
                    / "outer-folds"
                    / "fold-001"
                    / "selection"
                    / "scores"
                ).glob("*.json")
            )
            score = json.loads(score_path.read_text(encoding="utf-8"))
            score["evaluation_data_fingerprint"] = "0" * 64
            score_path.write_text(
                json.dumps(score, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            stderr = io.StringIO()
            with patch(
                "dfckit.cli._fit_cross_validation_model",
                side_effect=AssertionError("tampering must fail before refitting"),
            ) as fit_model, contextlib.redirect_stderr(stderr):
                self.assertEqual(main(command), 2)
            self.assertEqual(fit_model.call_count, 0)
            self.assertIn("checkpoint score does not match", stderr.getvalue())
            self.assertTrue(checkpoint.is_dir())
            self.assertFalse(output.exists())

    def test_nested_checkpoint_rejects_a_second_writer_before_output_changes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
            ]
            with acquire_nested_checkpoint_lock(checkpoint):
                lock_path = nested_checkpoint_lock_path(checkpoint)
                owner_record = lock_path.read_bytes()
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(command), 2)
                self.assertIn("already active", stderr.getvalue())
                self.assertEqual(lock_path.read_bytes(), owner_record)
                self.assertFalse(checkpoint.exists())
                self.assertFalse(output.exists())

    def test_inspect_nested_progress_reports_active_checkpoint_owner(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
            ]
            with patch(
                "dfckit.cli._cross_validate_state_counts",
                side_effect=RuntimeError("leave inspectable checkpoint"),
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 2)

            inspect_command = [
                "inspect-nested-state-counts",
                str(store.root),
                str(checkpoint),
            ]
            with acquire_nested_checkpoint_lock(checkpoint):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(main(inspect_command), 0)
                progress = json.loads(stdout.getvalue())
                self.assertEqual(progress["execution_status"], "active")
                self.assertEqual(progress["lock_owner_pid"], os.getpid())
                self.assertEqual(
                    progress["lock_owner_process_start_token"],
                    json.loads(
                        nested_checkpoint_lock_path(checkpoint).read_text(
                            encoding="utf-8"
                        )
                    )["process_start_token"],
                )
                self.assertEqual(
                    Path(progress["lock_path"]),
                    nested_checkpoint_lock_path(checkpoint),
                )
                self.assertIsNotNone(progress["lock_acquired_at_unix"])
                self.assertIsNone(progress["lock_released_at_unix"])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(inspect_command), 0)
            self.assertEqual(json.loads(stdout.getvalue())["execution_status"], "idle")

            lock_path = nested_checkpoint_lock_path(checkpoint)
            stale_record = json.loads(lock_path.read_text(encoding="utf-8"))
            stale_record["state"] = "active"
            stale_record["released_at_unix"] = None
            lock_path.write_text(
                json.dumps(stale_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            before = lock_path.read_bytes()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(inspect_command), 0)
            stale_progress = json.loads(stdout.getvalue())
            self.assertEqual(stale_progress["execution_status"], "stale")
            self.assertEqual(stale_progress["lock_owner_pid"], os.getpid())
            self.assertEqual(lock_path.read_bytes(), before)

    def test_nested_checkpoint_rejects_contract_or_data_drift(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            with patch(
                "dfckit.cli._evaluate_selected_state_count",
                side_effect=RuntimeError("stop after first selection"),
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 2)

            changed_command = list(command)
            changed_command[changed_command.index("101")] = "102"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(main(changed_command), 2)
            self.assertIn("does not match", stderr.getvalue())
            self.assertTrue(checkpoint.exists())
            self.assertFalse(output.exists())

            outside_checkpoint = root / "other" / "checkpoint"
            stderr = io.StringIO()
            outside_command = list(command)
            outside_command[outside_command.index(str(checkpoint))] = str(
                outside_checkpoint
            )
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(main(outside_command), 2)
            self.assertIn("sibling", stderr.getvalue())
            self.assertFalse(outside_checkpoint.exists())

            values_path = store.root / "chunks" / "00000000.values.npy"
            values = np.load(values_path, allow_pickle=False)
            changed = values.copy()
            changed[0, 0] += 0.5
            np.save(values_path, changed, allow_pickle=False)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(main(command), 2)
            self.assertIn("current data", stderr.getvalue())
            self.assertTrue(checkpoint.exists())
            self.assertFalse(output.exists())

    @unittest.skipIf(os.name == "nt", "directory symlinks require optional privileges")
    def test_nested_checkpoint_rejects_symlinked_internal_parent_before_writing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            with patch(
                "dfckit.cli._cross_validate_state_counts",
                side_effect=RuntimeError("stop before first selection"),
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 2)

            outside = root / "outside"
            outside.mkdir()
            (checkpoint / "outer-folds").symlink_to(outside, target_is_directory=True)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(main(command), 2)
            self.assertIn("must not contain a symlink", stderr.getvalue())
            self.assertEqual(tuple(outside.iterdir()), ())
            self.assertFalse(output.exists())

    def test_nested_checkpoint_recovers_after_atomic_manifest_failure(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            with patch(
                "dfckit.cli._write_new_json",
                side_effect=RuntimeError("injected final-manifest failure"),
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 2)
            self.assertTrue((checkpoint / "checkpoint.json").is_file())
            self.assertFalse((checkpoint / "nested_evaluation.json").exists())
            self.assertEqual(
                tuple(checkpoint.glob(".nested_evaluation.json.tmp-*")),
                (),
            )

            stdout = io.StringIO()
            with patch(
                "dfckit.cli._cross_validate_state_counts",
                side_effect=AssertionError("selection must be reused"),
            ), patch(
                "dfckit.cli._evaluate_selected_state_count",
                side_effect=AssertionError("evaluation must be reused"),
            ), contextlib.redirect_stdout(stdout):
                self.assertEqual(main(command), 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["reused_selections"], 2)
            self.assertEqual(summary["reused_evaluations"], 2)
            self.assertFalse(checkpoint.exists())
            self.assertTrue(output.is_dir())

    def test_nested_checkpoint_audits_completed_selection_with_internal_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            original_unlink = Path.unlink
            selection_manifest_removals = 0

            def fail_first_selection_manifest_removal(path, *args, **kwargs):
                nonlocal selection_manifest_removals
                if path.name == "checkpoint.json" and path.parent.name == "selection":
                    selection_manifest_removals += 1
                    if selection_manifest_removals == 1:
                        raise RuntimeError("crash after complete selection manifest")
                return original_unlink(path, *args, **kwargs)

            with patch.object(
                Path,
                "unlink",
                new=fail_first_selection_manifest_removal,
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 2)

            selection = checkpoint / "outer-folds" / "fold-001" / "selection"
            internal_manifest = selection / "checkpoint.json"
            self.assertTrue((selection / "workflow.json").is_file())
            self.assertTrue(internal_manifest.is_file())
            before = internal_manifest.read_bytes()

            inspect_command = [
                "inspect-nested-state-counts",
                str(store.root),
                str(checkpoint),
            ]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(inspect_command), 0)
            progress = json.loads(stdout.getvalue())
            self.assertEqual(progress["folds"][0]["status"], "pending-evaluation")
            self.assertEqual(progress["folds"][0]["completed_inner_models"], 4)
            self.assertEqual(progress["folds"][0]["completed_inner_scores"], 4)
            self.assertEqual(internal_manifest.read_bytes(), before)

            checkpoint_payload = json.loads(internal_manifest.read_text(encoding="utf-8"))
            checkpoint_payload["development_data_fingerprint"] = "0" * 64
            internal_manifest.write_text(
                json.dumps(checkpoint_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(main(inspect_command), 2)
            self.assertIn("does not match", stderr.getvalue())

            internal_manifest.write_bytes(before)
            stdout = io.StringIO()
            with patch(
                "dfckit.cli._cross_validate_state_counts",
                wraps=cli_module._cross_validate_state_counts,
            ) as cross_validate, contextlib.redirect_stdout(stdout):
                self.assertEqual(main(command), 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(cross_validate.call_count, 1)
            self.assertEqual(summary["reused_selections"], 1)
            self.assertFalse(checkpoint.exists())
            self.assertTrue(output.is_dir())

    def test_inspect_nested_progress_is_read_only_and_data_bound(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            original_evaluate = cli_module._evaluate_selected_state_count
            evaluation_calls = 0

            def fail_second_evaluation(namespace):
                nonlocal evaluation_calls
                evaluation_calls += 1
                if evaluation_calls == 2:
                    raise RuntimeError("injected progress checkpoint")
                return original_evaluate(namespace)

            with patch(
                "dfckit.cli._evaluate_selected_state_count",
                side_effect=fail_second_evaluation,
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 2)

            stale = checkpoint / "outer-folds" / "fold-002" / ".evaluation.tmp-stale"
            stale.mkdir()
            marker = stale / "marker.txt"
            marker.write_text("preserve during inspection", encoding="utf-8")
            inspect_command = [
                "inspect-nested-state-counts",
                str(store.root),
                str(checkpoint),
            ]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(inspect_command), 0)
            progress = json.loads(stdout.getvalue())
            self.assertEqual(progress["status"], "in-progress")
            self.assertEqual(progress["completed_selections"], 2)
            self.assertEqual(progress["completed_evaluations"], 1)
            self.assertEqual(progress["completed_inner_fits"], 8)
            self.assertEqual(progress["total_inner_fits"], 8)
            self.assertEqual(progress["completed_outer_models"], 1)
            self.assertEqual(progress["total_outer_models"], 2)
            self.assertEqual(progress["completed_fit_steps"], 9)
            self.assertEqual(progress["total_fit_steps"], 10)
            self.assertEqual(progress["fit_completion_fraction"], 0.9)
            self.assertEqual(progress["outer_split_seed"], 303)
            self.assertEqual(progress["inner_split_seed"], 101)
            self.assertEqual(
                progress["temporary_paths"],
                ["outer-folds/fold-002/.evaluation.tmp-stale"],
            )
            self.assertEqual(
                [fold["status"] for fold in progress["folds"]],
                ["complete", "pending-evaluation"],
            )
            self.assertTrue(marker.is_file())

            checkpoint_manifest = checkpoint / "checkpoint.json"
            original_checkpoint = checkpoint_manifest.read_text(encoding="utf-8")
            changed_checkpoint = json.loads(original_checkpoint)
            changed_checkpoint["model_kind"] = "gaussian-hmm-state"
            checkpoint_manifest.write_text(
                json.dumps(changed_checkpoint, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(main(inspect_command), 2)
            self.assertIn("model kind disagree", stderr.getvalue())
            checkpoint_manifest.write_text(original_checkpoint, encoding="utf-8")

            selection_manifest = (
                checkpoint
                / "outer-folds"
                / "fold-001"
                / "selection"
                / "workflow.json"
            )
            original_selection = selection_manifest.read_text(encoding="utf-8")
            changed_selection = json.loads(original_selection)
            changed_selection["development_data_fingerprint"] = "0" * 64
            selection_manifest.write_text(
                json.dumps(changed_selection, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(main(inspect_command), 2)
            self.assertIn("does not match", stderr.getvalue())
            selection_manifest.write_text(original_selection, encoding="utf-8")

            values_path = store.root / "chunks" / "00000000.values.npy"
            original_values = np.load(values_path, allow_pickle=False)
            changed = original_values.copy()
            changed[0, 0] += 0.5
            np.save(values_path, changed, allow_pickle=False)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(main(inspect_command), 2)
            self.assertIn("current FeatureStore", stderr.getvalue())
            np.save(values_path, original_values, allow_pickle=False)

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(command), 0)
            self.assertFalse(checkpoint.exists())
            completed_stdout = io.StringIO()
            with contextlib.redirect_stdout(completed_stdout):
                self.assertEqual(
                    main(
                        [
                            "inspect-nested-state-counts",
                            str(store.root),
                            str(output),
                        ]
                    ),
                    0,
                )
            completed = json.loads(completed_stdout.getvalue())
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["completed_fit_steps"], 10)
            self.assertEqual(completed["fit_completion_fraction"], 1.0)
            self.assertEqual(len(completed["workflow_fingerprint"]), 64)
            self.assertIsInstance(completed["cohort_score"], float)

    def test_inspect_nested_progress_reports_ready_for_promotion(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            original_unlink = Path.unlink

            def fail_checkpoint_unlink(path, *args, **kwargs):
                if path == checkpoint / "checkpoint.json":
                    raise RuntimeError("injected pre-promotion interruption")
                return original_unlink(path, *args, **kwargs)

            with patch.object(
                Path,
                "unlink",
                new=fail_checkpoint_unlink,
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 2)
            self.assertTrue((checkpoint / "checkpoint.json").is_file())
            self.assertTrue((checkpoint / "nested_evaluation.json").is_file())
            self.assertFalse(output.exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "inspect-nested-state-counts",
                            str(store.root),
                            str(checkpoint),
                        ]
                    ),
                    0,
                )
            progress = json.loads(stdout.getvalue())
            self.assertEqual(progress["status"], "ready-for-promotion")
            self.assertEqual(progress["completed_fit_steps"], 10)
            self.assertEqual(progress["fit_completion_fraction"], 1.0)
            self.assertEqual(len(progress["workflow_fingerprint"]), 64)

            with patch(
                "dfckit.cli._cross_validate_state_counts",
                side_effect=AssertionError("selection must not rerun"),
            ), patch(
                "dfckit.cli._evaluate_selected_state_count",
                side_effect=AssertionError("evaluation must not rerun"),
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(command), 0)
            self.assertFalse(checkpoint.exists())
            self.assertTrue(output.is_dir())

    def test_inspect_nested_progress_rejects_malformed_fold_structure(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            with patch(
                "dfckit.cli._cross_validate_state_counts",
                side_effect=RuntimeError("leave empty checkpoint"),
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 2)
            (checkpoint / "outer-folds").write_text("not a directory", encoding="utf-8")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(
                    main(
                        [
                            "inspect-nested-state-counts",
                            str(store.root),
                            str(checkpoint),
                        ]
                    ),
                    2,
                )
            self.assertIn("outer-folds path must be a directory", stderr.getvalue())

    def test_nested_completed_checkpoint_promotes_without_refitting(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=5)
            output = root / "nested"
            checkpoint = root / "nested-checkpoint"
            command = [
                "nested-cross-validate-state-counts",
                str(store.root),
                str(output),
                "--checkpoint",
                str(checkpoint),
                "--method",
                "kmeans",
                "--n-states",
                "2",
                "--n-states",
                "3",
                "--seed",
                "17",
                "--outer-n-folds",
                "2",
                "--outer-split-seed",
                "303",
                "--inner-n-folds",
                "2",
                "--inner-split-seed",
                "101",
                "--n-init",
                "1",
                "--max-iter",
                "1",
                "--batch-size",
                "16",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(command), 0)
            output.rename(checkpoint)
            self.assertFalse((checkpoint / "checkpoint.json").exists())
            self.assertTrue((checkpoint / "nested_evaluation.json").is_file())

            if os.name != "nt":
                outside = root / "outside.txt"
                outside.write_text("outside", encoding="utf-8")
                extra = checkpoint / "untracked-link"
                extra.symlink_to(outside)
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(command), 2)
                self.assertIn("must not contain symlinks", stderr.getvalue())
                self.assertTrue(checkpoint.is_dir())
                self.assertFalse(output.exists())
                inspect_stderr = io.StringIO()
                with contextlib.redirect_stderr(inspect_stderr):
                    self.assertEqual(
                        main(
                            [
                                "inspect-nested-state-counts",
                                str(store.root),
                                str(checkpoint),
                            ]
                        ),
                        2,
                    )
                self.assertIn("must not contain symlinks", inspect_stderr.getvalue())
                extra.unlink()

            stdout = io.StringIO()
            with patch(
                "dfckit.cli._cross_validate_state_counts",
                side_effect=AssertionError("selection must be reused"),
            ), patch(
                "dfckit.cli._evaluate_selected_state_count",
                side_effect=AssertionError("evaluation must be reused"),
            ), contextlib.redirect_stdout(stdout):
                self.assertEqual(main(command), 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["reused_selections"], 2)
            self.assertEqual(summary["reused_evaluations"], 2)
            self.assertFalse(checkpoint.exists())
            self.assertTrue(output.is_dir())

    @unittest.skipUnless(HAS_HMM_EXTRA, "requires dfc-kit[hmm]")
    def test_nested_cross_validation_runs_hmm(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store", n_subjects=6)
            output = root / "nested-hmm"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "nested-cross-validate-state-counts",
                        str(store.root),
                        str(output),
                        "--method",
                        "hmm",
                        "--n-states",
                        "2",
                        "--n-states",
                        "3",
                        "--seed",
                        "17",
                        "--outer-n-folds",
                        "2",
                        "--outer-split-seed",
                        "303",
                        "--inner-n-folds",
                        "2",
                        "--inner-split-seed",
                        "101",
                        "--n-pca-components",
                        "2",
                        "--n-init",
                        "1",
                        "--n-iter",
                        "3",
                        "--pca-batch-size",
                        "16",
                    ]
                )
            self.assertEqual(status, 0)
            nested = load_nested_state_count_cross_validation(output)
            self.assertEqual(nested.model_kind, "gaussian-hmm-state")
            self.assertEqual(nested.subjects, store.subjects)
            self.assertEqual(len(nested.folds), 2)
            self.assertTrue(np.isfinite(nested.cohort_score))
            progress_stdout = io.StringIO()
            with contextlib.redirect_stdout(progress_stdout):
                self.assertEqual(
                    main(
                        [
                            "inspect-nested-state-counts",
                            str(store.root),
                            str(output),
                        ]
                    ),
                    0,
                )
            progress = json.loads(progress_stdout.getvalue())
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(progress["model_kind"], "gaussian-hmm-state")
            self.assertEqual(progress["completed_inner_fits"], 8)
            self.assertEqual(progress["completed_outer_models"], 2)
            self.assertEqual(progress["fit_completion_fraction"], 1.0)

    @unittest.skipUnless(HAS_HMM_EXTRA, "requires dfc-kit[hmm]")
    def test_cross_validate_state_counts_runs_complete_hmm_grid(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            output = root / "hmm-selection"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "cross-validate-state-counts",
                        str(store.root),
                        str(output),
                        "--method",
                        "hmm",
                        "--n-states",
                        "2",
                        "--n-states",
                        "3",
                        "--seed",
                        "17",
                        "--n-folds",
                        "2",
                        "--split-seed",
                        "101",
                        "--n-pca-components",
                        "2",
                        "--n-init",
                        "1",
                        "--n-iter",
                        "20",
                        "--pca-batch-size",
                        "16",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            workflow = json.loads(
                (output / "workflow.json").read_text(encoding="utf-8")
            )
            comparison = json.loads(
                (output / "comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["model_kind"], "gaussian-hmm-state")
            self.assertEqual(summary["n_fits"], 4)
            self.assertEqual(workflow["fit_configuration"]["n_pca_components"], 2)
            self.assertEqual(comparison["selection_direction"], "higher is better")
            self.assertEqual(len(tuple((output / "models").iterdir())), 4)
            self.assertEqual(len(tuple((output / "scores").iterdir())), 4)

    def test_align_states_kmeans_relabels_model_and_predictions(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            reference = root / "reference.model"
            candidate = root / "candidate.model"
            for output, seed in ((reference, 17), (candidate, 29)):
                status = main(
                    [
                        "fit-states",
                        str(store.root),
                        str(output),
                        "--method",
                        "kmeans",
                        "--n-states",
                        "2",
                        "--seed",
                        str(seed),
                        "--n-init",
                        "2",
                        "--max-iter",
                        "2",
                        "--batch-size",
                        "16",
                    ]
                )
                self.assertEqual(status, 0)
            candidate_predictions = root / "candidate.labels"
            status = main(
                [
                    "predict-states",
                    str(store.root),
                    str(candidate),
                    str(candidate_predictions),
                    "--allow-fit-subjects",
                ]
            )
            self.assertEqual(status, 0)

            aligned_model_path = root / "aligned.model"
            alignment_path = root / "alignment"
            aligned_predictions_path = root / "aligned.labels"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "align-states",
                        str(reference),
                        str(candidate),
                        str(aligned_model_path),
                        str(alignment_path),
                        "--predictions",
                        str(candidate_predictions),
                        "--predictions-output",
                        str(aligned_predictions_path),
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["reference_seed"], 17)
            self.assertEqual(summary["candidate_seed"], 29)
            alignment = load_state_alignment(alignment_path)
            aligned_model = load_fitted_model(aligned_model_path)
            reference_model = load_fitted_model(reference)
            candidate_model = load_fitted_model(candidate)
            np.testing.assert_allclose(
                aligned_model.centers,
                candidate_model.centers[np.argsort(alignment.candidate_to_reference)],
            )
            self.assertEqual(aligned_model.seed, 29)
            self.assertEqual(reference_model.seed, 17)
            candidate_labels = load_state_predictions(candidate_predictions)
            aligned_labels = load_state_predictions(aligned_predictions_path)
            expected = alignment.candidate_to_reference[
                candidate_labels.assignments.sequences[0].labels
            ]
            np.testing.assert_array_equal(
                aligned_labels.assignments.sequences[0].labels,
                expected,
            )
            self.assertEqual(aligned_labels.model_seed, 29)
            self.assertEqual(
                aligned_labels.model_fingerprint,
                fitted_model_fingerprint(aligned_model),
            )

    @unittest.skipUnless(HAS_HMM_EXTRA, "requires dfc-kit[hmm]")
    def test_align_states_hmm_reorders_posterior_columns(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            reference = root / "reference-hmm.model"
            candidate = root / "candidate-hmm.model"
            for output, seed in ((reference, 31), (candidate, 47)):
                status = main(
                    [
                        "fit-states",
                        str(store.root),
                        str(output),
                        "--method",
                        "hmm",
                        "--n-states",
                        "2",
                        "--seed",
                        str(seed),
                        "--n-pca-components",
                        "2",
                        "--n-init",
                        "1",
                        "--n-iter",
                        "20",
                        "--pca-batch-size",
                        "16",
                    ]
                )
                self.assertEqual(status, 0)
            candidate_predictions = root / "candidate-hmm.labels"
            self.assertEqual(
                main(
                    [
                        "predict-states",
                        str(store.root),
                        str(candidate),
                        str(candidate_predictions),
                        "--allow-fit-subjects",
                    ]
                ),
                0,
            )
            aligned_predictions = root / "aligned-hmm.labels"
            self.assertEqual(
                main(
                    [
                        "align-states",
                        str(reference),
                        str(candidate),
                        str(root / "aligned-hmm.model"),
                        str(root / "hmm-alignment"),
                        "--predictions",
                        str(candidate_predictions),
                        "--predictions-output",
                        str(aligned_predictions),
                    ]
                ),
                0,
            )
            alignment = load_state_alignment(root / "hmm-alignment")
            before = load_state_predictions(candidate_predictions)
            after = load_state_predictions(aligned_predictions)
            assert before.posterior_probabilities is not None
            assert after.posterior_probabilities is not None
            expected = np.empty_like(before.posterior_probabilities[0])
            expected[:, alignment.candidate_to_reference] = before.posterior_probabilities[0]
            np.testing.assert_allclose(after.posterior_probabilities[0], expected)
            np.testing.assert_array_equal(
                after.assignments.sequences[0].labels,
                alignment.candidate_to_reference[
                    before.assignments.sequences[0].labels
                ],
            )

    def test_align_states_requires_prediction_path_pair(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            reference = root / "reference.model"
            candidate = root / "candidate.model"
            for output, seed in ((reference, 17), (candidate, 29)):
                self.assertEqual(
                    main(
                        [
                            "fit-states",
                            str(store.root),
                            str(output),
                            "--method",
                            "kmeans",
                            "--n-states",
                            "2",
                            "--seed",
                            str(seed),
                            "--n-init",
                            "2",
                            "--max-iter",
                            "2",
                        ]
                    ),
                    0,
                )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "align-states",
                        str(reference),
                        str(candidate),
                        str(root / "aligned.model"),
                        str(root / "alignment"),
                        "--predictions",
                        str(root / "labels"),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("provided together", stderr.getvalue())

    def test_align_states_rejects_same_seed_but_different_candidate_model(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            reference = root / "reference.model"
            candidate = root / "candidate.model"
            same_seed_different_cohort = root / "same-seed.model"
            for output, seed, subjects in (
                (reference, 17, ("sub-000", "sub-001")),
                (candidate, 29, ("sub-000", "sub-001", "sub-002")),
                (same_seed_different_cohort, 29, ("sub-000", "sub-001")),
            ):
                arguments = [
                    "fit-states",
                    str(store.root),
                    str(output),
                    "--method",
                    "kmeans",
                    "--n-states",
                    "2",
                    "--seed",
                    str(seed),
                    "--n-init",
                    "2",
                    "--max-iter",
                    "2",
                ]
                for subject in subjects:
                    arguments.extend(("--subject", subject))
                self.assertEqual(main(arguments), 0)
            predictions = root / "candidate.labels"
            self.assertEqual(
                main(
                    [
                        "predict-states",
                        str(store.root),
                        str(candidate),
                        str(predictions),
                        "--subject",
                        "sub-003",
                    ]
                ),
                0,
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "align-states",
                        str(reference),
                        str(same_seed_different_cohort),
                        str(root / "aligned.model"),
                        str(root / "alignment"),
                        "--predictions",
                        str(predictions),
                        "--predictions-output",
                        str(root / "aligned.labels"),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("fingerprint", stderr.getvalue())

    def test_summarize_stability_aligns_repeated_kmeans_on_one_heldout_cohort(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            models = tuple(root / f"seed-{seed}.model" for seed in (17, 29, 41))
            for model, seed in zip(models, (17, 29, 41), strict=True):
                arguments = [
                    "fit-states",
                    str(store.root),
                    str(model),
                    "--method",
                    "kmeans",
                    "--n-states",
                    "2",
                    "--seed",
                    str(seed),
                    "--n-init",
                    "2",
                    "--max-iter",
                    "2",
                    "--batch-size",
                    "16",
                    "--subject",
                    "sub-000",
                    "--subject",
                    "sub-001",
                ]
                self.assertEqual(main(arguments), 0)
            output = root / "stability.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "summarize-stability",
                        str(store.root),
                        str(models[0]),
                        str(output),
                        "--candidate-model",
                        str(models[1]),
                        "--candidate-model",
                        str(models[2]),
                        "--subject",
                        "sub-002",
                        "--subject",
                        "sub-003",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            report_text = output.read_text(encoding="utf-8")
            report = json.loads(report_text)
            self.assertEqual(summary["n_fits"], 3)
            self.assertEqual(summary["subjects"], ["sub-002", "sub-003"])
            self.assertEqual(report["state_numbering"], "reference-model state indices")
            self.assertEqual(report["n_fits"], 3)
            self.assertEqual(len(report["fits"]), 3)
            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual(len(report["runs"][0]["occupancy"]["by_fit"]), 3)
            self.assertEqual(
                report["reference_model_fingerprint"],
                fitted_model_fingerprint(load_fitted_model(models[0])),
            )
            self.assertNotIn("NaN", report_text)
            fingerprints = {
                load_fitted_model(model).training_data_fingerprint for model in models
            }
            self.assertEqual(len(fingerprints), 1)
            self.assertNotIn(None, fingerprints)

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                duplicate_status = main(
                    [
                        "summarize-stability",
                        str(store.root),
                        str(models[0]),
                        str(root / "duplicate.json"),
                        "--candidate-model",
                        str(models[0]),
                        "--subject",
                        "sub-002",
                    ]
                )
            self.assertEqual(duplicate_status, 2)
            self.assertIn("distinct fitted artifacts", stderr.getvalue())

    @unittest.skipUnless(HAS_HMM_EXTRA, "requires dfc-kit[hmm]")
    def test_summarize_stability_decodes_and_aligns_hmm_models(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            models = (root / "hmm-17.model", root / "hmm-29.model")
            for model, seed in zip(models, (17, 29), strict=True):
                self.assertEqual(
                    main(
                        [
                            "fit-states",
                            str(store.root),
                            str(model),
                            "--method",
                            "hmm",
                            "--n-states",
                            "2",
                            "--seed",
                            str(seed),
                            "--n-pca-components",
                            "2",
                            "--n-init",
                            "1",
                            "--n-iter",
                            "20",
                            "--pca-batch-size",
                            "16",
                            "--subject",
                            "sub-000",
                            "--subject",
                            "sub-001",
                        ]
                    ),
                    0,
                )
            output = root / "hmm-stability.json"
            self.assertEqual(
                main(
                    [
                        "summarize-stability",
                        str(store.root),
                        str(models[0]),
                        str(output),
                        "--candidate-model",
                        str(models[1]),
                        "--subject",
                        "sub-002",
                    ]
                ),
                0,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["model_kind"], "gaussian-hmm-state")
            self.assertEqual(report["n_fits"], 2)
            self.assertEqual(report["subjects"], ["sub-002"])

    @unittest.skipUnless(HAS_HMM_EXTRA, "requires dfc-kit[hmm]")
    def test_summarize_stability_rejects_mixed_model_families(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            kmeans = root / "kmeans.model"
            hmm = root / "hmm.model"
            self.assertEqual(
                main(
                    [
                        "fit-states",
                        str(store.root),
                        str(kmeans),
                        "--method",
                        "kmeans",
                        "--n-states",
                        "2",
                        "--seed",
                        "17",
                        "--subject",
                        "sub-000",
                        "--subject",
                        "sub-001",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "fit-states",
                        str(store.root),
                        str(hmm),
                        "--method",
                        "hmm",
                        "--n-states",
                        "2",
                        "--seed",
                        "29",
                        "--n-pca-components",
                        "2",
                        "--n-iter",
                        "20",
                        "--pca-batch-size",
                        "16",
                        "--subject",
                        "sub-000",
                        "--subject",
                        "sub-001",
                    ]
                ),
                0,
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "summarize-stability",
                        str(store.root),
                        str(kmeans),
                        str(root / "stability.json"),
                        "--candidate-model",
                        str(hmm),
                        "--subject",
                        "sub-002",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("same state-model family", stderr.getvalue())

    def test_summarize_stability_rejects_legacy_missing_fit_provenance(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            models = (root / "reference.model", root / "candidate.model")
            for model, seed in zip(models, (17, 29), strict=True):
                self.assertEqual(
                    main(
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
                            "--n-init",
                            "2",
                            "--max-iter",
                            "2",
                        ]
                    ),
                    0,
                )
                manifest_path = model / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["format_version"] = 1
                del manifest["metadata"]["init_sample_size"]
                del manifest["metadata"]["training_data_fingerprint"]
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "summarize-stability",
                        str(store.root),
                        str(models[0]),
                        str(root / "stability.json"),
                        "--candidate-model",
                        str(models[1]),
                        "--subject",
                        "sub-002",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("recorded init_sample_size", stderr.getvalue())

    def test_summarize_stability_rejects_different_training_contracts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            reference = root / "reference.model"
            candidate = root / "candidate.model"
            for model, subjects in (
                (reference, ("sub-000", "sub-001")),
                (candidate, ("sub-000", "sub-002")),
            ):
                arguments = [
                    "fit-states",
                    str(store.root),
                    str(model),
                    "--method",
                    "kmeans",
                    "--n-states",
                    "2",
                    "--seed",
                    "17",
                    "--n-init",
                    "2",
                    "--max-iter",
                    "2",
                ]
                for subject in subjects:
                    arguments.extend(("--subject", subject))
                self.assertEqual(main(arguments), 0)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "summarize-stability",
                        str(store.root),
                        str(reference),
                        str(root / "stability.json"),
                        "--candidate-model",
                        str(candidate),
                        "--subject",
                        "sub-003",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("training subjects", stderr.getvalue())

    def test_summarize_stability_rejects_different_training_content(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_store = self._write_state_store(root / "reference.store")
            candidate_store = self._write_state_store(root / "candidate.store")
            chunk_path = candidate_store.root / "chunks" / "00000000.values.npy"
            changed = np.load(chunk_path, allow_pickle=False)
            changed[0, 0] += 0.01
            np.save(chunk_path, changed, allow_pickle=False)
            models = (root / "reference.model", root / "candidate.model")
            for store, model, seed in (
                (reference_store, models[0], 17),
                (candidate_store, models[1], 29),
            ):
                self.assertEqual(
                    main(
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
                            "--n-init",
                            "2",
                            "--max-iter",
                            "2",
                            "--subject",
                            "sub-000",
                            "--subject",
                            "sub-001",
                        ]
                    ),
                    0,
                )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "summarize-stability",
                        str(reference_store.root),
                        str(models[0]),
                        str(root / "stability.json"),
                        "--candidate-model",
                        str(models[1]),
                        "--subject",
                        "sub-002",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("same exact training data", stderr.getvalue())

    def test_summarize_stability_rejects_existing_output_before_model_loading(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "stability.json"
            output.write_text("preserve", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "summarize-stability",
                        str(Path(temporary) / "missing.store"),
                        str(Path(temporary) / "missing-reference.model"),
                        str(output),
                        "--candidate-model",
                        str(Path(temporary) / "missing-candidate.model"),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("already exists", stderr.getvalue())
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

    def test_score_states_writes_heldout_kmeans_report_and_rejects_overlap(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            model = root / "kmeans.model"
            self.assertEqual(
                main(
                    [
                        "fit-states",
                        str(store.root),
                        str(model),
                        "--method",
                        "kmeans",
                        "--n-states",
                        "2",
                        "--seed",
                        "17",
                        "--n-init",
                        "2",
                        "--max-iter",
                        "2",
                        "--subject",
                        "sub-000",
                        "--subject",
                        "sub-001",
                    ]
                ),
                0,
            )
            output = root / "kmeans-scores.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "score-states",
                        str(store.root),
                        str(model),
                        str(output),
                        "--subject",
                        "sub-002",
                        "--subject",
                        "sub-003",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads(stdout.getvalue())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["n_runs"], 2)
            self.assertEqual(summary["n_samples"], 80)
            self.assertEqual(report["model_kind"], "kmeans-state")
            self.assertEqual(report["selection_metric"], "mean_squared_distance")
            self.assertEqual(report["subjects"], ["sub-002", "sub-003"])
            self.assertEqual(report["evaluation_data_fingerprint"], summary[
                "evaluation_data_fingerprint"
            ])
            self.assertEqual(len(report["model_fingerprint"]), 64)
            self.assertEqual(report["format_version"], 2)
            self.assertEqual(len(report["feature_contract_fingerprint"]), 64)
            self.assertEqual(report["model_specification"]["algorithm"], "minibatch")
            self.assertEqual(report["model_specification"]["n_init"], 2)
            self.assertGreater(report["model_specification"]["init_sample_size"], 0)

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                overlap_status = main(
                    [
                        "score-states",
                        str(store.root),
                        str(model),
                        str(root / "overlap.json"),
                        "--subject",
                        "sub-000",
                    ]
                )
            self.assertEqual(overlap_status, 2)
            self.assertIn("overlap", stderr.getvalue())

    @unittest.skipUnless(HAS_HMM_EXTRA, "requires dfc-kit[hmm]")
    def test_score_states_writes_gap_safe_hmm_report(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._write_state_store(root / "features.store")
            model = root / "hmm.model"
            self.assertEqual(
                main(
                    [
                        "fit-states",
                        str(store.root),
                        str(model),
                        "--method",
                        "hmm",
                        "--n-states",
                        "2",
                        "--seed",
                        "17",
                        "--n-pca-components",
                        "2",
                        "--n-init",
                        "1",
                        "--n-iter",
                        "20",
                        "--pca-batch-size",
                        "16",
                        "--subject",
                        "sub-000",
                        "--subject",
                        "sub-001",
                    ]
                ),
                0,
            )
            output = root / "hmm-scores.json"
            self.assertEqual(
                main(
                    [
                        "score-states",
                        str(store.root),
                        str(model),
                        str(output),
                        "--subject",
                        "sub-002",
                    ]
                ),
                0,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["model_kind"], "gaussian-hmm-state")
            self.assertEqual(report["selection_metric"], "log_likelihood_per_sample")
            self.assertEqual(report["minimum_sequence_length"], 2)
            self.assertEqual(report["n_sequences"], 1)
            self.assertLess(report["summary"]["log_likelihood"], 0.0)
            self.assertEqual(report["format_version"], 2)
            self.assertEqual(len(report["feature_contract_fingerprint"]), 64)
            self.assertEqual(report["model_specification"]["covariance_type"], "diag")
            self.assertEqual(report["model_specification"]["n_pca_components"], 2)

    def test_score_states_refuses_existing_output_before_loading_inputs(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "scores.json"
            output.write_text("preserve", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "score-states",
                        str(Path(temporary) / "missing.store"),
                        str(Path(temporary) / "missing.model"),
                        str(output),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("already exists", stderr.getvalue())
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
