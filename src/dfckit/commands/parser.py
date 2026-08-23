"""Argument parser construction for the dfc-kit command line."""

from __future__ import annotations

import argparse
from pathlib import Path


def _add_xcpd_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("root", type=Path, help="XCP-D derivatives root")
    parser.add_argument(
        "--atlas",
        action="append",
        required=True,
        help="atlas name; repeat for multiple atlases",
    )
    parser.add_argument(
        "--subject",
        action="append",
        help="optional BIDS subject label or value; repeat to select multiple subjects",
    )
    parser.add_argument("--session", help="optional BIDS session label or value")
    parser.add_argument("--task", help="optional BIDS task label or value")
    parser.add_argument("--space", help="optional XCP-D standard-space label")


def _add_load_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--roi-selection",
        type=Path,
        help="JSON object mapping atlas names to ordered ROI-name arrays",
    )
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=None,
        help="reject selected ROIs below this XCP-D coverage fraction",
    )
    parser.add_argument("--tr", type=float, default=None, help="run TR in seconds")


def build_parser() -> argparse.ArgumentParser:
    """Build the complete command-line parser."""
    parser = argparse.ArgumentParser(
        prog="dfc-kit",
        description="Censor-aware XCP-D to dynamic-connectivity workflows",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser(
        "inspect-xcpd",
        help="list matching XCP-D acquisitions without loading time series",
    )
    _add_xcpd_arguments(inspect)

    build = subparsers.add_parser(
        "build-store",
        help="load XCP-D acquisitions and stream a feature store",
    )
    _add_xcpd_arguments(build)
    _add_load_arguments(build)
    build.add_argument("output", type=Path, help="new FeatureStore directory")
    build.add_argument(
        "--method",
        choices=("window-fc", "cap", "ets", "leida", "mtd"),
        required=True,
    )
    build.add_argument("--chunk-size", type=int, default=128)
    build.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    build.add_argument("--window-length", type=int, default=56)
    build.add_argument("--window-step", type=int, default=8)
    build.add_argument("--taper", choices=("hamming", "uniform"), default="hamming")
    build.add_argument("--minimum-segment-length", type=int, default=20)

    information = subparsers.add_parser(
        "fixed-information",
        help="estimate fixed-length block MI/CMI from XCP-D acquisitions",
    )
    _add_xcpd_arguments(information)
    _add_load_arguments(information)
    information.add_argument(
        "output",
        type=Path,
        help="new fixed-information artifact directory",
    )
    information.add_argument(
        "--information-groups",
        type=Path,
        required=True,
        help="standalone JSON declaring named left, right, and conditioning ROI groups",
    )
    information.add_argument(
        "--length",
        action="append",
        type=int,
        required=True,
        help="fixed retained-frame length; repeat for a sensitivity set",
    )
    information.add_argument("--draws", type=int, required=True)
    information.add_argument("--sample-seed", type=int, required=True)
    information.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="validated execution parallelism (default: 1)",
    )
    information.add_argument(
        "--window-schedule",
        type=Path,
        help="optional strict TSV freezing acquisition, length, draw, and original frame bounds",
    )
    information.add_argument("--k", type=int, default=3)
    information.add_argument("--jitter", type=float, default=1e-10)
    information.add_argument("--jitter-seed", type=int, default=20260811)
    information.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="z-score each ROI within each fixed window (default: enabled)",
    )

    fit = subparsers.add_parser(
        "fit-states",
        help="fit a portable KMeans or Gaussian HMM model from a FeatureStore",
    )
    fit.add_argument("store", type=Path, help="existing FeatureStore directory")
    fit.add_argument("output", type=Path, help="new fitted-model artifact directory")
    fit.add_argument("--method", choices=("kmeans", "hmm"), required=True)
    fit.add_argument("--n-states", type=int, required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument(
        "--subject",
        action="append",
        help="fit only this subject; repeat for an explicit training cohort",
    )
    fit.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="initializations (default: 10 for KMeans, 1 for HMM)",
    )
    fit.add_argument(
        "--standardize-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    fit.add_argument("--fitting-mode", choices=("streaming", "materialized"), default="streaming")
    fit.add_argument("--algorithm", choices=("lloyd", "minibatch"), default="minibatch")
    fit.add_argument("--batch-size", type=int, default=4096)
    fit.add_argument("--max-iter", type=int, default=10)
    fit.add_argument("--reassignment-ratio", type=float, default=0.01)
    fit.add_argument("--init-sample-size", type=int, default=None)
    fit.add_argument(
        "--n-pca-components",
        type=int,
        default=None,
        help="optional PCA dimension for KMeans; required for HMM",
    )
    fit.add_argument("--covariance-type", choices=("diag", "full"), default="diag")
    fit.add_argument("--n-iter", type=int, default=200)
    fit.add_argument("--tol", type=float, default=1e-3)
    fit.add_argument("--pca-batch-size", type=int, default=4096)
    fit.add_argument("--minimum-sequence-length", type=int, default=2)

    predict = subparsers.add_parser(
        "predict-states",
        help="decode a FeatureStore with a saved KMeans or Gaussian HMM",
    )
    predict.add_argument("store", type=Path, help="FeatureStore to decode")
    predict.add_argument("model", type=Path, help="saved fitted-model artifact")
    predict.add_argument("output", type=Path, help="new state-prediction artifact directory")
    predict.add_argument(
        "--subject",
        action="append",
        help="decode only this subject; repeat for an explicit held-out cohort",
    )
    predict.add_argument(
        "--allow-fit-subjects",
        action="store_true",
        help="explicitly allow decoding subjects used during model fitting",
    )

    summarize = subparsers.add_parser(
        "summarize-states",
        help="write occupancy, dwell, switch, and transition metrics as JSON",
    )
    summarize.add_argument("predictions", type=Path, help="state-prediction artifact directory")
    summarize.add_argument("output", type=Path, help="new state-metrics JSON file")

    summarize_features = subparsers.add_parser(
        "summarize-store",
        help="write acquisition-level statistics for every named FeatureStore feature",
    )
    summarize_features.add_argument("store", type=Path, help="FeatureStore to summarize")
    summarize_features.add_argument("output", type=Path, help="new endpoint-summary JSON file")
    summarize_features.add_argument(
        "--statistic",
        action="append",
        choices=("mean", "variance", "standard_deviation", "minimum", "maximum"),
        default=None,
        help="statistic to compute; repeat for multiple statistics (default: mean)",
    )

    summarize_information = subparsers.add_parser(
        "summarize-information",
        help="extract acquisition-level MI and CMI endpoints from an audited artifact",
    )
    summarize_information.add_argument("artifact", type=Path)
    summarize_information.add_argument("output", type=Path)

    describe = subparsers.add_parser(
        "describe-states",
        help="describe state patterns using named ROI or edge features",
    )
    describe.add_argument("store", type=Path, help="FeatureStore used to fit the model")
    describe.add_argument("model", type=Path, help="saved KMeans or Gaussian HMM artifact")
    describe.add_argument("output", type=Path, help="new state-description JSON file")
    describe.add_argument("--top-features", type=int, default=10)
    describe.add_argument(
        "--network-map",
        type=Path,
        help="optional JSON object mapping every model ROI to a network label",
    )

    infer_states = subparsers.add_parser(
        "infer-state-metrics",
        help="run paired sign-flip inference and declared-family FDR over state metrics",
    )
    infer_states.add_argument("metrics", type=Path, help="state-metrics JSON file")
    infer_states.add_argument("output", type=Path, help="new inference artifact directory")
    infer_states.add_argument("--condition-a", required=True)
    infer_states.add_argument("--condition-b", required=True)
    infer_states.add_argument(
        "--metric",
        action="append",
        choices=(
            "occupancy",
            "mean_dwell_seconds",
            "switch_rate",
            "transition_probabilities",
        ),
        required=True,
    )
    infer_states.add_argument("--fdr-family", required=True)
    infer_states.add_argument("--alpha", type=float, default=0.05)
    infer_states.add_argument("--permutations", type=int, default=10_000)
    infer_states.add_argument("--bootstrap", type=int, default=10_000)
    infer_states.add_argument("--seed", type=int, required=True)
    infer_states.add_argument("--exact", action="store_true")

    infer_endpoints = subparsers.add_parser(
        "infer-paired-endpoints",
        help="run paired sign-flip inference over a named endpoint JSON artifact",
    )
    infer_endpoints.add_argument("endpoints", type=Path)
    infer_endpoints.add_argument("output", type=Path)
    infer_endpoints.add_argument("--condition-a", required=True)
    infer_endpoints.add_argument("--condition-b", required=True)
    infer_endpoints.add_argument("--fdr-family", required=True)
    infer_endpoints.add_argument("--alpha", type=float, default=0.05)
    infer_endpoints.add_argument("--permutations", type=int, default=10_000)
    infer_endpoints.add_argument("--bootstrap", type=int, default=10_000)
    infer_endpoints.add_argument("--seed", type=int, required=True)
    infer_endpoints.add_argument("--exact", action="store_true")

    score = subparsers.add_parser(
        "score-states",
        help="score a saved KMeans or HMM on held-out FeatureStore acquisitions",
    )
    score.add_argument("store", type=Path, help="FeatureStore to score")
    score.add_argument("model", type=Path, help="saved fitted-model artifact")
    score.add_argument("output", type=Path, help="new held-out score JSON file")
    score.add_argument(
        "--subject",
        action="append",
        help="score only this subject; repeat for an explicit held-out cohort",
    )
    score.add_argument(
        "--allow-fit-subjects",
        action="store_true",
        help="explicitly allow scoring subjects used during model fitting",
    )

    compare = subparsers.add_parser(
        "compare-state-counts",
        help="compare complete candidate-state sets across subject-disjoint folds",
    )
    compare.add_argument("output", type=Path, help="new state-count comparison JSON file")
    compare.add_argument(
        "--score",
        action="append",
        type=Path,
        required=True,
        help=(
            "score-states JSON artifact; repeat for every fold, state count, "
            "and complete-fit seed"
        ),
    )

    align = subparsers.add_parser(
        "align-states",
        help="align a candidate KMeans/HMM model and optional predictions to a reference",
    )
    align.add_argument("reference_model", type=Path, help="reference fitted-model artifact")
    align.add_argument("candidate_model", type=Path, help="candidate fitted-model artifact")
    align.add_argument("output_model", type=Path, help="new relabeled model artifact")
    align.add_argument("alignment_output", type=Path, help="new alignment artifact directory")
    align.add_argument(
        "--alignment-metric",
        choices=("auto", "euclidean", "pearson"),
        default="auto",
        help="state-pattern matching metric (default: auto; CAP uses pearson)",
    )
    align.add_argument(
        "--predictions",
        type=Path,
        help="candidate state-prediction artifact to relabel",
    )
    align.add_argument(
        "--predictions-output",
        type=Path,
        help="new aligned state-prediction artifact directory",
    )
    stability = subparsers.add_parser(
        "summarize-stability",
        help="decode, align, and summarize repeated-seed state models",
    )
    stability.add_argument("store", type=Path, help="FeatureStore to decode identically")
    stability.add_argument("reference_model", type=Path, help="reference fitted-model artifact")
    stability.add_argument("output", type=Path, help="new repeated-fit stability JSON file")
    stability.add_argument(
        "--candidate-model",
        action="append",
        type=Path,
        required=True,
        help="candidate fitted model; repeat for every additional seed",
    )
    stability.add_argument(
        "--alignment-metric",
        choices=("auto", "euclidean", "pearson"),
        default="auto",
        help="state-pattern matching metric (default: auto; CAP uses pearson)",
    )
    stability.add_argument(
        "--subject",
        action="append",
        help="decode only this subject; repeat for an explicit held-out cohort",
    )
    stability.add_argument(
        "--allow-fit-subjects",
        action="store_true",
        help="explicitly allow decoding subjects used during model fitting",
    )
    return parser
