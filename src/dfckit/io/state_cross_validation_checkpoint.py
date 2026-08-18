"""Strict per-cell checkpoints for state-count cross-validation grids."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..states.cross_validation import SubjectValidationFold, make_subject_validation_folds
from ..states.hmm import GaussianHMMStateModel
from ..states.kmeans import KMeansStateModel
from ..storage import FeatureStore
from .models import load_fitted_model
from .state_cross_validation import SPLIT_ALGORITHM, StateCountCrossValidationArtifact
from .state_nested_contract import (
    sample_intervals_match,
    validate_state_count_cell_model,
    validate_state_count_cell_score,
)
from .state_scoring import StateModelScoreReport, load_state_model_scores

FORMAT_NAME = "dfckit-state-count-cross-validation-checkpoint"
FORMAT_VERSION = 1
MANIFEST_NAME = "checkpoint.json"


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(
                f"duplicate JSON field in state-count checkpoint: {key}"
            )
        output[key] = value
    return output


def _invalid_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant in state-count checkpoint: {value}")


def _validate_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("state-count checkpoint must be a JSON object")
    expected = {
        "candidate_n_states",
        "development_data_fingerprint",
        "development_subjects",
        "feature_contract_fingerprint",
        "fit_configuration",
        "format",
        "format_version",
        "method",
        "model_kind",
        "model_seeds",
        "sample_interval_seconds",
        "source_contract",
        "split",
    }
    if set(payload) != expected:
        raise ValueError("state-count checkpoint fields do not match the schema")
    if payload["format"] != FORMAT_NAME or payload["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported state-count checkpoint format or version")
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "state-count checkpoint must contain finite JSON values"
        ) from error
    return payload


def load_state_count_checkpoint(path: str | Path) -> dict[str, object]:
    """Load one strict per-cell state-count checkpoint contract."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"state-count checkpoint manifest does not exist: {target}")
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_invalid_constant,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read state-count checkpoint {target}: {error}") from error
    return _validate_payload(payload)


def write_state_count_checkpoint(
    payload: dict[str, object],
    path: str | Path,
) -> Path:
    """Atomically create one per-cell state-count checkpoint contract."""
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"state-count checkpoint already exists: {target}")
    validated = _validate_payload(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(validated, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"state-count checkpoint already exists: {target}")
        os.rename(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


@dataclass(frozen=True)
class StateCountCheckpointPlan:
    """Current-data-bound plan for one complete fold-by-K-by-seed grid."""

    payload: dict[str, object]
    method: str
    model_kind: str
    subjects: tuple[str, ...]
    candidate_n_states: tuple[int, ...]
    model_seeds: tuple[int, ...]
    configuration: dict[str, object]
    minimum_sequence_length: int
    ordered_folds: tuple[tuple[str, SubjectValidationFold], ...]

    @property
    def total_cells(self) -> int:
        return (
            len(self.ordered_folds)
            * len(self.candidate_n_states)
            * len(self.model_seeds)
        )


def make_state_count_checkpoint_plan(
    store: FeatureStore,
    *,
    method: str,
    subjects: tuple[str, ...],
    candidate_n_states: tuple[int, ...],
    model_seeds: tuple[int, ...],
    n_folds: int,
    split_seed: int,
    configuration: dict[str, object],
    minimum_sequence_length: int,
) -> StateCountCheckpointPlan:
    """Build the canonical checkpoint plan from the current FeatureStore."""
    if not isinstance(store, FeatureStore):
        raise TypeError("store must be a FeatureStore")
    if method not in {"kmeans", "hmm"}:
        raise ValueError("state-count checkpoint method must be kmeans or hmm")
    model_kind = "kmeans-state" if method == "kmeans" else "gaussian-hmm-state"
    folds = make_subject_validation_folds(subjects, n_folds=n_folds, seed=split_seed)
    ordered = tuple(
        sorted(
            (
                store.data_fingerprint(
                    subjects=fold.evaluation_subjects,
                    minimum_sequence_length=minimum_sequence_length,
                ),
                fold,
            )
            for fold in folds
        )
    )
    fold_records = []
    for fold_index, (evaluation_fingerprint, fold) in enumerate(ordered):
        fold_records.append(
            {
                "fold_index": fold_index,
                "assignment_index": fold.fold_index,
                "fit_subjects": list(fold.fit_subjects),
                "evaluation_subjects": list(fold.evaluation_subjects),
                "training_data_fingerprint": store.data_fingerprint(
                    subjects=fold.fit_subjects,
                    minimum_sequence_length=minimum_sequence_length,
                ),
                "evaluation_data_fingerprint": evaluation_fingerprint,
            }
        )
    payload = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "method": method,
        "model_kind": model_kind,
        "source_contract": store.source_contract,
        "sample_interval_seconds": store.sample_interval_seconds,
        "feature_contract_fingerprint": store.feature_contract_fingerprint(),
        "development_data_fingerprint": store.data_fingerprint(
            subjects=subjects,
            minimum_sequence_length=minimum_sequence_length,
        ),
        "development_subjects": list(subjects),
        "candidate_n_states": list(candidate_n_states),
        "model_seeds": list(model_seeds),
        "split": {
            "algorithm": SPLIT_ALGORITHM,
            "seed": split_seed,
            "n_folds": n_folds,
            "folds": fold_records,
        },
        "fit_configuration": dict(configuration),
    }
    _validate_payload(payload)
    return StateCountCheckpointPlan(
        payload=payload,
        method=method,
        model_kind=model_kind,
        subjects=subjects,
        candidate_n_states=candidate_n_states,
        model_seeds=model_seeds,
        configuration=dict(configuration),
        minimum_sequence_length=minimum_sequence_length,
        ordered_folds=ordered,
    )


def state_count_checkpoint_cell_paths(
    root: str | Path,
    fold_index: int,
    n_states: int,
    seed: int,
) -> tuple[Path, Path]:
    """Return canonical model and score paths for one checkpoint cell."""
    base = Path(root)
    stem = f"fold-{fold_index + 1:03d}_k-{n_states}_seed-{seed}"
    return base / "models" / f"{stem}.model", base / "scores" / f"{stem}.json"


def _temporary_paths(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob(".*.tmp-*")
        )
    )


def _require_symlink_free_tree(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("state-count checkpoint root must not be a symlink")
    for parent, directories, files in os.walk(root, followlinks=False):
        base = Path(parent)
        for entry in (*directories, *files):
            if (base / entry).is_symlink():
                raise ValueError("state-count checkpoint tree must not contain symlinks")


def validate_state_count_checkpoint_contract(
    root: str | Path,
    plan: StateCountCheckpointPlan,
) -> tuple[str, ...]:
    """Validate the checkpoint manifest, path types, and immediate artifact set."""
    checkpoint = Path(root)
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError("state-count checkpoint must be a real directory")
    _require_symlink_free_tree(checkpoint)
    observed = load_state_count_checkpoint(checkpoint / MANIFEST_NAME)
    if observed != plan.payload:
        raise ValueError(
            "state-count checkpoint does not match the requested workflow or current data"
        )

    expected_models: set[str] = set()
    expected_scores: set[str] = set()
    for fold_index, _ in enumerate(plan.ordered_folds):
        for n_states in plan.candidate_n_states:
            for seed in plan.model_seeds:
                model_path, score_path = state_count_checkpoint_cell_paths(
                    checkpoint,
                    fold_index,
                    n_states,
                    seed,
                )
                expected_models.add(model_path.name)
                expected_scores.add(score_path.name)

    allowed_root = {
        MANIFEST_NAME,
        "comparison.json",
        "models",
        "scores",
        "workflow.json",
    }
    allowed_root_temporaries = {
        MANIFEST_NAME,
        "comparison.json",
        "workflow.json",
    }
    unexpected_root = sorted(
        path.name
        for path in checkpoint.iterdir()
        if path.name not in allowed_root
        and not any(
            path.name.startswith(f".{name}.tmp-")
            for name in allowed_root_temporaries
        )
    )
    if unexpected_root:
        raise ValueError(
            f"state-count checkpoint contains unexpected root paths: {unexpected_root}"
        )

    models_root = checkpoint / "models"
    scores_root = checkpoint / "scores"
    if models_root.exists() and not models_root.is_dir():
        raise ValueError("state-count checkpoint models path must be a directory")
    if scores_root.exists() and not scores_root.is_dir():
        raise ValueError("state-count checkpoint scores path must be a directory")
    if models_root.is_dir():
        unexpected_models = sorted(
            path.name
            for path in models_root.iterdir()
            if path.name not in expected_models
            and not any(
                path.name.startswith(f".{name}.tmp-")
                for name in expected_models
            )
        )
        if unexpected_models:
            raise ValueError(
                f"state-count checkpoint contains unexpected models: {unexpected_models}"
            )
    if scores_root.is_dir():
        unexpected_scores = sorted(
            path.name
            for path in scores_root.iterdir()
            if path.name not in expected_scores
            and not any(
                path.name.startswith(f".{name}.tmp-")
                for name in expected_scores
            )
        )
        if unexpected_scores:
            raise ValueError(
                f"state-count checkpoint contains unexpected scores: {unexpected_scores}"
            )
    for name in ("comparison.json", "workflow.json"):
        path = checkpoint / name
        if path.exists() and not path.is_file():
            raise ValueError(
                f"state-count checkpoint {name} path must be a regular file"
            )
    return _temporary_paths(checkpoint)


def validate_completed_state_count_checkpoint(
    selection: StateCountCrossValidationArtifact,
    store: FeatureStore,
    plan: StateCountCheckpointPlan,
) -> None:
    """Validate a completed selection against its checkpoint plan and current data."""
    if (
        selection.method != plan.method
        or selection.model_kind != plan.model_kind
        or selection.development_subjects != plan.subjects
        or selection.candidate_n_states != plan.candidate_n_states
        or selection.model_seeds != plan.model_seeds
        or selection.split_seed != plan.payload["split"]["seed"]
        or selection.comparison.selection.n_folds
        != plan.payload["split"]["n_folds"]
        or dict(selection.fit_configuration) != plan.configuration
        or selection.source_contract != store.source_contract
        or not sample_intervals_match(
            selection.sample_interval_seconds,
            store.sample_interval_seconds,
        )
        or selection.feature_contract_fingerprint
        != store.feature_contract_fingerprint()
        or selection.development_data_fingerprint
        != plan.payload["development_data_fingerprint"]
    ):
        raise ValueError(
            "completed state-count checkpoint disagrees with its plan or current data"
        )
    if len(selection.comparison.folds) != len(plan.ordered_folds):
        raise ValueError(
            "completed state-count checkpoint has the wrong number of inner folds"
        )
    for comparison_fold, (_, planned_fold) in zip(
        selection.comparison.folds,
        plan.ordered_folds,
        strict=True,
    ):
        if (
            comparison_fold.fit_subjects != planned_fold.fit_subjects
            or comparison_fold.evaluation_subjects
            != planned_fold.evaluation_subjects
            or comparison_fold.training_data_fingerprint
            != store.data_fingerprint(
                subjects=planned_fold.fit_subjects,
                minimum_sequence_length=plan.minimum_sequence_length,
            )
            or comparison_fold.evaluation_data_fingerprint
            != store.data_fingerprint(
                subjects=planned_fold.evaluation_subjects,
                minimum_sequence_length=plan.minimum_sequence_length,
            )
        ):
            raise ValueError(
                "completed state-count checkpoint inner fold disagrees with "
                "its plan or current data"
            )


def load_state_count_checkpoint_cell(
    store: FeatureStore,
    plan: StateCountCheckpointPlan,
    root: str | Path,
    *,
    fold_index: int,
    n_states: int,
    seed: int,
) -> tuple[
    KMeansStateModel | GaussianHMMStateModel | None,
    StateModelScoreReport | None,
]:
    """Load and validate one available model/score pair without mutation."""
    checkpoint = Path(root)
    evaluation_fingerprint, fold = plan.ordered_folds[fold_index]
    model_path, score_path = state_count_checkpoint_cell_paths(
        checkpoint,
        fold_index,
        n_states,
        seed,
    )
    if model_path.is_symlink() or score_path.is_symlink():
        raise ValueError("state-count checkpoint cell paths must not be symlinks")
    if model_path.exists() and not model_path.is_dir():
        raise ValueError("state-count checkpoint model path must be a directory")
    if score_path.exists() and not score_path.is_file():
        raise ValueError("state-count checkpoint score path must be a regular file")
    if score_path.exists() and not model_path.exists():
        raise ValueError("state-count checkpoint score exists without its fitted model")
    if not model_path.exists():
        return None, None

    model = load_fitted_model(model_path)
    if not isinstance(model, (KMeansStateModel, GaussianHMMStateModel)):
        raise TypeError("state-count checkpoint model has the wrong fitted-model type")
    validate_state_count_cell_model(
        model,
        store,
        fold,
        method=plan.method,
        n_states=n_states,
        seed=seed,
        configuration=plan.configuration,
        minimum_sequence_length=plan.minimum_sequence_length,
    )
    if not score_path.exists():
        return model, None
    report = load_state_model_scores(score_path)
    validate_state_count_cell_score(
        report,
        model,
        store,
        fold,
        evaluation_data_fingerprint=evaluation_fingerprint,
        minimum_sequence_length=plan.minimum_sequence_length,
    )
    return model, report


@dataclass(frozen=True)
class StateCountCheckpointCellProgress:
    """Audited availability of one fold-by-K-by-seed cell."""

    fold_index: int
    n_states: int
    model_seed: int
    status: str


@dataclass(frozen=True)
class StateCountCheckpointProgress:
    """Audited per-cell progress for one incomplete state-count workflow."""

    root: Path
    completed_models: int
    completed_scores: int
    total_cells: int
    temporary_paths: tuple[str, ...]
    cells: tuple[StateCountCheckpointCellProgress, ...]


def inspect_state_count_checkpoint(
    store: FeatureStore,
    plan: StateCountCheckpointPlan,
    root: str | Path,
) -> StateCountCheckpointProgress:
    """Audit every available checkpoint cell without cleaning or refitting."""
    checkpoint = Path(root)
    temporary_paths = validate_state_count_checkpoint_contract(checkpoint, plan)
    cells = []
    completed_models = 0
    completed_scores = 0
    for fold_index, _ in enumerate(plan.ordered_folds):
        for n_states in plan.candidate_n_states:
            for seed in plan.model_seeds:
                model, report = load_state_count_checkpoint_cell(
                    store,
                    plan,
                    checkpoint,
                    fold_index=fold_index,
                    n_states=n_states,
                    seed=seed,
                )
                if model is None:
                    status = "pending-model"
                elif report is None:
                    completed_models += 1
                    status = "pending-score"
                else:
                    completed_models += 1
                    completed_scores += 1
                    status = "complete"
                cells.append(
                    StateCountCheckpointCellProgress(
                        fold_index=fold_index,
                        n_states=n_states,
                        model_seed=seed,
                        status=status,
                    )
                )
    return StateCountCheckpointProgress(
        root=checkpoint,
        completed_models=completed_models,
        completed_scores=completed_scores,
        total_cells=plan.total_cells,
        temporary_paths=temporary_paths,
        cells=tuple(cells),
    )


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "StateCountCheckpointCellProgress",
    "StateCountCheckpointPlan",
    "StateCountCheckpointProgress",
    "inspect_state_count_checkpoint",
    "load_state_count_checkpoint",
    "load_state_count_checkpoint_cell",
    "make_state_count_checkpoint_plan",
    "state_count_checkpoint_cell_paths",
    "validate_completed_state_count_checkpoint",
    "validate_state_count_checkpoint_contract",
    "write_state_count_checkpoint",
]
