"""Command handlers that turn XCP-D outputs into analysis artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ..connectivity import ETS, MTD, LEiDA, SlidingWindowFC
from ..data import TimeSeriesDataset
from ..information import (
    compute_fixed_information,
    load_fixed_window_schedule,
    load_information_groups,
    save_fixed_information,
)
from ..io import discover_xcpd_runs, load_xcpd_dataset
from ..storage import (
    write_cap_store,
    write_instantaneous_edge_store,
    write_leida_store,
    write_window_fc_store,
)


def load_roi_selection(path: Path | None) -> dict[str, tuple[str, ...]] | None:
    """Load and validate an atlas-to-ROI selection mapping."""
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read ROI selection JSON {path}: {error}") from error
    if not isinstance(raw, dict) or not raw:
        raise ValueError("ROI selection must be a non-empty JSON object")
    output: dict[str, tuple[str, ...]] = {}
    for atlas, names in raw.items():
        if not isinstance(atlas, str) or not atlas.strip():
            raise ValueError("ROI selection atlas names must be non-empty strings")
        if isinstance(names, str) or not isinstance(names, Sequence) or not names:
            raise ValueError(f"ROI selection for {atlas!r} must be a non-empty array")
        selected = tuple(str(name) for name in names)
        if any(not name.strip() for name in selected) or len(set(selected)) != len(selected):
            raise ValueError(f"ROI selection for {atlas!r} must contain unique non-empty names")
        output[atlas] = selected
    return output


def xcpd_filters(
    namespace: argparse.Namespace,
    *,
    subject: str | None = None,
) -> dict[str, object]:
    """Build the discovery filters shared by XCP-D commands."""
    return {
        "atlases": tuple(namespace.atlas),
        "subject": subject,
        "session": namespace.session,
        "task": namespace.task,
        "space": namespace.space,
    }


def selected_xcpd_subjects(namespace: argparse.Namespace) -> tuple[str | None, ...]:
    """Normalize an optional repeated ``--subject`` selection."""
    subjects = namespace.subject
    if subjects is None:
        return (None,)
    values = tuple(str(subject).strip().removeprefix("sub-") for subject in subjects)
    if any(not subject for subject in values):
        raise ValueError("--subject values must be non-empty")
    normalized = tuple(f"sub-{subject}" for subject in values)
    unique = tuple(dict.fromkeys(normalized))
    if len(unique) != len(subjects):
        raise ValueError("--subject values must be unique")
    return unique


def discover_selected_xcpd_runs(namespace: argparse.Namespace):
    """Discover runs for the selected subject set."""
    discovered = []
    for subject in selected_xcpd_subjects(namespace):
        discovered.extend(
            discover_xcpd_runs(namespace.root, **xcpd_filters(namespace, subject=subject))
        )
    return tuple(discovered)


def inspect_xcpd(namespace: argparse.Namespace) -> dict[str, object]:
    """Describe matching XCP-D acquisitions without loading their time series."""
    files = discover_selected_xcpd_runs(namespace)
    acquisitions = []
    for item in files:
        stem = item.outliers.name.removesuffix("_outliers.tsv")
        entities = {
            token.split("-", 1)[0]: token.split("-", 1)[1]
            for token in stem.split("_")
            if "-" in token
        }
        acquisitions.append(
            {
                "acquisition_id": stem,
                "subject": entities.get("sub"),
                "session": entities.get("ses"),
                "task": entities.get("task"),
                "atlases": [atlas.atlas for atlas in item.atlases],
                "files": {
                    atlas.atlas: {
                        "timeseries": str(atlas.timeseries),
                        "coverage": str(atlas.coverage),
                    }
                    for atlas in item.atlases
                },
                "outliers": str(item.outliers),
                "motion": None if item.motion is None else str(item.motion),
            }
        )
    return {"n_acquisitions": len(acquisitions), "acquisitions": acquisitions}


def _load_selected_dataset(namespace: argparse.Namespace) -> TimeSeriesDataset:
    selection = load_roi_selection(namespace.roi_selection)
    datasets = tuple(
        load_xcpd_dataset(
            namespace.root,
            roi_names=selection,
            minimum_coverage=namespace.minimum_coverage,
            tr=namespace.tr,
            **xcpd_filters(namespace, subject=subject),
        )
        for subject in selected_xcpd_subjects(namespace)
    )
    return TimeSeriesDataset(tuple(run for item in datasets for run in item.runs))


def build_store(namespace: argparse.Namespace) -> dict[str, object]:
    """Build a disk-backed feature store from selected XCP-D runs."""
    dataset = _load_selected_dataset(namespace)
    if namespace.method == "window-fc":
        estimator = SlidingWindowFC(
            length=namespace.window_length,
            step=namespace.window_step,
            taper=namespace.taper,
        )
        store = write_window_fc_store(
            namespace.output,
            dataset.runs,
            estimator,
            chunk_size=namespace.chunk_size,
            dtype=namespace.dtype,
        )
    elif namespace.method == "cap":
        store = write_cap_store(
            namespace.output,
            dataset.runs,
            chunk_size=namespace.chunk_size,
            dtype=namespace.dtype,
        )
    elif namespace.method == "leida":
        store = write_leida_store(
            namespace.output,
            dataset.runs,
            LEiDA(minimum_segment_length=namespace.minimum_segment_length),
            chunk_size=namespace.chunk_size,
            dtype=namespace.dtype,
        )
    elif namespace.method == "ets":
        store = write_instantaneous_edge_store(
            namespace.output,
            dataset.runs,
            ETS(),
            chunk_size=namespace.chunk_size,
            dtype=namespace.dtype,
        )
    else:
        store = write_instantaneous_edge_store(
            namespace.output,
            dataset.runs,
            MTD(),
            chunk_size=namespace.chunk_size,
            dtype=namespace.dtype,
        )
    return {
        "method": namespace.method,
        "output": str(store.root),
        "n_runs": dataset.n_runs,
        "subjects": list(dataset.subjects),
        "acquisition_ids": [run.acquisition_id for run in dataset.runs],
        "n_sequences": store.n_sequences,
        "n_samples": store.n_samples,
        "n_chunks": store.n_chunks,
        "n_features": store.n_features,
        "format_version": store.format_version,
    }


def fixed_information(namespace: argparse.Namespace) -> dict[str, object]:
    """Compute and persist fixed-length MI and CMI endpoints."""
    if namespace.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    dataset = _load_selected_dataset(namespace)
    groups = load_information_groups(namespace.information_groups)
    schedule = (
        None
        if namespace.window_schedule is None
        else load_fixed_window_schedule(namespace.window_schedule)
    )
    artifact = compute_fixed_information(
        dataset,
        groups,
        lengths=namespace.length,
        draws=namespace.draws,
        sample_seed=namespace.sample_seed,
        schedule=schedule,
        schedule_source=(
            None if namespace.window_schedule is None else str(namespace.window_schedule)
        ),
        k=namespace.k,
        jitter=namespace.jitter,
        jitter_seed=namespace.jitter_seed,
        standardize=namespace.standardize,
        jobs=namespace.jobs,
    )
    output = save_fixed_information(artifact, namespace.output)
    return {
        "acquisition_ids": [item.acquisition_id for item in artifact.acquisitions],
        "conditioning_rois": (
            [] if artifact.groups.conditioning is None else list(artifact.groups.conditioning)
        ),
        "draws_per_length": artifact.draws_per_length,
        "format_version": artifact.format_version,
        "has_cmi": artifact.has_cmi,
        "left_rois": list(artifact.groups.left),
        "lengths": list(artifact.lengths),
        "method": "fixed-information",
        "n_draws": artifact.n_draws,
        "n_runs": len(artifact.acquisitions),
        "n_cells": artifact.n_cells,
        "jobs": namespace.jobs,
        "output": str(output),
        "right_rois": list(artifact.groups.right),
        "schedule_mode": artifact.schedule_mode,
        "subjects": list(dataset.subjects),
    }
