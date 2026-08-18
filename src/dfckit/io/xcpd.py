"""XCP-D-first input adapter for parcellated ROI time series."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..data import TimeSeriesDataset, TimeSeriesRun


@dataclass(frozen=True)
class XCPDAtlasFiles:
    """The parcellated time-series and coverage files for one XCP-D atlas."""

    atlas: str
    timeseries: Path
    coverage: Path


@dataclass(frozen=True)
class XCPDRunFiles:
    """XCP-D files that describe one acquisition across one or more atlases."""

    atlases: tuple[XCPDAtlasFiles, ...]
    outliers: Path
    motion: Path | None = None


@dataclass(frozen=True)
class XCPDLoadResult:
    """A validated core run plus XCP-D provenance aligned to its ROI order."""

    run: TimeSeriesRun
    files: XCPDRunFiles
    coverage: NDArray[np.float64]
    source_axes: tuple[str, ...]


def _entity_label(value: str, prefix: str) -> str:
    text = str(value)
    return text if text.startswith(f"{prefix}-") else f"{prefix}-{text}"


def _atlas_names(atlases: str | Sequence[str]) -> tuple[str, ...]:
    names = (atlases,) if isinstance(atlases, str) else tuple(str(name) for name in atlases)
    if not names or any(not name for name in names):
        raise ValueError("at least one non-empty XCP-D atlas name is required")
    if len(set(names)) != len(names):
        raise ValueError("XCP-D atlas names must be unique")
    return names


def _acquisition_stem(timeseries: Path) -> str:
    name = timeseries.name
    if "_space-" in name:
        return name.split("_space-", 1)[0]
    if "_atlas-" in name:
        return name.split("_atlas-", 1)[0]
    raise ValueError(f"not an XCP-D atlas time-series filename: {timeseries}")


def discover_xcpd_files(
    root: str | Path,
    *,
    subject: str,
    atlases: str | Sequence[str],
    session: str | None = None,
    task: str = "rest",
    space: str | None = None,
) -> XCPDRunFiles:
    """Discover one unambiguous XCP-D run using its BIDS derivative structure."""
    root_path = Path(root)
    subject_label = _entity_label(subject, "sub")
    session_label = _entity_label(session, "ses") if session is not None else None
    atlas_names = _atlas_names(atlases)

    relative_func = (
        Path(subject_label) / session_label / "func"
        if session_label is not None
        else Path(subject_label) / "func"
    )
    func_dirs = sorted(path for path in root_path.glob(f"**/{relative_func}") if path.is_dir())
    if not func_dirs:
        raise FileNotFoundError(f"no XCP-D func directory found below {root_path / relative_func}")

    entity_prefix = f"{subject_label}_"
    if session_label is not None:
        entity_prefix += f"{session_label}_"
    entity_prefix += f"task-{task}"

    atlas_files: list[XCPDAtlasFiles] = []
    acquisition_stems: set[str] = set()
    for atlas in atlas_names:
        candidates = []
        suffix = f"_atlas-{atlas}_stat-mean_timeseries.tsv"
        for func_dir in func_dirs:
            for path in func_dir.glob(f"{entity_prefix}*{suffix}"):
                if space is None or f"_space-{space}_" in path.name:
                    candidates.append(path)
        candidates = sorted(set(candidates))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"expected exactly one XCP-D {atlas} time series for {entity_prefix}, "
                f"found {candidates}; specify space when multiple derivatives exist"
            )
        timeseries = candidates[0]
        coverage = timeseries.with_name(
            timeseries.name.removesuffix("_stat-mean_timeseries.tsv")
            + "_stat-coverage_bold.tsv"
        )
        if not coverage.is_file():
            raise FileNotFoundError(f"missing XCP-D atlas coverage file: {coverage}")
        atlas_files.append(XCPDAtlasFiles(atlas, timeseries, coverage))
        acquisition_stems.add(_acquisition_stem(timeseries))

    if len(acquisition_stems) != 1:
        raise ValueError(f"selected XCP-D atlases do not describe one acquisition: {atlas_files}")
    stem = acquisition_stems.pop()
    parent = atlas_files[0].timeseries.parent
    outliers = parent / f"{stem}_outliers.tsv"
    if not outliers.is_file():
        raise FileNotFoundError(f"missing XCP-D outlier mask: {outliers}")
    motion = parent / f"{stem}_motion.tsv"
    return XCPDRunFiles(tuple(atlas_files), outliers, motion if motion.is_file() else None)


def _entity_map(stem: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    for token in stem.split("_"):
        if "-" not in token:
            continue
        key, value = token.split("-", 1)
        if key and value:
            entities[key] = value
    return entities


def discover_xcpd_runs(
    root: str | Path,
    *,
    atlases: str | Sequence[str],
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    space: str | None = None,
) -> tuple[XCPDRunFiles, ...]:
    """Discover every unambiguous acquisition matching optional BIDS filters.

    Unlike :func:`discover_xcpd_files`, this function does not reject multiple
    ``task/acq/run`` entities for one subject/session. Each filename stem is
    returned as a separate :class:`XCPDRunFiles` object, and the loader uses
    that stem as the run's ``acquisition_id``.
    """
    root_path = Path(root)
    atlas_names = _atlas_names(atlases)
    subject_value = None if subject is None else _entity_label(subject, "sub").removeprefix("sub-")
    session_value = None if session is None else _entity_label(session, "ses").removeprefix("ses-")
    first_atlas = atlas_names[0]
    grouped: dict[tuple[Path, str], Path] = {}
    for path in sorted(
        root_path.glob(f"**/func/*_atlas-{first_atlas}_stat-mean_timeseries.tsv")
    ):
        stem = _acquisition_stem(path)
        entities = _entity_map(stem)
        if subject_value is not None and entities.get("sub") != subject_value:
            continue
        if session_value is not None and entities.get("ses") != session_value:
            continue
        if task is not None and entities.get("task") != str(task).removeprefix("task-"):
            continue
        if space is not None and f"_space-{space}_" not in path.name:
            continue
        key = (path.parent, stem)
        if key in grouped:
            raise ValueError(f"duplicate XCP-D acquisition files for {key}")
        grouped[key] = path

    discovered: list[XCPDRunFiles] = []
    for (func_dir, stem), first_path in sorted(grouped.items(), key=lambda item: str(item[0])):
        atlas_prefix = first_path.name.removesuffix(
            f"_atlas-{first_atlas}_stat-mean_timeseries.tsv"
        )
        atlas_files: list[XCPDAtlasFiles] = []
        for atlas in atlas_names:
            path = func_dir / (
                f"{atlas_prefix}_atlas-{atlas}_stat-mean_timeseries.tsv"
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing XCP-D {atlas} time series for acquisition {stem!r}: {path}"
                )
            coverage = path.with_name(
                path.name.removesuffix("_stat-mean_timeseries.tsv")
                + "_stat-coverage_bold.tsv"
            )
            if not coverage.is_file():
                raise FileNotFoundError(f"missing XCP-D atlas coverage file: {coverage}")
            atlas_files.append(XCPDAtlasFiles(atlas, path, coverage))
        outliers = func_dir / f"{stem}_outliers.tsv"
        if not outliers.is_file():
            raise FileNotFoundError(f"missing XCP-D outlier mask: {outliers}")
        motion = func_dir / f"{stem}_motion.tsv"
        discovered.append(XCPDRunFiles(tuple(atlas_files), outliers, motion if motion.is_file() else None))
    if not discovered:
        raise FileNotFoundError(f"no matching XCP-D acquisitions found below {root_path}")
    return tuple(discovered)


def _read_tsv(path: Path) -> tuple[tuple[str, ...], list[list[str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    if not rows or not rows[0] or any(not name for name in rows[0]):
        raise ValueError(f"{path}: missing or invalid TSV header")
    header = tuple(rows[0])
    if len(set(header)) != len(header):
        raise ValueError(f"{path}: duplicate TSV columns")
    width = len(header)
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != width:
            raise ValueError(f"{path}:{row_number}: expected {width} columns, found {len(row)}")
    return header, rows[1:]


def _read_numeric_table(path: Path) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    header, rows = _read_tsv(path)
    values = np.empty((len(rows), len(header)), dtype=float)
    missing = {"", "n/a", "na", "nan"}
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            try:
                values[row_index, column_index] = (
                    np.nan if cell.strip().lower() in missing else float(cell)
                )
            except ValueError as error:
                raise ValueError(
                    f"{path}:{row_index + 2}: non-numeric value in {header[column_index]}"
                ) from error
    return header, values


def _read_outliers(path: Path) -> NDArray[np.bool_]:
    header, rows = _read_tsv(path)
    if len(header) != 1:
        raise ValueError(f"{path}: expected exactly one outlier-mask column")
    mapping = {
        "false": False,
        "true": True,
        "0": False,
        "1": True,
        "no": False,
        "yes": True,
    }
    output = np.empty(len(rows), dtype=bool)
    for row_index, row in enumerate(rows):
        normalized = row[0].strip().lower()
        if normalized not in mapping:
            raise ValueError(f"{path}:{row_index + 2}: invalid binary outlier value {row[0]!r}")
        output[row_index] = mapping[normalized]
    return output


def _read_coverage(path: Path) -> dict[str, float]:
    header, rows = _read_tsv(path)
    required = {"Node", "coverage"}
    if not required.issubset(header):
        raise ValueError(f"{path}: coverage table must contain Node and coverage columns")
    node_index = header.index("Node")
    coverage_index = header.index("coverage")
    output: dict[str, float] = {}
    for row_number, row in enumerate(rows, start=2):
        node = row[node_index]
        if not node or node in output:
            raise ValueError(f"{path}:{row_number}: empty or duplicate coverage node {node!r}")
        try:
            value = float(row[coverage_index])
        except ValueError as error:
            raise ValueError(f"{path}:{row_number}: invalid coverage value") from error
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{path}:{row_number}: coverage must be within [0, 1]")
        output[node] = value
    return output


def _selected_names(
    atlas: str,
    available: tuple[str, ...],
    selections: Mapping[str, Sequence[str]] | None,
) -> tuple[str, ...]:
    if selections is None or atlas not in selections:
        return available
    selected = tuple(str(name) for name in selections[atlas])
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"{atlas}: selected ROI names must be non-empty and unique")
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"{atlas}: selected ROI columns are missing: {missing}")
    return selected


def load_xcpd_files(
    files: XCPDRunFiles,
    *,
    roi_names: Mapping[str, Sequence[str]] | None = None,
    minimum_coverage: float | None = None,
    subject: str | None = None,
    session: str | None = None,
    acquisition_id: str | None = None,
    tr: float | None = None,
) -> XCPDLoadResult:
    """Load XCP-D atlas TSVs and recover censor gaps on the original frame axis."""
    if minimum_coverage is not None and not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be within [0, 1]")
    outliers = _read_outliers(files.outliers)
    retained = np.flatnonzero(~outliers).astype(np.int64)
    if not len(retained):
        raise ValueError(f"{files.outliers}: every frame is marked as an outlier")

    pieces: list[NDArray[np.float64]] = []
    all_names: list[str] = []
    all_coverage: list[float] = []
    source_axes: list[str] = []
    for atlas_files in files.atlases:
        available, values = _read_numeric_table(atlas_files.timeseries)
        selected = _selected_names(atlas_files.atlas, available, roi_names)
        positions = [available.index(name) for name in selected]
        values = values[:, positions]
        if len(values) == len(outliers):
            values = values[~outliers]
            source_axes.append("full-length")
        elif len(values) == len(retained):
            source_axes.append("censored")
        else:
            raise ValueError(
                f"{atlas_files.timeseries}: time-series/outlier mismatch "
                f"({len(values)} rows, {len(outliers)} original frames, "
                f"{len(retained)} retained frames)"
            )

        coverage_map = _read_coverage(atlas_files.coverage)
        missing_coverage = sorted(set(selected) - set(coverage_map))
        if missing_coverage:
            raise ValueError(
                f"{atlas_files.coverage}: selected ROI coverage is missing: {missing_coverage}"
            )
        selected_coverage = [coverage_map[name] for name in selected]
        if minimum_coverage is not None:
            below = [
                f"{name}={value:.6g}"
                for name, value in zip(selected, selected_coverage, strict=True)
                if value < minimum_coverage
            ]
            if below:
                raise ValueError(
                    f"{atlas_files.atlas}: ROI coverage below {minimum_coverage}: {below}"
                )
        duplicate = sorted(set(all_names).intersection(selected))
        if duplicate:
            raise ValueError(f"duplicate ROI names across XCP-D atlases: {duplicate}")
        pieces.append(values)
        all_names.extend(selected)
        all_coverage.extend(selected_coverage)

    acquisition_stem = _acquisition_stem(files.atlases[0].timeseries)
    if subject is None or session is None:
        entities = acquisition_stem.split("_")
        if subject is None:
            subject = next((value for value in entities if value.startswith("sub-")), None)
        if session is None:
            session = next(
                (
                    value.removeprefix("ses-")
                    for value in entities
                    if value.startswith("ses-")
                ),
                None,
            )
    if acquisition_id is None:
        acquisition_id = acquisition_stem
    try:
        run = TimeSeriesRun(
            values=np.column_stack(pieces),
            original_indices=retained,
            roi_names=tuple(all_names),
            subject=subject,
            session=session,
            acquisition_id=acquisition_id,
            tr=tr,
        )
    except ValueError as error:
        raise ValueError(f"invalid XCP-D ROI time series: {error}") from error

    coverage = np.asarray(all_coverage, dtype=float)
    coverage.setflags(write=False)
    return XCPDLoadResult(run, files, coverage, tuple(source_axes))


def load_xcpd_run(
    root: str | Path,
    *,
    subject: str,
    atlases: str | Sequence[str],
    session: str | None = None,
    task: str = "rest",
    space: str | None = None,
    roi_names: Mapping[str, Sequence[str]] | None = None,
    minimum_coverage: float | None = None,
    acquisition_id: str | None = None,
    tr: float | None = None,
) -> XCPDLoadResult:
    """Discover and load one XCP-D run from its derivative root."""
    files = discover_xcpd_files(
        root,
        subject=subject,
        session=session,
        task=task,
        atlases=atlases,
        space=space,
    )
    return load_xcpd_files(
        files,
        roi_names=roi_names,
        minimum_coverage=minimum_coverage,
        subject=_entity_label(subject, "sub"),
        session=session.removeprefix("ses-") if session is not None else None,
        acquisition_id=acquisition_id,
        tr=tr,
    )


def load_xcpd_dataset(
    root: str | Path,
    *,
    atlases: str | Sequence[str],
    subject: str | None = None,
    session: str | None = None,
    task: str | None = None,
    space: str | None = None,
    roi_names: Mapping[str, Sequence[str]] | None = None,
    minimum_coverage: float | None = None,
    tr: float | None = None,
) -> TimeSeriesDataset:
    """Discover and load a subject-safe dataset of XCP-D acquisitions."""
    files = discover_xcpd_runs(
        root,
        atlases=atlases,
        subject=subject,
        session=session,
        task=task,
        space=space,
    )
    loaded = tuple(
        load_xcpd_files(
            item,
            roi_names=roi_names,
            minimum_coverage=minimum_coverage,
            tr=tr,
        )
        for item in files
    )
    try:
        return TimeSeriesDataset(tuple(item.run for item in loaded))
    except ValueError as error:
        raise ValueError(f"invalid XCP-D dataset: {error}") from error
