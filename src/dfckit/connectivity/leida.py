"""Censor-aware Leading Eigenvector Dynamics Analysis (LEiDA)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..data import TimeSeriesRun


def _readonly(values: NDArray) -> NDArray:
    output = np.asarray(values).copy()
    output.setflags(write=False)
    return output


def _validated_phase(values: ArrayLike) -> NDArray[np.float64]:
    phase = np.asarray(values, dtype=float)
    if phase.ndim != 2 or not len(phase) or phase.shape[1] < 2:
        raise ValueError("phase must be a non-empty frames-by-ROI array with at least two ROIs")
    if not np.isfinite(phase).all():
        raise ValueError("phase contains non-finite values")
    return phase


def _validated_nodes(
    nodes: Iterable[int],
    *,
    n_rois: int,
    label: str,
    minimum: int,
) -> NDArray[np.int64]:
    raw = tuple(nodes)
    if len(raw) < minimum:
        raise ValueError(f"{label} must contain at least {minimum} ROI indices")
    if any(
        isinstance(node, (bool, np.bool_)) or not isinstance(node, (int, np.integer))
        for node in raw
    ):
        raise TypeError(f"{label} must contain integer ROI indices")
    output = np.asarray(raw, dtype=np.int64)
    if len(set(output.tolist())) != len(output):
        raise ValueError(f"{label} contains duplicate ROI indices")
    if np.any(output < 0) or np.any(output >= n_rois):
        raise ValueError(f"{label} contains an ROI index outside [0, {n_rois})")
    return output


@dataclass(frozen=True)
class LEiDAResult:
    """Instantaneous phase and leading phase-coherence geometry for one run."""

    phase: NDArray[np.float64]
    leading_vectors: NDArray[np.float64]
    leading_eigenvalues: NDArray[np.float64]
    original_indices: NDArray[np.int64]
    segment_ids: NDArray[np.int64]
    roi_names: tuple[str, ...]
    subject: str | None
    session: str | None
    tr: float | None
    minimum_segment_length: int
    orientation: str
    implementation: str
    acquisition_id: str | None = None


def analytic_phase(values: ArrayLike) -> NDArray[np.float64]:
    """Return Hilbert phase after centering each ROI over one segment."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[0] < 3 or data.shape[1] < 2:
        raise ValueError("values must have at least three frames and two ROIs")
    if not np.isfinite(data).all():
        raise ValueError("values contain non-finite samples")
    scale = data.std(axis=0, ddof=0)
    invalid = np.flatnonzero((~np.isfinite(scale)) | (scale <= 1e-12))
    if len(invalid):
        raise ValueError(f"Hilbert phase is undefined for constant ROI indices {invalid.tolist()}")
    try:
        from scipy.signal import hilbert
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "LEiDA phase extraction requires the 'phase' extra: "
            "pip install 'dfc-kit[phase]'"
        ) from error
    centered = data - data.mean(axis=0)
    return _readonly(np.angle(hilbert(centered, axis=0)))


def leading_phase_eigenvectors(
    phase: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return oriented leading eigenvectors of instantaneous phase coherence.

    The matrix with entries ``cos(phase_i - phase_j)`` has rank at most two.
    Solving its equivalent two-dimensional eigensystem avoids materializing a
    frames-by-ROI-by-ROI tensor.
    """
    angles = _validated_phase(phase)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    cosine_energy = np.sum(cosine**2, axis=1)
    sine_energy = np.sum(sine**2, axis=1)
    cross = np.sum(cosine * sine, axis=1)
    direction = 0.5 * np.arctan2(
        2.0 * cross,
        cosine_energy - sine_energy,
    )
    vectors = (
        cosine * np.cos(direction)[:, None]
        + sine * np.sin(direction)[:, None]
    )
    norm = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 1e-12):
        raise ValueError("leading phase eigenvector has undefined norm")
    vectors /= norm[:, None]
    vectors *= np.where(vectors.sum(axis=1, keepdims=True) < 0.0, -1.0, 1.0)

    discriminant = np.sqrt(
        (cosine_energy - sine_energy) ** 2 + 4.0 * cross**2
    )
    eigenvalues = (cosine_energy + sine_energy + discriminant) / 2.0
    return _readonly(vectors), _readonly(eigenvalues)


def cross_block_phase_coherence(
    phase: ArrayLike,
    left: Iterable[int],
    right: Iterable[int],
) -> NDArray[np.float64]:
    """Return mean instantaneous phase coherence between two disjoint blocks."""
    angles = _validated_phase(phase)
    left_nodes = _validated_nodes(
        left,
        n_rois=angles.shape[1],
        label="left block",
        minimum=1,
    )
    right_nodes = _validated_nodes(
        right,
        n_rois=angles.shape[1],
        label="right block",
        minimum=1,
    )
    overlap = sorted(set(left_nodes.tolist()).intersection(right_nodes.tolist()))
    if overlap:
        raise ValueError(f"cross-block phase coherence requires disjoint blocks; overlap={overlap}")
    values = np.cos(
        angles[:, left_nodes, None] - angles[:, None, right_nodes]
    ).mean(axis=(1, 2))
    return _readonly(values)


def within_block_phase_coherence(
    phase: ArrayLike,
    nodes: Iterable[int],
) -> NDArray[np.float64]:
    """Return off-diagonal mean instantaneous phase coherence within a block."""
    angles = _validated_phase(phase)
    selected = _validated_nodes(
        nodes,
        n_rois=angles.shape[1],
        label="within block",
        minimum=2,
    )
    left, right = np.triu_indices(len(selected), k=1)
    values = np.cos(
        angles[:, selected[left]] - angles[:, selected[right]]
    ).mean(axis=1)
    return _readonly(values)


class LEiDA:
    """Extract leading phase-coherence vectors inside retained segments."""

    def __init__(self, minimum_segment_length: int = 20) -> None:
        if isinstance(minimum_segment_length, (bool, np.bool_)) or not isinstance(
            minimum_segment_length, (int, np.integer)
        ):
            raise TypeError("minimum_segment_length must be an integer")
        if minimum_segment_length < 3:
            raise ValueError("minimum_segment_length must be at least three")
        self.minimum_segment_length = int(minimum_segment_length)

    def transform(self, run: TimeSeriesRun) -> LEiDAResult:
        phases: list[NDArray[np.float64]] = []
        vectors: list[NDArray[np.float64]] = []
        eigenvalues: list[NDArray[np.float64]] = []
        original_indices: list[NDArray[np.int64]] = []
        segment_ids: list[NDArray[np.int64]] = []

        for segment_id, positions in enumerate(run.segments()):
            if len(positions) < self.minimum_segment_length:
                continue
            segment_phase = analytic_phase(run.values[positions])
            segment_vectors, segment_eigenvalues = leading_phase_eigenvectors(segment_phase)
            phases.append(segment_phase)
            vectors.append(segment_vectors)
            eigenvalues.append(segment_eigenvalues)
            original_indices.append(run.original_indices[positions])
            segment_ids.append(np.full(len(positions), segment_id, dtype=np.int64))

        if not phases:
            raise ValueError(
                "no retained segment meets the requested minimum_segment_length"
            )
        try:
            import scipy
        except ModuleNotFoundError as error:  # pragma: no cover - analytic_phase fails first
            raise ModuleNotFoundError(
                "LEiDA phase extraction requires the 'phase' extra: "
                "pip install 'dfc-kit[phase]'"
            ) from error
        return LEiDAResult(
            phase=_readonly(np.concatenate(phases, axis=0)),
            leading_vectors=_readonly(np.concatenate(vectors, axis=0)),
            leading_eigenvalues=_readonly(np.concatenate(eigenvalues)),
            original_indices=_readonly(np.concatenate(original_indices)),
            segment_ids=_readonly(np.concatenate(segment_ids)),
            roi_names=run.roi_names,
            subject=run.subject,
            session=run.session,
            acquisition_id=run.acquisition_id,
            tr=run.tr,
            minimum_segment_length=self.minimum_segment_length,
            orientation="positive-vector-sum",
            implementation=f"scipy {scipy.__version__} signal.hilbert",
        )
