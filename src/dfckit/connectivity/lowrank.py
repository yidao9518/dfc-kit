"""Low-rank covariance geometry inside censor-bounded time windows."""

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


def _validated_values(values: ArrayLike, *, label: str) -> NDArray[np.float64]:
    output = np.asarray(values, dtype=float)
    if output.ndim != 2 or output.shape[0] < 2 or output.shape[1] < 2:
        raise ValueError(f"{label} must have at least two frames and two ROIs")
    if not np.isfinite(output).all():
        raise ValueError(f"{label} contain non-finite samples")
    return output


def _validated_rank(rank: int, *, n_frames: int, n_rois: int) -> int:
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
        raise TypeError("rank must be an integer")
    maximum = min(n_frames - 1, n_rois)
    if rank < 1 or rank > maximum:
        raise ValueError(f"rank must be between 1 and {maximum} for these values")
    return int(rank)


def _validated_basis(values: ArrayLike, *, label: str) -> NDArray[np.float64]:
    basis = np.asarray(values, dtype=float)
    if basis.ndim != 2 or basis.shape[0] < 2 or basis.shape[1] < 1:
        raise ValueError(f"{label} must be a two-dimensional ROI-by-rank basis")
    if not np.isfinite(basis).all():
        raise ValueError(f"{label} contains non-finite values")
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(basis.shape[1]), rtol=1e-7, atol=1e-9):
        raise ValueError(f"{label} columns must be orthonormal")
    return basis


@dataclass(frozen=True)
class StandardizedPCA:
    """PCA fitted after independent population-SD scaling of every ROI."""

    basis: NDArray[np.float64]
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    variance: NDArray[np.float64]
    variance_proportion: NDArray[np.float64]
    n_frames: int
    n_rois: int

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def standardize(self, values: ArrayLike) -> NDArray[np.float64]:
        data = _validated_values(values, label="values")
        if data.shape[1] != self.n_rois:
            raise ValueError("values use a different number of ROIs from the PCA fit")
        return _readonly((data - self.mean) / self.scale)

    def reconstruct_standardized(self, values: ArrayLike) -> NDArray[np.float64]:
        standardized = self.standardize(values)
        return _readonly(standardized @ self.basis @ self.basis.T)


@dataclass(frozen=True)
class LowRankCovarianceResult:
    """Window and run summaries for one low-rank covariance transform."""

    effective_rank: NDArray[np.float64]
    eigen_concentration: NDArray[np.float64]
    heldout_r2: NDArray[np.float64]
    split_similarity: NDArray[np.float64]
    adjacent_similarity: NDArray[np.float64]
    window_bases: tuple[NDArray[np.float64], ...]
    run_bases: tuple[NDArray[np.float64], ...]
    adjacent_left_windows: NDArray[np.int64]
    adjacent_right_windows: NDArray[np.int64]
    start_frames: NDArray[np.int64]
    end_frames: NDArray[np.int64]
    segment_ids: NDArray[np.int64]
    ranks: tuple[int, ...]
    roi_names: tuple[str, ...]
    subject: str | None
    session: str | None
    tr: float | None
    length: int
    step: int
    split: int
    acquisition_id: str | None = None

    def rank_index(self, rank: int) -> int:
        """Return the column/tuple index associated with a requested rank."""
        try:
            return self.ranks.index(rank)
        except ValueError as error:
            raise KeyError(f"rank {rank} was not fitted; available ranks={self.ranks}") from error


def fit_standardized_pca(values: ArrayLike, rank: int) -> StandardizedPCA:
    """Fit PCA after centering and population-SD scaling each ROI.

    ``variance`` contains all non-trivial standardized covariance eigenvalues,
    represented as squared singular values. Their common covariance divisor is
    omitted because it cancels in every variance proportion.
    """
    data = _validated_values(values, label="values")
    fitted_rank = _validated_rank(rank, n_frames=data.shape[0], n_rois=data.shape[1])
    mean = data.mean(axis=0)
    scale = data.std(axis=0, ddof=0)
    invalid = np.flatnonzero((~np.isfinite(scale)) | (scale <= 1e-12))
    if len(invalid):
        raise ValueError(f"PCA scale is undefined for ROI indices {invalid.tolist()}")
    standardized = (data - mean) / scale
    _, singular, transposed_basis = np.linalg.svd(standardized, full_matrices=False)
    variance = singular**2
    total = float(variance.sum())
    if not np.isfinite(total) or total <= 1e-12:
        raise ValueError("standardized PCA has no finite variance")
    proportion = variance / total
    return StandardizedPCA(
        basis=_readonly(transposed_basis[:fitted_rank].T),
        mean=_readonly(mean),
        scale=_readonly(scale),
        variance=_readonly(variance),
        variance_proportion=_readonly(proportion),
        n_frames=int(data.shape[0]),
        n_rois=int(data.shape[1]),
    )


def effective_rank(variance_proportion: ArrayLike) -> float:
    """Return entropy-based effective rank from normalized component weights."""
    proportion = np.asarray(variance_proportion, dtype=float)
    if proportion.ndim != 1 or not len(proportion):
        raise ValueError("variance_proportion must be a non-empty one-dimensional array")
    if not np.isfinite(proportion).all() or np.any(proportion < 0):
        raise ValueError("variance_proportion must be finite and non-negative")
    total = float(proportion.sum())
    if not np.isclose(total, 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("variance_proportion must sum to one")
    positive = proportion[proportion > 0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def eigen_concentration(variance_proportion: ArrayLike, rank: int) -> float:
    """Return the fraction of standardized covariance variance in the top components."""
    proportion = np.asarray(variance_proportion, dtype=float)
    if proportion.ndim != 1 or not len(proportion):
        raise ValueError("variance_proportion must be a non-empty one-dimensional array")
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
        raise TypeError("rank must be an integer")
    if rank < 1 or rank > len(proportion):
        raise ValueError(f"rank must be between 1 and {len(proportion)}")
    if not np.isfinite(proportion).all() or np.any(proportion < 0):
        raise ValueError("variance_proportion must be finite and non-negative")
    if not np.isclose(proportion.sum(), 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("variance_proportion must sum to one")
    return float(proportion[:rank].sum())


def heldout_reconstruction_r2(train: ArrayLike, test: ArrayLike, rank: int) -> float:
    """Fit on ``train`` and reconstruct standardized ``test`` samples."""
    training = _validated_values(train, label="train")
    testing = _validated_values(test, label="test")
    if training.shape[1] != testing.shape[1]:
        raise ValueError("train and test must use the same number of ROIs")
    fit = fit_standardized_pca(training, rank)
    standardized = fit.standardize(testing)
    reconstructed = standardized @ fit.basis @ fit.basis.T
    denominator = float(np.sum(standardized**2))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("held-out standardized values have no finite energy")
    residual = float(np.sum((standardized - reconstructed) ** 2))
    return float(1.0 - residual / denominator)


def bidirectional_heldout_r2(
    values: ArrayLike,
    rank: int,
    *,
    split: int | None = None,
) -> float:
    """Average first-to-second and second-to-first held-out reconstruction R2."""
    data = _validated_values(values, label="values")
    boundary = data.shape[0] // 2 if split is None else split
    if isinstance(boundary, (bool, np.bool_)) or not isinstance(boundary, (int, np.integer)):
        raise TypeError("split must be an integer")
    if boundary < 2 or data.shape[0] - boundary < 2:
        raise ValueError("split must leave at least two frames on each side")
    left, right = data[:boundary], data[boundary:]
    return float(
        np.mean(
            [
                heldout_reconstruction_r2(left, right, rank),
                heldout_reconstruction_r2(right, left, rank),
            ]
        )
    )


def subspace_similarity(left: ArrayLike, right: ArrayLike) -> float:
    """Return mean squared cosine between two equal-rank orthonormal subspaces."""
    left_basis = _validated_basis(left, label="left basis")
    right_basis = _validated_basis(right, label="right basis")
    if left_basis.shape != right_basis.shape:
        raise ValueError("subspace bases must have the same ROI and rank dimensions")
    similarity = float(np.sum((left_basis.T @ right_basis) ** 2) / left_basis.shape[1])
    return float(np.clip(similarity, 0.0, 1.0))


def subspace_distance(left: ArrayLike, right: ArrayLike) -> float:
    """Return square-root projection distance scaled to the interval [0, 1]."""
    return float(np.sqrt(max(0.0, 1.0 - subspace_similarity(left, right))))


def mean_projector_basis(
    bases: Iterable[ArrayLike],
    rank: int | None = None,
) -> NDArray[np.float64]:
    """Average projection matrices and return their leading eigenvectors."""
    validated = tuple(
        _validated_basis(basis, label=f"basis {index}")
        for index, basis in enumerate(bases)
    )
    if not validated:
        raise ValueError("at least one basis is required")
    shape = validated[0].shape
    if any(basis.shape != shape for basis in validated[1:]):
        raise ValueError("all bases must have the same ROI and rank dimensions")
    output_rank = shape[1] if rank is None else rank
    if isinstance(output_rank, (bool, np.bool_)) or not isinstance(
        output_rank, (int, np.integer)
    ):
        raise TypeError("rank must be an integer")
    if output_rank < 1 or output_rank > shape[0]:
        raise ValueError(f"rank must be between 1 and {shape[0]}")
    projector = np.mean([basis @ basis.T for basis in validated], axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh((projector + projector.T) / 2.0)
    order = np.argsort(eigenvalues)[::-1]
    return _readonly(eigenvectors[:, order[:output_rank]])


class LowRankCovariance:
    """Summarize standardized covariance geometry in censor-bounded windows."""

    def __init__(
        self,
        length: int,
        step: int,
        ranks: Iterable[int] = (1, 2, 4),
        *,
        split: int | None = None,
    ) -> None:
        if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)):
            raise TypeError("length must be an integer")
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)):
            raise TypeError("step must be an integer")
        if length < 4:
            raise ValueError("length must be at least four")
        if step < 1:
            raise ValueError("step must be at least one")
        boundary = length // 2 if split is None else split
        if isinstance(boundary, (bool, np.bool_)) or not isinstance(boundary, (int, np.integer)):
            raise TypeError("split must be an integer")
        if boundary < 2 or length - boundary < 2:
            raise ValueError("split must leave at least two frames on each side")

        raw_ranks = tuple(ranks)
        if not raw_ranks:
            raise ValueError("ranks must contain at least one rank")
        if any(
            isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer))
            for rank in raw_ranks
        ):
            raise TypeError("ranks must contain only integers")
        normalized_ranks = tuple(int(rank) for rank in raw_ranks)
        if len(set(normalized_ranks)) != len(normalized_ranks):
            raise ValueError("ranks must be unique")
        if tuple(sorted(normalized_ranks)) != normalized_ranks:
            raise ValueError("ranks must be in strictly increasing order")
        maximum = min(boundary - 1, length - boundary - 1)
        if normalized_ranks[0] < 1 or normalized_ranks[-1] > maximum:
            raise ValueError(
                f"ranks must be between 1 and {maximum} for the requested split"
            )

        self.length = int(length)
        self.step = int(step)
        self.ranks = normalized_ranks
        self.split = int(boundary)

    def transform(self, run: TimeSeriesRun) -> LowRankCovarianceResult:
        windows = run.windows(self.length, self.step)
        if not windows:
            raise ValueError("no contiguous segment is long enough for the requested window")
        if self.ranks[-1] > run.n_rois:
            raise ValueError(
                f"maximum rank {self.ranks[-1]} exceeds the run's {run.n_rois} ROIs"
            )

        n_windows = len(windows)
        n_ranks = len(self.ranks)
        rank_values = np.empty(n_windows, dtype=float)
        concentrations = np.empty((n_windows, n_ranks), dtype=float)
        heldout = np.empty((n_windows, n_ranks), dtype=float)
        split_stability = np.empty((n_windows, n_ranks), dtype=float)
        bases: list[list[NDArray[np.float64]]] = [[] for _ in self.ranks]

        for window_index, window in enumerate(windows):
            full_fit = fit_standardized_pca(window.values, self.ranks[-1])
            rank_values[window_index] = effective_rank(full_fit.variance_proportion)
            left = window.values[: self.split]
            right = window.values[self.split :]
            for rank_index, rank in enumerate(self.ranks):
                concentrations[window_index, rank_index] = eigen_concentration(
                    full_fit.variance_proportion, rank
                )
                heldout[window_index, rank_index] = bidirectional_heldout_r2(
                    window.values, rank, split=self.split
                )
                left_basis = fit_standardized_pca(left, rank).basis
                right_basis = fit_standardized_pca(right, rank).basis
                split_stability[window_index, rank_index] = subspace_similarity(
                    left_basis, right_basis
                )
                bases[rank_index].append(full_fit.basis[:, :rank])

        adjacent_pairs = tuple(
            (left, right)
            for left, right in zip(range(n_windows - 1), range(1, n_windows), strict=True)
            if windows[left].segment_id == windows[right].segment_id
        )
        adjacent = np.empty((len(adjacent_pairs), n_ranks), dtype=float)
        for pair_index, (left, right) in enumerate(adjacent_pairs):
            for rank_index in range(n_ranks):
                adjacent[pair_index, rank_index] = subspace_similarity(
                    bases[rank_index][left], bases[rank_index][right]
                )

        stacked_bases = tuple(_readonly(np.stack(per_rank)) for per_rank in bases)
        run_bases = tuple(
            mean_projector_basis(per_rank, rank=rank)
            for per_rank, rank in zip(bases, self.ranks, strict=True)
        )
        return LowRankCovarianceResult(
            effective_rank=_readonly(rank_values),
            eigen_concentration=_readonly(concentrations),
            heldout_r2=_readonly(heldout),
            split_similarity=_readonly(split_stability),
            adjacent_similarity=_readonly(adjacent),
            window_bases=stacked_bases,
            run_bases=run_bases,
            adjacent_left_windows=_readonly(
                np.asarray([left for left, _ in adjacent_pairs], dtype=np.int64)
            ),
            adjacent_right_windows=_readonly(
                np.asarray([right for _, right in adjacent_pairs], dtype=np.int64)
            ),
            start_frames=_readonly(
                np.asarray([window.start_frame for window in windows], dtype=np.int64)
            ),
            end_frames=_readonly(
                np.asarray([window.end_frame for window in windows], dtype=np.int64)
            ),
            segment_ids=_readonly(
                np.asarray([window.segment_id for window in windows], dtype=np.int64)
            ),
            ranks=self.ranks,
            roi_names=run.roi_names,
            subject=run.subject,
            session=run.session,
            acquisition_id=run.acquisition_id,
            tr=run.tr,
            length=self.length,
            step=self.step,
            split=self.split,
        )
