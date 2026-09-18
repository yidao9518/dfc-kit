"""Co-activation pattern feature construction and KMeans fitting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._arrays import readonly_copy
from .._preprocessing import _standardized_samples
from ..data import TimeSeriesDataset
from .data import (
    FeatureSequence,
    FeatureSequenceDataset,
    StateAssignments,
    _segment_feature_sequences,
)
from .kmeans import KMeansFitResult, fit_kmeans_states


def cap_sequences(
    dataset: TimeSeriesDataset,
    *,
    standardization: str = "run",
) -> FeatureSequenceDataset:
    """Create instantaneous ROI patterns from all retained frames.

    Each run is ROI-wise z-scored over its retained frames by default (ddof=0).
    ``standardization='segment'`` instead scales each contiguous segment and
    omits singleton segments. Both modes keep censor gaps as sequence boundaries.
    """
    dataset.require_subject_ids("CAP state modeling")
    feature_keys = tuple((roi,) for roi in dataset.roi_names)
    sequences: list[FeatureSequence] = []
    for run in dataset.runs:
        assert run.subject is not None
        standardized, original_indices, segment_ids = _standardized_samples(
            run, standardization=standardization, method_name="CAP"
        )
        sequences.extend(
            _segment_feature_sequences(
                run, standardized, original_indices, original_indices, segment_ids,
                feature_keys=feature_keys,
                source_contract=f"cap:within-{standardization}-roi-zscore-ddof0",
                interval=run.tr,
            )
        )
    return FeatureSequenceDataset(sequences)


def fit_cap_states(
    dataset: TimeSeriesDataset,
    *,
    n_states: int,
    seed: int,
    n_init: int = 20,
    max_iter: int = 300,
    standardization: str = "run",
    algorithm: str = "lloyd",
    batch_size: int = 4096,
    reassignment_ratio: float = 0.01,
) -> KMeansFitResult:
    """Fit Euclidean KMeans to standardized instantaneous ROI patterns.

    The default uses full-batch Lloyd KMeans, with no PCA or second feature
    standardization. ``algorithm='minibatch'`` enables approximate fitting.
    Use :func:`cap_state_maps` for exact frame-average activity maps under the
    final assignments, rather than assuming estimated centers equal those maps.
    """
    return fit_kmeans_states(
        cap_sequences(dataset, standardization=standardization),
        n_states=n_states,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        algorithm=algorithm,
        standardize_features=False,
        batch_size=batch_size,
        reassignment_ratio=reassignment_ratio,
    )


def cap_state_maps(
    features: FeatureSequenceDataset,
    assignments: StateAssignments,
) -> NDArray[np.float64]:
    """Average standardized ROI activity over the frames assigned to each CAP.

    Returns a read-only states-by-ROIs array in ``features.feature_keys`` order.
    Frames have equal weight; states with no assigned frames have NaN rows.
    Sequences may be reordered, but acquisition identities and frame indices
    must match. These descriptive maps do not change the fitted model centers.
    """
    if not features.source_contract.startswith("cap:") or any(
        len(key) != 1 for key in features.feature_keys
    ):
        raise ValueError("CAP maps require CAP ROI features from cap_sequences or a CAP store")
    if features.source_contract != assignments.source_contract:
        raise ValueError("CAP maps require matching standardization source contracts")
    left, right = features.sample_interval_seconds, assignments.sample_interval_seconds
    if (left is None) != (right is None) or (
        left is not None and right is not None and not np.isclose(left, right, rtol=0, atol=1e-9)
    ):
        raise ValueError("CAP maps require matching sample intervals")

    def identity(sequence):
        return sequence.subject, sequence.session, sequence.acquisition_id, sequence.segment_id

    labels_by_sequence = {identity(sequence): sequence for sequence in assignments.sequences}
    if {identity(sequence) for sequence in features.sequences} != set(labels_by_sequence):
        raise ValueError("CAP maps require matching sequence identities")
    sums = np.zeros((assignments.n_states, len(features.feature_keys)), dtype=float)
    counts = np.zeros(assignments.n_states, dtype=np.int64)
    for sequence in features.sequences:
        assigned = labels_by_sequence[identity(sequence)]
        if not (
            np.array_equal(sequence.sample_start_indices, assigned.sample_start_indices)
            and np.array_equal(sequence.sample_end_indices, assigned.sample_end_indices)
        ):
            raise ValueError("CAP maps require matching frame indices")
        np.add.at(sums, assigned.labels, sequence.values)
        counts += np.bincount(assigned.labels, minlength=assignments.n_states)
    maps = np.full(sums.shape, np.nan)
    np.divide(sums, counts[:, None], out=maps, where=counts[:, None] > 0)
    return readonly_copy(maps)
