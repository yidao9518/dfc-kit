"""Co-activation pattern feature construction and KMeans fitting."""

from __future__ import annotations

import numpy as np

from ..data import TimeSeriesDataset
from .data import FeatureSequence, FeatureSequenceDataset
from .kmeans import KMeansFitResult, fit_kmeans_states


def cap_sequences(dataset: TimeSeriesDataset) -> FeatureSequenceDataset:
    """Create segment-standardized instantaneous ROI patterns from XCP-D runs."""
    dataset.require_subject_ids("CAP state modeling")
    feature_keys = tuple((roi,) for roi in dataset.roi_names)
    sequences: list[FeatureSequence] = []
    for run in dataset.runs:
        assert run.subject is not None
        for segment_id, positions in enumerate(run.segments()):
            if len(positions) < 2:
                continue
            values = run.values[positions]
            scale = values.std(axis=0, ddof=0)
            scale = np.where(scale < 1e-8, 1.0, scale)
            standardized = (values - values.mean(axis=0)) / scale
            original = run.original_indices[positions]
            sequences.append(
                FeatureSequence(
                    values=standardized,
                    sample_start_indices=original,
                    sample_end_indices=original,
                    feature_keys=feature_keys,
                    subject=run.subject,
                    session=run.session,
                    acquisition_id=run.acquisition_id,
                    segment_id=segment_id,
                    source_contract="cap:within-segment-roi-zscore-ddof0",
                    sample_interval_seconds=run.tr,
                )
            )
    if not sequences:
        raise ValueError("CAP requires at least one retained segment with two frames")
    return FeatureSequenceDataset(sequences)


def fit_cap_states(
    dataset: TimeSeriesDataset,
    *,
    n_states: int,
    seed: int,
    n_init: int = 20,
    max_iter: int = 300,
) -> KMeansFitResult:
    """Fit KMeans to segment-standardized instantaneous ROI patterns."""
    return fit_kmeans_states(
        cap_sequences(dataset),
        n_states=n_states,
        seed=seed,
        n_init=n_init,
        max_iter=max_iter,
        algorithm="minibatch",
        standardize_features=False,
    )
