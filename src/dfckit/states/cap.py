"""Co-activation pattern feature construction and KMeans fitting."""

from __future__ import annotations

from .._preprocessing import _segment_standardized_samples
from ..data import TimeSeriesDataset
from .data import FeatureSequence, FeatureSequenceDataset, _segment_feature_sequences
from .kmeans import KMeansFitResult, fit_kmeans_states


def cap_sequences(dataset: TimeSeriesDataset) -> FeatureSequenceDataset:
    """Create segment-standardized instantaneous ROI patterns from XCP-D runs."""
    dataset.require_subject_ids("CAP state modeling")
    feature_keys = tuple((roi,) for roi in dataset.roi_names)
    sequences: list[FeatureSequence] = []
    for run in dataset.runs:
        assert run.subject is not None
        standardized, original_indices, segment_ids = _segment_standardized_samples(
            run, method_name="CAP"
        )
        sequences.extend(
            _segment_feature_sequences(
                run, standardized, original_indices, original_indices, segment_ids,
                feature_keys=feature_keys, source_contract="cap:within-segment-roi-zscore-ddof0",
                interval=run.tr,
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
