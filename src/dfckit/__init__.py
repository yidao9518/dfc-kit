"""Censor-aware dynamic functional connectivity tools."""

from .data import TimeSeriesDataset, TimeSeriesRun, TimeWindow, validate_subject_disjoint
from .reference import (
    FeatureReferenceModel,
    SubspaceReferenceModel,
    SubspaceReferenceScores,
    fit_feature_reference,
    fit_subspace_reference,
    leave_one_subject_out_feature_similarity,
    pearson_pattern_similarity,
    reference_subspace_similarity,
    score_feature_reference,
    score_subspace_reference,
    subject_balanced_quantiles,
)

__all__ = [
    "FeatureReferenceModel",
    "SubspaceReferenceModel",
    "SubspaceReferenceScores",
    "TimeSeriesDataset",
    "TimeSeriesRun",
    "TimeWindow",
    "fit_feature_reference",
    "fit_subspace_reference",
    "leave_one_subject_out_feature_similarity",
    "pearson_pattern_similarity",
    "reference_subspace_similarity",
    "score_feature_reference",
    "score_subspace_reference",
    "subject_balanced_quantiles",
    "validate_subject_disjoint",
]
__version__ = "0.28.0"
