"""State models and label-alignment utilities."""

from .alignment import (
    StateAlignment,
    align_gaussian_hmm_emissions,
    align_kmeans_centroids,
    apply_gaussian_hmm_alignment,
    apply_state_alignment,
    relabel_gaussian_hmm_model,
    relabel_kmeans_model,
)
from .cap import cap_sequences, fit_cap_states
from .cross_validation import SubjectValidationFold, make_subject_validation_folds
from .data import (
    FeatureSequence,
    FeatureSequenceDataset,
    StateAssignments,
    StateLabelSequence,
    leida_sequences,
    timeseries_sequences,
    window_fc_sequences,
)
from .hmm import (
    GaussianHMMFitResult,
    GaussianHMMStateModel,
    GaussianHMMStateResult,
    fit_gaussian_hmm_states,
    predict_gaussian_hmm_states,
    reconstruct_emission_covariance,
)
from .kmeans import (
    KMeansFitResult,
    KMeansStateModel,
    fit_kmeans_states,
    predict_kmeans_states,
)
from .metrics import RunStateMetrics, summarize_state_assignments
from .scoring import RunGaussianHMMScore, RunKMeansScore, RunStateModelScore
from .selection import StateCountSelection, select_state_count
from .stability import RunStateStability, summarize_state_stability

__all__ = [
    "FeatureSequence",
    "FeatureSequenceDataset",
    "GaussianHMMFitResult",
    "GaussianHMMStateModel",
    "GaussianHMMStateResult",
    "KMeansFitResult",
    "KMeansStateModel",
    "RunGaussianHMMScore",
    "RunKMeansScore",
    "RunStateMetrics",
    "RunStateModelScore",
    "RunStateStability",
    "StateAlignment",
    "StateAssignments",
    "StateCountSelection",
    "StateLabelSequence",
    "SubjectValidationFold",
    "align_gaussian_hmm_emissions",
    "align_kmeans_centroids",
    "apply_gaussian_hmm_alignment",
    "apply_state_alignment",
    "cap_sequences",
    "fit_cap_states",
    "fit_gaussian_hmm_states",
    "fit_kmeans_states",
    "leida_sequences",
    "make_subject_validation_folds",
    "predict_gaussian_hmm_states",
    "predict_kmeans_states",
    "reconstruct_emission_covariance",
    "relabel_gaussian_hmm_model",
    "relabel_kmeans_model",
    "select_state_count",
    "summarize_state_assignments",
    "summarize_state_stability",
    "timeseries_sequences",
    "window_fc_sequences",
]
