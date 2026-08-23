"""Paired and group-level statistical inference."""

from .endpoints import (
    infer_paired_endpoints,
)
from .hc3 import HC3Result, PairedHC3Result, ols_hc3, paired_hc3
from .matching import (
    MatchedSubjectDifferences,
    WindowMotionSummary,
    WithinSubjectMatchResult,
    match_within_subject,
    matched_subject_differences,
    summarize_window_motion,
)
from .multiple_testing import FDRResult, benjamini_hochberg
from .nbs import (
    NBSComponent,
    NBSThresholdResult,
    PairedNBSResult,
    paired_nbs,
)
from .paired import (
    BootstrapCIResult,
    PairedSignFlipResult,
    paired_bootstrap_mean_ci,
    paired_sign_flip,
)
from .state_metrics import (
    infer_paired_state_metrics,
)

__all__ = [
    "BootstrapCIResult",
    "FDRResult",
    "HC3Result",
    "MatchedSubjectDifferences",
    "NBSComponent",
    "NBSThresholdResult",
    "PairedHC3Result",
    "PairedNBSResult",
    "PairedSignFlipResult",
    "WindowMotionSummary",
    "WithinSubjectMatchResult",
    "benjamini_hochberg",
    "infer_paired_endpoints",
    "infer_paired_state_metrics",
    "match_within_subject",
    "matched_subject_differences",
    "ols_hc3",
    "paired_bootstrap_mean_ci",
    "paired_hc3",
    "paired_nbs",
    "paired_sign_flip",
    "summarize_window_motion",
]
