"""Paired and group-level statistical inference."""

from .hc3 import HC3Result, PairedHC3Result, ols_hc3, paired_hc3
from .multiple_testing import FDRResult, benjamini_hochberg
from .nbs import (
    NBSComponent,
    NBSThresholdResult,
    PairedNBSResult,
    intercept_t_statistic,
    paired_nbs,
    threshold_components,
)
from .paired import (
    BootstrapCIResult,
    PairedSignFlipResult,
    paired_bootstrap_mean_ci,
    paired_sign_flip,
)

__all__ = [
    "BootstrapCIResult",
    "FDRResult",
    "HC3Result",
    "NBSComponent",
    "NBSThresholdResult",
    "PairedHC3Result",
    "PairedNBSResult",
    "PairedSignFlipResult",
    "benjamini_hochberg",
    "intercept_t_statistic",
    "ols_hc3",
    "paired_bootstrap_mean_ci",
    "paired_hc3",
    "paired_nbs",
    "paired_sign_flip",
    "threshold_components",
]
