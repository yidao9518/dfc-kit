"""Paired and group-level statistical inference."""

from .endpoints import (
    infer_paired_endpoints,
    infer_paired_endpoints_file,
    write_paired_endpoint_inference,
)
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
from .result_families import (
    adjust_result_families,
    adjust_result_families_file,
    write_adjusted_result_families,
)
from .state_metrics import (
    SUPPORTED_STATE_METRICS,
    infer_paired_state_metrics,
    infer_paired_state_metrics_file,
    write_paired_state_inference,
)

__all__ = [
    "SUPPORTED_STATE_METRICS",
    "BootstrapCIResult",
    "FDRResult",
    "HC3Result",
    "NBSComponent",
    "NBSThresholdResult",
    "PairedHC3Result",
    "PairedNBSResult",
    "PairedSignFlipResult",
    "adjust_result_families",
    "adjust_result_families_file",
    "benjamini_hochberg",
    "infer_paired_endpoints",
    "infer_paired_endpoints_file",
    "infer_paired_state_metrics",
    "infer_paired_state_metrics_file",
    "intercept_t_statistic",
    "ols_hc3",
    "paired_bootstrap_mean_ci",
    "paired_hc3",
    "paired_nbs",
    "paired_sign_flip",
    "threshold_components",
    "write_adjusted_result_families",
    "write_paired_endpoint_inference",
    "write_paired_state_inference",
]
