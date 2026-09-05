"""Small command-line adapters for the XCP-D-first workflows."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence

from .commands import reporting, source, stability, states
from .commands.parser import build_parser

_HANDLERS = {
    "inspect-xcpd": source.inspect_xcpd,
    "build-store": source.build_store,
    "fixed-information": source.fixed_information,
    "lowrank-endpoints": source.lowrank_endpoints,
    "window-pattern-endpoints": source.window_pattern_endpoints,
    "static-fc-endpoints": source.static_fc_endpoints,
    "fit-states": states.fit_states,
    "predict-states": states.predict_states,
    "summarize-states": states.summarize_states,
    "summarize-store": reporting.summarize_store,
    "summarize-information": reporting.summarize_information,
    "describe-states": reporting.describe_states,
    "infer-state-metrics": reporting.infer_state_metrics,
    "infer-paired-endpoints": reporting.infer_endpoints,
    "infer-independent-endpoints": reporting.infer_independent_endpoints,
    "infer-paired-nbs": reporting.infer_nbs,
    "score-states": states.score_states,
    "compare-state-counts": reporting.compare_state_counts,
    "align-states": stability.align_states,
    "summarize-stability": stability.summarize_stability,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and return a process-style status code."""
    parser = build_parser()
    namespace = parser.parse_args(argv)
    try:
        result = _HANDLERS[namespace.command](namespace)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"dfc-kit: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
