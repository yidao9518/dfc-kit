"""Small command-line adapters for the XCP-D-first workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .commands.parser import build_parser
from .commands.reporting import (
    compare_state_counts as _compare_state_counts,
)
from .commands.reporting import (
    describe_states as _describe_states,
)
from .commands.reporting import (
    infer_endpoints as _infer_endpoints,
)
from .commands.reporting import (
    infer_state_metrics as _infer_state_metrics,
)
from .commands.reporting import (
    summarize_information as _summarize_information,
)
from .commands.reporting import (
    summarize_store as _summarize_store,
)
from .commands.source import (
    build_store as _build_store,
)
from .commands.source import (
    fixed_information as _fixed_information,
)
from .commands.source import (
    inspect_xcpd as _inspect,
)
from .commands.stability import (
    align_states as _align_states,
)
from .commands.stability import (
    summarize_stability as _summarize_stability,
)
from .commands.states import (
    fit_states as _fit_states,
)
from .commands.states import (
    predict_states as _predict_states,
)
from .commands.states import (
    score_states as _score_states,
)
from .commands.states import (
    summarize_states as _summarize_states,
)


def _parser() -> argparse.ArgumentParser:
    return build_parser()


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and return a process-style status code."""
    parser = _parser()
    namespace = parser.parse_args(argv)
    try:
        if namespace.command == "inspect-xcpd":
            result = _inspect(namespace)
        elif namespace.command == "build-store":
            result = _build_store(namespace)
        elif namespace.command == "fixed-information":
            result = _fixed_information(namespace)
        elif namespace.command == "fit-states":
            result = _fit_states(namespace)
        elif namespace.command == "predict-states":
            result = _predict_states(namespace)
        elif namespace.command == "summarize-states":
            result = _summarize_states(namespace)
        elif namespace.command == "summarize-store":
            result = _summarize_store(namespace)
        elif namespace.command == "summarize-information":
            result = _summarize_information(namespace)
        elif namespace.command == "describe-states":
            result = _describe_states(namespace)
        elif namespace.command == "infer-state-metrics":
            result = _infer_state_metrics(namespace)
        elif namespace.command == "infer-paired-endpoints":
            result = _infer_endpoints(namespace)
        elif namespace.command == "score-states":
            result = _score_states(namespace)
        elif namespace.command == "compare-state-counts":
            result = _compare_state_counts(namespace)
        elif namespace.command == "align-states":
            result = _align_states(namespace)
        else:
            result = _summarize_stability(namespace)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"dfc-kit: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
