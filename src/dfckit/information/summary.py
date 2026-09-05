"""Acquisition-level endpoint extraction from fixed-information artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts._json import write_json_atomic
from .fixed import load_fixed_information


def summarize_information_artifact(path: str | Path) -> dict[str, Any]:
    artifact = load_fixed_information(path)
    rows = []
    for cell in artifact.cells:
        selected = np.flatnonzero(
            (artifact.acquisition_index == cell.acquisition_index)
            & (artifact.length == cell.length)
        )
        acquisition = artifact.acquisitions[cell.acquisition_index]
        base = {
            "subject": acquisition.subject,
            "session": acquisition.session,
            "acquisition_id": acquisition.acquisition_id,
            "length": cell.length,
            "n_draws": len(selected),
        }
        rows.append(
            {
                **base,
                "endpoint": f"mean_mi.length_{cell.length}",
                "measure": "mean_mi",
                "value": float(np.mean(artifact.mean_mutual_information[selected])),
            }
        )
        rows.append(
            {
                **base,
                "endpoint": f"mean_cmi.length_{cell.length}",
                "measure": "mean_cmi",
                "value": (
                    None
                    if artifact.mean_conditional_mutual_information is None
                    else float(np.mean(artifact.mean_conditional_mutual_information[selected]))
                ),
            }
        )
    return {
        "format": "dfc-kit-information-endpoints",
        "format_version": 1,
        "source_contract": "fixed-length MI/CMI within retained segments",
        "groups": {
            "left": list(artifact.groups.left),
            "right": list(artifact.groups.right),
            "conditioning": (
                None if artifact.groups.conditioning is None else list(artifact.groups.conditioning)
            ),
        },
        "lengths": list(artifact.lengths),
        "rows": rows,
    }


def write_information_summary(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"information-summary output already exists: {target}")
    return write_json_atomic(target, payload)


__all__ = ["summarize_information_artifact", "write_information_summary"]
