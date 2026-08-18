"""Across-fit stability summaries for already reference-aligned state labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import StateAssignments, _readonly
from .metrics import RunStateMetrics, summarize_state_assignments


@dataclass(frozen=True)
class RunStateStability:
    """Per-run state metrics stacked in a declared fitted-model order."""

    subject: str
    session: str | None
    acquisition_id: str | None
    n_samples: int
    n_sequences: int
    occupancy_by_fit: NDArray[np.float64]
    mean_dwell_samples_by_fit: NDArray[np.float64]
    mean_dwell_seconds_by_fit: NDArray[np.float64]
    switch_rate_by_fit: NDArray[np.float64]
    transition_probabilities_by_fit: NDArray[np.float64]

    def __post_init__(self) -> None:
        arrays = {
            "occupancy_by_fit": self.occupancy_by_fit,
            "mean_dwell_samples_by_fit": self.mean_dwell_samples_by_fit,
            "mean_dwell_seconds_by_fit": self.mean_dwell_seconds_by_fit,
            "switch_rate_by_fit": self.switch_rate_by_fit,
            "transition_probabilities_by_fit": self.transition_probabilities_by_fit,
        }
        converted = {name: np.asarray(value, dtype=float) for name, value in arrays.items()}
        occupancy = converted["occupancy_by_fit"]
        if occupancy.ndim != 2 or occupancy.shape[0] < 2 or occupancy.shape[1] < 2:
            raise ValueError("state stability requires at least two fits and two states")
        expected = occupancy.shape
        if converted["mean_dwell_samples_by_fit"].shape != expected:
            raise ValueError("state stability dwell shape does not match occupancy")
        if converted["mean_dwell_seconds_by_fit"].shape != expected:
            raise ValueError("state stability dwell-seconds shape does not match occupancy")
        if converted["switch_rate_by_fit"].shape != (expected[0],):
            raise ValueError("state stability switch-rate shape does not match fit count")
        if converted["transition_probabilities_by_fit"].shape != (
            expected[0],
            expected[1],
            expected[1],
        ):
            raise ValueError("state stability transition shape is invalid")
        if not np.isfinite(occupancy).all():
            raise ValueError("state stability occupancy must be finite")
        if not str(self.subject).strip() or self.n_samples < 1 or self.n_sequences < 1:
            raise ValueError("state stability run metadata is invalid")
        for name, array in converted.items():
            object.__setattr__(self, name, _readonly(array))

    @property
    def n_fits(self) -> int:
        return int(self.occupancy_by_fit.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.occupancy_by_fit.shape[1])


def _identity(metric: RunStateMetrics) -> tuple[str, str | None, str | None]:
    return metric.subject, metric.session, metric.acquisition_id


def _sampling_boundaries_match(
    reference: StateAssignments,
    candidate: StateAssignments,
) -> bool:
    if len(reference.sequences) != len(candidate.sequences):
        return False
    for expected, observed in zip(reference.sequences, candidate.sequences, strict=True):
        expected_identity = (
            expected.subject,
            expected.session,
            expected.acquisition_id,
            expected.segment_id,
        )
        observed_identity = (
            observed.subject,
            observed.session,
            observed.acquisition_id,
            observed.segment_id,
        )
        if expected_identity != observed_identity:
            return False
        if not np.array_equal(
            expected.sample_start_indices,
            observed.sample_start_indices,
        ) or not np.array_equal(
            expected.sample_end_indices,
            observed.sample_end_indices,
        ):
            return False
    return True


def summarize_state_stability(
    assignments_by_fit: Sequence[StateAssignments],
) -> tuple[RunStateStability, ...]:
    """Stack gap-safe run metrics after callers establish common state numbering."""
    assignments = tuple(assignments_by_fit)
    if len(assignments) < 2:
        raise ValueError("state stability requires at least two fitted assignments")
    reference = assignments[0]
    for candidate in assignments[1:]:
        if candidate.n_states != reference.n_states:
            raise ValueError("state stability requires identical n_states")
        if candidate.source_contract != reference.source_contract:
            raise ValueError("state stability requires identical source contracts")
        intervals_match = (
            candidate.sample_interval_seconds is None
            and reference.sample_interval_seconds is None
        ) or (
            candidate.sample_interval_seconds is not None
            and reference.sample_interval_seconds is not None
            and np.isclose(
                candidate.sample_interval_seconds,
                reference.sample_interval_seconds,
                rtol=0.0,
                atol=1e-9,
            )
        )
        if not intervals_match:
            raise ValueError("state stability requires identical sample intervals")

    metrics_by_fit = tuple(summarize_state_assignments(item) for item in assignments)
    reference_metrics = metrics_by_fit[0]
    reference_identities = tuple(_identity(metric) for metric in reference_metrics)
    for metrics in metrics_by_fit[1:]:
        if tuple(_identity(metric) for metric in metrics) != reference_identities:
            raise ValueError("state stability requires identical run identities and order")
        for expected, observed in zip(reference_metrics, metrics, strict=True):
            if (
                observed.n_samples != expected.n_samples
                or observed.n_sequences != expected.n_sequences
                or observed.n_possible_transitions != expected.n_possible_transitions
            ):
                raise ValueError("state stability requires identical run sampling boundaries")
    for candidate in assignments[1:]:
        if not _sampling_boundaries_match(reference, candidate):
            raise ValueError("state stability requires identical sequence sampling boundaries")

    output = []
    for run_index, reference_metric in enumerate(reference_metrics):
        run_metrics = tuple(metrics[run_index] for metrics in metrics_by_fit)
        output.append(
            RunStateStability(
                subject=reference_metric.subject,
                session=reference_metric.session,
                acquisition_id=reference_metric.acquisition_id,
                n_samples=reference_metric.n_samples,
                n_sequences=reference_metric.n_sequences,
                occupancy_by_fit=np.stack([metric.occupancy for metric in run_metrics]),
                mean_dwell_samples_by_fit=np.stack(
                    [metric.mean_dwell_samples for metric in run_metrics]
                ),
                mean_dwell_seconds_by_fit=np.stack(
                    [metric.mean_dwell_seconds for metric in run_metrics]
                ),
                switch_rate_by_fit=np.asarray(
                    [metric.switch_rate for metric in run_metrics],
                    dtype=float,
                ),
                transition_probabilities_by_fit=np.stack(
                    [metric.transition_probabilities for metric in run_metrics]
                ),
            )
        )
    return tuple(output)


__all__ = ["RunStateStability", "summarize_state_stability"]
