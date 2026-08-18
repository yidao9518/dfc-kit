"""Run-level occupancy, dwell, switching, and transition summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import StateAssignments, _readonly


@dataclass(frozen=True)
class RunStateMetrics:
    subject: str
    session: str | None
    n_samples: int
    n_sequences: int
    n_possible_transitions: int
    n_switches: int
    occupancy: NDArray[np.float64]
    mean_dwell_samples: NDArray[np.float64]
    mean_dwell_seconds: NDArray[np.float64]
    transition_counts: NDArray[np.int64]
    transition_probabilities: NDArray[np.float64]
    switch_rate: float
    acquisition_id: str | None = None


def _runs(labels: NDArray[np.int64]) -> tuple[tuple[int, int], ...]:
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], len(labels)]
    return tuple(
        (int(labels[start]), int(end - start))
        for start, end in zip(starts, ends, strict=True)
    )


def summarize_state_assignments(assignments: StateAssignments) -> tuple[RunStateMetrics, ...]:
    """Summarize labels without transitions or dwell runs crossing a sequence gap."""
    grouped: dict[tuple[str, str | None, str | None], list] = {}
    for sequence in assignments.sequences:
        grouped.setdefault(
            (sequence.subject, sequence.session, sequence.acquisition_id), []
        ).append(sequence)

    output: list[RunStateMetrics] = []
    for (subject, session, acquisition_id), sequences in grouped.items():
        counts = np.zeros(assignments.n_states, dtype=np.int64)
        transitions = np.zeros((assignments.n_states, assignments.n_states), dtype=np.int64)
        dwell: list[list[int]] = [[] for _ in range(assignments.n_states)]
        possible = 0
        for sequence in sequences:
            labels = np.asarray(sequence.labels, dtype=np.int64)
            counts += np.bincount(labels, minlength=assignments.n_states)
            for state, length in _runs(labels):
                dwell[state].append(length)
            if len(labels) > 1:
                np.add.at(transitions, (labels[:-1], labels[1:]), 1)
                possible += len(labels) - 1
        n_samples = int(counts.sum())
        occupancy = counts / n_samples
        mean_dwell = np.asarray(
            [np.mean(lengths) if lengths else np.nan for lengths in dwell], dtype=float
        )
        mean_dwell_seconds = (
            np.full(assignments.n_states, np.nan)
            if assignments.sample_interval_seconds is None
            else mean_dwell * assignments.sample_interval_seconds
        )
        row_totals = transitions.sum(axis=1)
        probabilities = np.full(transitions.shape, np.nan, dtype=float)
        valid = row_totals > 0
        probabilities[valid] = transitions[valid] / row_totals[valid, None]
        switches = int(transitions.sum() - np.trace(transitions))
        output.append(
            RunStateMetrics(
                subject=subject,
                session=session,
                acquisition_id=acquisition_id,
                n_samples=n_samples,
                n_sequences=len(sequences),
                n_possible_transitions=possible,
                n_switches=switches,
                occupancy=_readonly(occupancy),
                mean_dwell_samples=_readonly(mean_dwell),
                mean_dwell_seconds=_readonly(mean_dwell_seconds),
                transition_counts=_readonly(transitions),
                transition_probabilities=_readonly(probabilities),
                switch_rate=float(switches / possible) if possible else float("nan"),
            )
        )
    return tuple(output)
