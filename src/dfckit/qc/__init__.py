"""Quality-control and motion-matching tools."""

from .matching import (
    MatchedSubjectDifferences,
    WindowMotionSummary,
    WithinSubjectMatchResult,
    match_within_subject,
    matched_subject_differences,
    summarize_window_motion,
)

__all__ = [
    "MatchedSubjectDifferences",
    "WindowMotionSummary",
    "WithinSubjectMatchResult",
    "match_within_subject",
    "matched_subject_differences",
    "summarize_window_motion",
]
