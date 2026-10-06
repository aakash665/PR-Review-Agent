"""Confidence-threshold filtering for generated findings and summaries."""

from dataclasses import dataclass

from app.models.review import ReviewFinding


@dataclass(frozen=True)
class ConfidenceResult:
    """Partition findings into inline, summary-only, and discarded confidence bands."""

    inline: list[ReviewFinding]
    summary_only: list[ReviewFinding]
    discarded: list[ReviewFinding]


def filter_by_confidence(
    findings: list[ReviewFinding],
    *,
    inline_threshold: float = 0.80,
    summary_threshold: float = 0.60,
) -> ConfidenceResult:
    """Apply inclusive confidence thresholds, rejecting invalid threshold ordering."""
    if not 0 <= summary_threshold <= inline_threshold <= 1:
        raise ValueError("Confidence thresholds must satisfy 0 <= summary <= inline <= 1")
    inline: list[ReviewFinding] = []
    summary_only: list[ReviewFinding] = []
    discarded: list[ReviewFinding] = []
    for finding in findings:
        if finding.confidence >= inline_threshold:
            inline.append(finding)
        elif finding.confidence >= summary_threshold:
            summary_only.append(finding)
        else:
            discarded.append(finding)
    return ConfidenceResult(inline, summary_only, discarded)
