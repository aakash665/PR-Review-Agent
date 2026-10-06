"""Matching and quality metrics for labeled findings and retrieved context."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationMetrics:
    """Precision, recall, location, category, and retrieval-quality measurements."""

    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    line_accuracy: float
    retrieval_hit_rate: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def as_dict(self) -> dict[str, float | int]:
        """Return metric values as a plain dictionary for reports and serialization."""
        return asdict(self)


def calculate_metrics(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    retrieved_paths: set[str] | None = None,
    retrieval_targets: list[str] | None = None,
) -> EvaluationMetrics:
    """Match expected findings to actual findings and compute aggregate quality metrics."""
    unmatched = set(range(len(expected)))
    matched: list[tuple[int, int]] = []
    for actual_index, finding in enumerate(actual):
        candidates = [
            expected_index
            for expected_index in unmatched
            if _matches(finding, expected[expected_index])
        ]
        if candidates:
            expected_index = min(
                candidates,
                key=lambda index: abs(
                    int(finding.get("line_start", finding.get("line", 0)))
                    - int(expected[index].get("line", expected[index].get("line_start", 0)))
                ),
            )
            unmatched.remove(expected_index)
            matched.append((expected_index, actual_index))
    true_positives = len(matched)
    false_positives = len(actual) - true_positives
    false_negatives = len(expected) - true_positives
    precision = true_positives / max(len(actual), 1)
    recall = true_positives / max(len(expected), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    line_accuracy = sum(
        int(actual[actual_index].get("line_start", actual[actual_index].get("line", 0)))
        == int(expected[expected_index].get("line", expected[expected_index].get("line_start", 0)))
        for expected_index, actual_index in matched
    ) / max(true_positives, 1)
    targets = retrieval_targets or []
    hits = sum(1 for path in targets if path in (retrieved_paths or set()))
    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=false_positives / max(len(actual), 1),
        line_accuracy=line_accuracy,
        retrieval_hit_rate=hits / max(len(targets), 1),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def _matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Determine whether an expected finding matches a candidate by location and content."""
    if actual.get("file_path") != expected.get("file_path"):
        return False
    actual_line = int(actual.get("line_start", actual.get("line", 0)))
    expected_line = int(expected.get("line", expected.get("line_start", 0)))
    if abs(actual_line - expected_line) > 2:
        return False
    expected_category = str(expected.get("category", "")).lower()
    actual_category = str(actual.get("category", "")).lower()
    related = {
        "security": {"security", "bug"},
        "bug": {"bug", "security"},
        "performance": {"performance"},
        "maintainability": {"maintainability"},
        "testing": {"testing"},
    }
    return not expected_category or actual_category in related.get(
        expected_category, {expected_category}
    )
