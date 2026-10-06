"""Loading fixture labels and evaluating generated review results."""

import json
from pathlib import Path
from typing import Any

from evaluation.metrics import EvaluationMetrics, calculate_metrics


def evaluate_fixture_results(
    fixture_root: Path, actual_findings: list[dict[str, Any]], retrieved_paths: set[str]
) -> EvaluationMetrics:
    """Load labeled expectations for a fixture and score its findings and retrieval."""
    with (fixture_root / "expected.json").open(encoding="utf-8") as file:
        expected = json.load(file)
    return calculate_metrics(
        expected.get("findings", []),
        actual_findings,
        retrieved_paths=retrieved_paths,
        retrieval_targets=expected.get("retrieval_targets", []),
    )
