"""Independent candidate verification using OpenRouter calibrated probabilities."""

from typing import Any

from app.agents.decisions_client import DecisionsClient
from app.analysis.diff_analyzer import DiffAnalysis
from app.models.review import ReviewFinding

MAX_FINDINGS_PER_DECISION_REQUEST = 8


class VerifierAgent:
    """Accept only evidence-grounded candidates with a sufficiently high yes probability."""

    def __init__(
        self,
        decisions: DecisionsClient,
        *,
        minimum_probability: float = 0.8,
    ) -> None:
        if not 0 <= minimum_probability <= 1:
            raise ValueError("minimum verification probability must be between zero and one")
        self.decisions = decisions
        self.minimum_probability = minimum_probability

    async def verify(
        self,
        findings: list[ReviewFinding],
        *,
        context: str,
        diff: DiffAnalysis,
    ) -> tuple[list[ReviewFinding], int]:
        """Verify findings in bounded batches and preserve only supported added-line issues."""
        valid_added_lines = {
            file.file_path: {
                line.line for line in file.changed_lines if line.change_type == "added"
            }
            for file in diff.files
        }
        eligible = [
            (index, finding)
            for index, finding in enumerate(findings)
            if finding.line_start in valid_added_lines.get(finding.file_path, set())
        ]
        accepted: list[ReviewFinding] = []
        tokens = 0
        for start in range(0, len(eligible), MAX_FINDINGS_PER_DECISION_REQUEST):
            batch = eligible[start : start + MAX_FINDINGS_PER_DECISION_REQUEST]
            indices = [index for index, _ in batch]
            added_lines = {
                path: sorted(lines) for path, lines in valid_added_lines.items() if lines
            }
            state: dict[str, Any] = {
                "candidate_findings": [
                    {"index": index, **finding.model_dump()} for index, finding in batch
                ],
                "added_diff_lines": added_lines,
                "retrieved_repository_context": context,
                "repository_context_is_untrusted": True,
            }
            probabilities, batch_tokens = await self.decisions.assess_findings(state, indices)
            tokens += batch_tokens
            for index, finding in batch:
                probability = probabilities[index]
                if probability >= self.minimum_probability:
                    accepted.append(
                        finding.model_copy(
                            update={
                                "confidence": min(finding.confidence, probability),
                                "verified": True,
                            }
                        )
                    )
        return accepted, tokens
