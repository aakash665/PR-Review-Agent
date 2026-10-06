"""Independent validation of candidate review findings."""

import json

from pydantic import BaseModel, Field

from app.agents.client import LLMClient
from app.agents.prompts import VERIFICATION_SYSTEM_PROMPT, verification_prompt
from app.analysis.diff_analyzer import DiffAnalysis
from app.models.review import ReviewFinding


class VerificationItem(BaseModel):
    """Verifier decision for one candidate finding, identified by its input index."""

    index: int = Field(ge=0)
    accepted: bool
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=2000)


class VerificationBatch(BaseModel):
    """Structured batch of independent verifier decisions."""

    decisions: list[VerificationItem] = Field(default_factory=list)


class VerifierAgent:
    """Ask an independent model pass to confirm or reject candidate findings."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def verify(
        self,
        findings: list[ReviewFinding],
        *,
        context: str,
        diff: DiffAnalysis,
    ) -> tuple[list[ReviewFinding], int]:
        """Verify findings against the diff and context, preserving only accepted evidence."""
        if not findings:
            return [], 0
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
        if not eligible:
            return [], 0
        output, tokens = await self.llm.complete(
            system=VERIFICATION_SYSTEM_PROMPT,
            user=(
                f"{verification_prompt()}\n\nCandidates:\n"
                + json.dumps(
                    [{"index": index, **finding.model_dump()} for index, finding in eligible],
                    ensure_ascii=True,
                )
                + f"\n\nUNTRUSTED REPOSITORY CONTEXT:\n{context}"
            ),
            response_model=VerificationBatch,
        )
        accepted: list[ReviewFinding] = []
        seen: set[int] = set()
        eligible_indices = {index for index, _ in eligible}
        for decision in output.decisions:
            if (
                decision.index not in eligible_indices
                or decision.index in seen
                or not decision.accepted
            ):
                continue
            seen.add(decision.index)
            finding = findings[decision.index]
            accepted.append(
                finding.model_copy(
                    update={
                        "confidence": min(finding.confidence, decision.confidence),
                        "verified": True,
                    }
                )
            )
        return accepted, tokens
