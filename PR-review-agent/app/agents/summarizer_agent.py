"""Structured pull-request summary generation."""

import json

from pydantic import BaseModel, Field

from app.agents.client import LLMClient
from app.agents.prompts import SUMMARY_SYSTEM_PROMPT
from app.models.review import ReviewFinding


class SummaryResponse(BaseModel):
    """Validated summary text returned for a pull request."""

    summary: str = Field(min_length=1, max_length=4000)
    positive_changes: list[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(ge=0, le=1)


class SummarizerAgent:
    """Produce a concise structured summary of the pull request and findings."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def summarize(
        self,
        *,
        title: str,
        changed_files: int,
        additions: int,
        deletions: int,
        findings: list[ReviewFinding],
    ) -> tuple[SummaryResponse, int]:
        """Generate a pull-request summary and return its token usage."""
        return await self.llm.complete(
            system=SUMMARY_SYSTEM_PROMPT,
            user=json.dumps(
                {
                    "title": title,
                    "files_analyzed": changed_files,
                    "lines_added": additions,
                    "lines_deleted": deletions,
                    "verified_findings": [item.model_dump() for item in findings],
                    "task": "Provide a concise summary and evidence-grounded positive changes.",
                }
            ),
            response_model=SummaryResponse,
        )
