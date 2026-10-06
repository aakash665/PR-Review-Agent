"""Validated data models shared by analysis, review, and persistence layers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Category = Literal["bug", "security", "performance", "maintainability", "testing"]
Severity = Literal["critical", "high", "medium", "low"]


class ReviewFinding(BaseModel):
    """A review issue with severity, evidence, location, and confidence."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    title: str = Field(min_length=4, max_length=160)
    explanation: str = Field(min_length=10, max_length=4000)
    evidence: list[str] = Field(min_length=1, max_length=10)
    file_path: str = Field(min_length=1, max_length=1000)
    line_start: int = Field(gt=0)
    line_end: int = Field(gt=0)
    suggestion: str | None = Field(default=None, max_length=4000)
    rationale: str = Field(min_length=5, max_length=2000)
    rule_id: str | None = None
    verified: bool = False

    @field_validator("line_end")
    @classmethod
    def valid_line_range(cls, line_end: int, info: object) -> int:
        """Reject a finding whose ending line precedes its starting line."""
        line_start = getattr(info, "data", {}).get("line_start")
        if line_start is not None and line_end < line_start:
            raise ValueError("line_end must be greater than or equal to line_start")
        return line_end


class ReviewResult(BaseModel):
    """A set of findings and an optional generated pull-request summary."""

    findings: list[ReviewFinding] = Field(default_factory=list)
    summary: str = Field(default="", max_length=8000)
    positive_changes: list[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(default=0.0, ge=0, le=1)
    tokens_used: int = Field(default=0, ge=0)
    retrieved_context: str = Field(default="", exclude=True)
    retrieval_latency: float = Field(default=0, ge=0)
    llm_latency: float = Field(default=0, ge=0)
    retrieved_chunks: int = Field(default=0, ge=0)


class VerificationDecision(BaseModel):
    """The verifier outcome and rationale for one candidate finding."""

    accepted: bool
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=2000)
    corrected_line_start: int | None = Field(default=None, gt=0)
    corrected_line_end: int | None = Field(default=None, gt=0)


class StaticFinding(BaseModel):
    """A normalized result emitted by a deterministic analysis tool."""

    tool: str
    rule_id: str
    severity: str
    file_path: str
    line: int = Field(gt=0)
    message: str


class ChangedLine(BaseModel):
    """A changed source line paired with its enclosing symbol when available."""

    line: int = Field(gt=0)
    content: str
    change_type: Literal["added", "removed"]


class ChangedFile(BaseModel):
    """Validated diff metadata and commentable added lines for one file."""

    file_path: str
    status: str
    additions: int = 0
    deletions: int = 0
    language: str | None = None
    changed_lines: list[ChangedLine] = Field(default_factory=list)
    patch: str | None = None


class PullRequestMetadata(BaseModel):
    """Normalized GitHub pull-request identity, revisions, title, and description."""

    repository: str
    repository_id: int
    owner: str
    name: str
    number: int
    title: str
    body: str | None = None
    base_branch: str
    head_branch: str
    head_sha: str
    author: str
    installation_id: int | None = None
    files: list[ChangedFile] = Field(default_factory=list)
