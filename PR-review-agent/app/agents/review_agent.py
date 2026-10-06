"""Evidence-grounded review generation and validation against changed code."""

import contextvars
import json
import logging
import time
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from app.agents.client import LLMClient
from app.agents.prompts import REVIEW_SYSTEM_PROMPT
from app.agents.tools import ReviewToolbox
from app.analysis.dependency_analyzer import imported_modules
from app.analysis.diff_analyzer import DiffAnalysis, identify_changed_symbols
from app.models.review import PullRequestMetadata, ReviewFinding, ReviewResult, StaticFinding
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)
current_toolbox: contextvars.ContextVar[ReviewToolbox | None] = contextvars.ContextVar(
    "review_toolbox", default=None
)


class ReviewResponse(BaseModel):
    """Structured model response containing candidate pull-request findings."""

    findings: list[ReviewFinding] = Field(default_factory=list)
    summary: str = Field(default="", max_length=8000)
    positive_changes: list[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(default=0.0, ge=0, le=1)


class ReviewAgent:
    """Generate findings grounded in changed lines and retrieved repository evidence."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: HybridRetriever,
        context_builder: ContextBuilder,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.context_builder = context_builder

    async def review(
        self,
        *,
        metadata: PullRequestMetadata,
        diff: DiffAnalysis,
        repository_files: dict[str, bytes],
        static_findings: list[StaticFinding],
    ) -> ReviewResult:
        """Analyze the pull-request diff and return evidence-grounded findings and usage."""
        symbols_by_file = identify_changed_symbols(diff, repository_files)
        dependencies_by_symbol: dict[str, list[str]] = {}
        for file_path, symbols in symbols_by_file.items():
            source = repository_files[file_path].decode("utf-8")
            language = _language(file_path)
            dependencies = sorted(imported_modules(source, language))
            for symbol in symbols:
                dependencies_by_symbol[symbol] = dependencies
        retrieval_started = time.monotonic()
        queries = [
            self.retriever.retrieve(
                repository=str(metadata.repository_id),
                commit_sha=metadata.head_sha,
                query=(
                    f"{file.file_path} "
                    f"{' '.join(symbols_by_file.get(file.file_path, []))} "
                    f"{' '.join(line.content for line in file.changed_lines[:8])}"
                ),
                symbol=(
                    symbols_by_file[file.file_path][0]
                    if symbols_by_file.get(file.file_path)
                    else None
                ),
                file_path=file.file_path,
                top_k=5,
            )
            for file in diff.files
            if file.changed_lines
        ]
        retrieved_per_file = await _gather(queries)
        retrieval_latency = time.monotonic() - retrieval_started
        retrieved = [context for group in retrieved_per_file for context in group]
        context = self.context_builder.build(
            diff=diff, retrieved=retrieved, static_findings=static_findings
        )
        source_diffs = [
            {
                "path": item.file_path,
                "status": item.status,
                "changed_lines": [line.model_dump() for line in item.changed_lines],
                "symbols": symbols_by_file.get(item.file_path, []),
                "dependencies": sorted(
                    {
                        dependency
                        for name in symbols_by_file.get(item.file_path, [])
                        for dependency in dependencies_by_symbol.get(name, [])
                    }
                ),
            }
            for item in diff.files
            if item.changed_lines
        ]
        changed_symbols_and_imports = [
            {
                "path": path,
                "symbols": symbols,
                "dependencies": sorted(
                    {
                        dependency
                        for symbol in symbols
                        for dependency in dependencies_by_symbol.get(symbol, [])
                    }
                ),
            }
            for path, symbols in symbols_by_file.items()
        ]
        tool_box = ReviewToolbox(
            repository=str(metadata.repository_id),
            commit_sha=metadata.head_sha,
            retriever=self.retriever,
            files=repository_files,
            diff=json.dumps(source_diffs)[:30000],
            static_findings=static_findings,
            dependencies=dependencies_by_symbol,
        )
        valid_added_lines = {
            file.file_path: {
                item.line for item in file.changed_lines if item.change_type == "added"
            }
            for file in diff.files
        }
        line_ranges = {
            path: _compress_lines(lines) for path, lines in valid_added_lines.items() if lines
        }
        token = current_toolbox.set(tool_box)
        llm_started = time.monotonic()
        try:
            result, tokens = await self.llm.complete(
                system=REVIEW_SYSTEM_PROMPT,
                user=(
                    "Review metadata and changed code follow. Only comment on added lines, cite "
                    "retrieved evidence, and return a ReviewResponse JSON object.\n"
                    f"METADATA: {metadata.model_dump_json(exclude={'body', 'files'})}\n"
                    "CHANGED SYMBOLS AND IMPORTS: "
                    f"{ContextBuilder._safe_json(changed_symbols_and_imports)}\n"
                    "Valid added-line ranges (code is in the bounded context): "
                    f"{json.dumps(line_ranges)}\n"
                    f"RETRIEVED CONTEXT: {context}"
                ),
                response_model=ReviewResponse,
                tools=tool_box.schemas(),
            )
        finally:
            current_toolbox.reset(token)
        llm_latency = time.monotonic() - llm_started
        findings = [
            finding
            for finding in result.findings
            if finding.file_path in valid_added_lines
            and finding.line_start in valid_added_lines[finding.file_path]
            and finding.line_end >= finding.line_start
            and finding.line_end <= max(valid_added_lines[finding.file_path], default=0)
        ]
        dropped = len(result.findings) - len(findings)
        if dropped:
            logger.info("Dropped %d review findings with invalid changed-line locations", dropped)
        return ReviewResult(
            findings=findings,
            summary=result.summary,
            positive_changes=result.positive_changes,
            confidence=result.confidence,
            tokens_used=tokens,
            retrieved_context=context,
            retrieval_latency=retrieval_latency,
            llm_latency=llm_latency,
            retrieved_chunks=len(retrieved),
        )


async def _gather(awaitables: list[Any]) -> list[Any]:
    """Collect bounded supporting context for changed symbols and files."""
    import asyncio

    return list(await asyncio.gather(*awaitables)) if awaitables else []


def _compress_lines(lines: set[int]) -> list[str]:
    """Trim source context while preserving the lines needed to explain a finding."""
    ordered = sorted(lines)
    if not ordered:
        return []
    ranges: list[str] = []
    start = previous = ordered[0]
    for line in ordered[1:]:
        if line == previous + 1:
            previous = line
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = line
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ranges


def _language(file_path: str) -> str:
    """Map a source path to a GitHub inline-comment language label."""
    return {
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
    }.get(PurePosixPath(file_path).suffix.lower(), "")
