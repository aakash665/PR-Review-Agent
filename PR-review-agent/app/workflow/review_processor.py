"""End-to-end processing of a durable pull-request review job."""

import json
import logging
import sqlite3
import time
from typing import Any

from app.agents.client import LLMClient
from app.agents.review_agent import ReviewAgent
from app.agents.summarizer_agent import SummarizerAgent
from app.agents.verifier_agent import VerifierAgent
from app.analysis.diff_analyzer import DiffAnalysis, DiffAnalyzer
from app.analysis.static_analysis import StaticAnalyzer
from app.config import Settings
from app.database.repositories import JobRepository
from app.database.session import Database
from app.github.client import GitHubClient
from app.ingestion.chunker import CodeChunker
from app.ingestion.indexer import RepositoryIndexer
from app.ingestion.repository_loader import GitHubRepositoryLoader
from app.models.review import PullRequestMetadata, ReviewFinding
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.embeddings import OpenAIEmbeddingProvider
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import QdrantVectorStore
from app.review.confidence import filter_by_confidence
from app.review.deduplicator import deduplicate_findings
from app.review.github_formatter import format_review_summary
from app.review.github_publisher import GitHubReviewPublisher

logger = logging.getLogger(__name__)


class ReviewProcessor:
    """Coordinate fetch, index, analysis, verification, and publication for a job."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        *,
        github: GitHubClient | None = None,
        vectors: QdrantVectorStore | None = None,
        static_analyzer: StaticAnalyzer | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.jobs = JobRepository(database)
        self.github = github or GitHubClient(settings)
        self.vectors = vectors or QdrantVectorStore(
            settings.qdrant_url,
            settings.qdrant_collection,
            settings.embedding_dimensions,
        )
        self.static_analyzer = static_analyzer or StaticAnalyzer()
        self.diff_analyzer = DiffAnalyzer()

    async def process(self, job: sqlite3.Row | dict[str, Any]) -> None:
        """Run a review job through analysis and persist its terminal outcome and metrics."""
        started = time.monotonic()
        job_id = int(job["id"])
        owner, repo = str(job["owner"]), str(job["name"])
        installation_id = job["installation_id"]
        if installation_id is None:
            raise ValueError("Review job does not have a GitHub App installation ID")
        installation_id = int(installation_id)
        llm = LLMClient(self.settings)
        embeddings = OpenAIEmbeddingProvider(self.settings, self.database)
        retriever = HybridRetriever(embeddings, self.vectors, top_k=self.settings.top_k)
        loader = GitHubRepositoryLoader(self.github)
        indexer = RepositoryIndexer(CodeChunker(), embeddings, self.vectors)
        pr_number = int(job["github_pr_number"])
        pr = await self.github.get_pull_request(owner, repo, pr_number, installation_id)
        base_repo_id = int(pr.get("base", {}).get("repo", {}).get("id", 0))
        if base_repo_id != int(job["github_repo_id"]):
            raise ValueError("GitHub pull request belongs to a different repository")
        if pr.get("state") != "open":
            self.jobs.cancel(job_id, "Pull request is no longer open")
            return
        head_sha = str(pr["head"]["sha"])
        if head_sha != str(job["commit_sha"]):
            logger.info(
                "Skipping stale review job review_id=%s repository=%s pr=%s",
                job_id,
                f"{owner}/{repo}",
                pr_number,
            )
            self.jobs.cancel(job_id, "Pull request head changed before review started")
            return
        pr_files = await self.github.get_pull_request_files(owner, repo, pr_number, installation_id)
        diff_text = await self.github.get_pull_request_diff(owner, repo, pr_number, installation_id)
        diff = self.diff_analyzer.analyze(diff_text, pr_files)
        metadata = PullRequestMetadata(
            repository=f"{owner}/{repo}",
            repository_id=int(job["github_repo_id"]),
            owner=owner,
            name=repo,
            number=pr_number,
            title=str(pr.get("title") or ""),
            body=pr.get("body"),
            base_branch=str(pr["base"]["ref"]),
            head_branch=str(pr["head"]["ref"]),
            head_sha=head_sha,
            author=str(pr.get("user", {}).get("login", "unknown")),
            installation_id=installation_id,
            files=diff.files,
        )
        embedding_started = embeddings.total_latency_seconds
        repository_files, _indexed_chunk_count = await self._index_snapshot(
            pr_files=pr_files,
            metadata=metadata,
            loader=loader,
            indexer=indexer,
        )
        changed_files = {
            path: content
            for path, content in repository_files.items()
            if path in {file.file_path for file in diff.files}
        }
        static_started = time.monotonic()
        static_findings = self.static_analyzer.run(changed_files)
        static_latency = time.monotonic() - static_started
        agent = ReviewAgent(
            llm,
            retriever,
            ContextBuilder(self.settings.max_context_tokens),
        )
        review = await agent.review(
            metadata=metadata,
            diff=diff,
            repository_files=repository_files,
            static_findings=static_findings,
        )
        verifier = VerifierAgent(llm)
        verify_started = time.monotonic()
        accepted, verifier_tokens = await verifier.verify(
            review.findings, context=review.retrieved_context, diff=diff
        )
        verification_latency = time.monotonic() - verify_started
        findings = deduplicate_findings(accepted)
        confidence = filter_by_confidence(
            findings,
            inline_threshold=self.settings.confidence_threshold,
            summary_threshold=self.settings.summary_confidence_threshold,
        )
        added = sum(file.additions for file in diff.files)
        deleted = sum(file.deletions for file in diff.files)
        summary_findings = list(confidence.summary_only)
        publishable: list[ReviewFinding] = []
        unmapped: list[ReviewFinding] = []
        for finding in confidence.inline:
            if _is_added_line(diff, finding):
                publishable.append(finding)
            else:
                unmapped.append(finding)
                summary_findings.append(finding)
        summary_agent = SummarizerAgent(llm)
        summary_started = time.monotonic()
        summary_result, summary_tokens = await summary_agent.summarize(
            title=metadata.title,
            changed_files=len(diff.files),
            additions=added,
            deletions=deleted,
            findings=[*publishable, *confidence.summary_only, *unmapped],
        )
        summary_latency = time.monotonic() - summary_started
        summary = format_review_summary(
            files_analyzed=len(diff.files),
            additions=added,
            deletions=deleted,
            findings=publishable,
            summary_only=summary_findings,
            discarded_count=len(confidence.discarded),
            rejected_count=len(review.findings) - len(accepted),
            llm_summary=summary_result.summary,
            positive_changes=summary_result.positive_changes or review.positive_changes,
            review_confidence=summary_result.confidence,
            unmapped_count=len(unmapped) + max(0, len(publishable) - 50),
        )
        total_tokens = (
            review.tokens_used + verifier_tokens + summary_tokens + embeddings.total_tokens
        )
        cost = (
            total_tokens * self.settings.cost_per_million_tokens / 1_000_000
            if self.settings.cost_per_million_tokens is not None
            else None
        )
        metrics: dict[str, float | int | None] = {
            "retrieval_latency": review.retrieval_latency,
            "llm_latency": review.llm_latency + verification_latency + summary_latency,
            "embedding_latency": embeddings.total_latency_seconds - embedding_started,
            "static_analysis_latency": static_latency,
            "number_of_chunks": review.retrieved_chunks,
            "number_of_findings": len(findings),
            "number_of_rejected_findings": len(review.findings) - len(accepted),
            "tokens_used": total_tokens,
            "estimated_cost_usd": cost,
        }
        current_pr = await self.github.get_pull_request(owner, repo, pr_number, installation_id)
        if current_pr.get("state") != "open" or current_pr.get("head", {}).get("sha") != head_sha:
            self.jobs.cancel(job_id, "Pull request closed or updated before review publication")
            return
        await GitHubReviewPublisher(self.github).publish(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            head_sha=head_sha,
            installation_id=installation_id,
            summary=summary,
            findings=publishable,
            diff=diff,
        )
        self.jobs.complete(job_id, findings, metrics)
        logger.info(
            json.dumps(
                {
                    "event": "review_completed",
                    "review_id": job_id,
                    "repository": f"{owner}/{repo}",
                    "pr_number": pr_number,
                    "commit_sha": head_sha,
                    "retrieval_latency": metrics["retrieval_latency"],
                    "llm_latency": metrics["llm_latency"],
                    "embedding_latency": metrics["embedding_latency"],
                    "static_analysis_latency": metrics["static_analysis_latency"],
                    "number_of_chunks": review.retrieved_chunks,
                    "number_of_findings": len(findings),
                    "number_of_rejected_findings": metrics["number_of_rejected_findings"],
                    "tokens_used": total_tokens,
                    "estimated_cost_usd": cost,
                    "duration_seconds": round(time.monotonic() - started, 3),
                }
            )
        )

    async def _index_snapshot(
        self,
        *,
        pr_files: list[dict[str, Any]],
        metadata: PullRequestMetadata,
        loader: GitHubRepositoryLoader,
        indexer: RepositoryIndexer,
    ) -> tuple[dict[str, bytes], int]:
        """Load and index a commit snapshot while reusing prior unchanged vectors."""
        repo_key = str(metadata.repository_id)
        sha = metadata.head_sha
        state = self.jobs.repository_state(metadata.repository_id)
        previous = str(state["last_indexed_sha"]) if state and state["last_indexed_sha"] else None
        pr_paths = {
            str(item["filename"])
            for item in pr_files
            if item.get("filename") and item.get("status") != "removed"
        }
        if previous == sha:
            contents = await loader.load(
                metadata.owner, metadata.name, sha, metadata.installation_id or 0, pr_paths
            )
            return contents, 0
        if previous:
            comparison = await self.github.get_commit_comparison(
                metadata.owner, metadata.name, previous, sha, metadata.installation_id or 0
            )
            compare_files = comparison.get("files") or []
            if comparison.get("status") in {"ahead", "identical"} and len(compare_files) < 300:
                changed = {str(item["filename"]) for item in compare_files if item.get("filename")}
                deleted = {
                    str(item["filename"])
                    for item in compare_files
                    if item.get("filename") and item.get("status") == "removed"
                }
                paths = changed - deleted
                contents = await loader.load(
                    metadata.owner, metadata.name, sha, metadata.installation_id or 0, paths
                )
                count = await indexer.index_files(
                    repository=repo_key,
                    commit_sha=sha,
                    files=contents,
                    previous_commit_sha=previous,
                    deleted_files=deleted,
                )
                expired = self.jobs.record_indexed_sha(
                    metadata.repository_id, sha, self.settings.index_snapshot_retention
                )
                for expired_sha in expired:
                    self.vectors.delete_commit(repo_key, expired_sha)
                if not paths:
                    contents = await loader.load(
                        metadata.owner, metadata.name, sha, metadata.installation_id or 0, pr_paths
                    )
                return contents, count
        contents = await loader.load(
            metadata.owner, metadata.name, sha, metadata.installation_id or 0
        )
        count = await indexer.index_files(
            repository=repo_key, commit_sha=sha, files=contents, full_reindex=True
        )
        expired = self.jobs.record_indexed_sha(
            metadata.repository_id, sha, self.settings.index_snapshot_retention
        )
        for expired_sha in expired:
            self.vectors.delete_commit(repo_key, expired_sha)
        return contents, count


def _is_added_line(diff: DiffAnalysis, finding: ReviewFinding) -> bool:
    """Check whether a source line is an added, commentable diff line."""
    return any(
        file.file_path == finding.file_path
        and any(
            line.line == finding.line_start and line.change_type == "added"
            for line in file.changed_lines
        )
        for file in diff.files
    )
