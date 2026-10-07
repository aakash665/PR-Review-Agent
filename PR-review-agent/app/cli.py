"""Command-line entry points for repository indexing, reviews, fixtures, and evaluation."""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from app.agents.client import LLMClient
from app.agents.decisions_client import DecisionsClient
from app.agents.review_agent import ReviewAgent
from app.agents.verifier_agent import VerifierAgent
from app.analysis.diff_analyzer import DiffAnalyzer
from app.analysis.static_analysis import StaticAnalyzer
from app.config import Settings, get_settings
from app.database.repositories import JobRepository
from app.database.session import Database
from app.github.client import GitHubClient
from app.ingestion.chunker import CodeChunker
from app.ingestion.file_discovery import discover_files
from app.ingestion.indexer import RepositoryIndexer
from app.ingestion.repository_loader import GitHubRepositoryLoader
from app.models.review import PullRequestMetadata, ReviewFinding
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.embeddings import LocalFeatureEmbeddingProvider, OpenRouterEmbeddingProvider
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import QdrantVectorStore
from app.review.confidence import filter_by_confidence
from app.review.deduplicator import deduplicate_findings
from app.workflow.review_processor import ReviewProcessor
from evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class FixtureRun:
    """Result of processing a local fixture, including findings and evaluation data."""

    findings: list[ReviewFinding]
    retrieved_paths: set[str]
    static_findings: list[dict[str, Any]]
    retrieved_count: int
    indexed_chunks: int
    ai_enabled: bool


async def run_fixture(root: Path, settings: Settings, *, require_llm: bool = False) -> FixtureRun:
    """Index and review a local fixture without making GitHub API requests."""
    root = root.resolve()
    repository_id = 1
    repository_key = str(repository_id)
    if not root.is_dir():
        raise ValueError(f"Fixture directory does not exist: {root}")
    diff_path = root / "sample.diff"
    if not diff_path.is_file():
        raise ValueError(f"Fixture is missing sample.diff: {root}")
    repository_files = {
        path.relative_to(root).as_posix(): path.read_bytes() for path in discover_files(root)
    }
    digest = hashlib.sha256(
        b"".join(path.encode() + content for path, content in sorted(repository_files.items()))
    ).hexdigest()
    diff = DiffAnalyzer().analyze(diff_path.read_text(encoding="utf-8"))
    static_findings = StaticAnalyzer().run(
        {
            item.file_path: repository_files[item.file_path]
            for item in diff.files
            if item.file_path in repository_files
        }
    )
    client = QdrantClient(location=":memory:")
    try:
        vectors = QdrantVectorStore(
            url=":memory:",
            collection=f"fixture_{digest[:20]}",
            dimensions=384,
            client=client,
            create_payload_indexes=False,
        )
        embeddings = LocalFeatureEmbeddingProvider(dimensions=384)
        indexer = RepositoryIndexer(CodeChunker(), embeddings, vectors)
        indexed_chunks = await indexer.index_files(
            repository=repository_key,
            commit_sha=digest,
            files=repository_files,
            full_reindex=True,
        )
        retriever = HybridRetriever(embeddings, vectors, top_k=settings.top_k)
        queries = await asyncio.gather(
            *(
                retriever.retrieve(
                    repository=repository_key,
                    commit_sha=digest,
                    query=f"{file.file_path} "
                    + " ".join(line.content for line in file.changed_lines[:6]),
                    file_path=file.file_path,
                    top_k=settings.top_k,
                )
                for file in diff.files
                if file.changed_lines
            )
        )
        retrieved = [item for result in queries for item in result]
        metadata = PullRequestMetadata(
            repository=root.name,
            repository_id=repository_id,
            owner="fixture",
            name=root.name,
            number=1,
            title=f"Fixture review: {root.name}",
            base_branch="main",
            head_branch="fixture",
            head_sha=digest,
            author="fixture",
            files=diff.files,
        )
        if settings.openrouter_api_key is None:
            if require_llm:
                raise ValueError("OPENROUTER_API_KEY is required to run AI fixture evaluation")
            return FixtureRun(
                findings=[],
                retrieved_paths={item.file_path for item in retrieved},
                static_findings=[item.model_dump() for item in static_findings],
                retrieved_count=len(retrieved),
                indexed_chunks=indexed_chunks,
                ai_enabled=False,
            )
        llm = LLMClient(settings)
        review = await ReviewAgent(
            llm, retriever, ContextBuilder(settings.max_context_tokens)
        ).review(
            metadata=metadata,
            diff=diff,
            repository_files=repository_files,
            static_findings=static_findings,
        )
        verified, _ = await VerifierAgent(
            DecisionsClient(settings),
            minimum_probability=settings.verification_threshold,
        ).verify(
            review.findings,
            context=review.retrieved_context,
            diff=diff,
        )
        findings = deduplicate_findings(verified)
        confidence = filter_by_confidence(
            findings,
            inline_threshold=settings.confidence_threshold,
            summary_threshold=settings.summary_confidence_threshold,
        )
        return FixtureRun(
            findings=[*confidence.inline, *confidence.summary_only],
            retrieved_paths={item.file_path for item in retrieved},
            static_findings=[item.model_dump() for item in static_findings],
            retrieved_count=len(retrieved),
            indexed_chunks=indexed_chunks,
            ai_enabled=True,
        )
    finally:
        client.close()


async def _index_repository(owner: str, repo: str, commit: str | None, settings: Settings) -> None:
    """Index a remote repository revision through the configured CLI clients."""
    database = Database(settings.database_path)
    database.initialize()
    github = GitHubClient(settings)
    try:
        installation_id = await github.get_repository_installation(owner, repo)
        repository = await github.get_repository(owner, repo, installation_id)
        default_branch = str(repository["default_branch"])
        if commit is None:
            branch = await github.get_branch(owner, repo, default_branch, installation_id)
            commit = str(branch["commit"]["sha"])
        else:
            commit = await github.resolve_commit(owner, repo, commit, installation_id)
        contents = await GitHubRepositoryLoader(github).load(owner, repo, commit, installation_id)
        vectors = QdrantVectorStore(
            settings.qdrant_url, settings.qdrant_collection, settings.embedding_dimensions
        )
        indexer = RepositoryIndexer(
            CodeChunker(), OpenRouterEmbeddingProvider(settings, database), vectors
        )
        count = await indexer.index_files(
            repository=str(repository["id"]),
            commit_sha=commit,
            files=contents,
            full_reindex=True,
        )
        jobs = JobRepository(database)
        jobs.upsert_repository(
            github_repo_id=int(repository["id"]),
            owner=owner,
            name=repo,
            default_branch=default_branch,
            installation_id=installation_id,
        )
        expired = jobs.record_indexed_sha(
            int(repository["id"]), commit, settings.index_snapshot_retention
        )
        for expired_sha in expired:
            vectors.delete_commit(str(repository["id"]), expired_sha)
        print(f"Indexed {len(contents)} files into {count} chunks at {commit}.")
    finally:
        await github.close()


async def _review_pull_request(owner: str, repo: str, number: int, settings: Settings) -> None:
    """Fetch and process a pull request from the command line."""
    database = Database(settings.database_path)
    database.initialize()
    github = GitHubClient(settings)
    try:
        installation_id = await github.get_repository_installation(owner, repo)
        repository = await github.get_repository(owner, repo, installation_id)
        pull_request = await github.get_pull_request(owner, repo, number, installation_id)
        job_id, created = JobRepository(database).enqueue(
            github_repo_id=int(repository["id"]),
            owner=owner,
            name=repo,
            default_branch=str(repository["default_branch"]),
            pr_number=number,
            head_sha=str(pull_request["head"]["sha"]),
            installation_id=installation_id,
        )
        if not created:
            print(json.dumps(JobRepository(database).review_status(job_id), indent=2, default=str))
            return
        job = JobRepository(database).claim_by_id(job_id, max(1, settings.max_retries))
        if job is None:
            raise RuntimeError("Could not claim the newly enqueued review job")
        try:
            processor = ReviewProcessor(settings, database, github=github)
            await processor.process(job)
        except Exception as error:
            JobRepository(database).fail(job_id, f"{type(error).__name__}: {error}", retry=False)
            raise
        print(json.dumps(JobRepository(database).review_status(job_id), indent=2, default=str))
    finally:
        await github.close()


async def _evaluate(dataset: Path, settings: Settings) -> None:
    """Run fixture evaluations and write aggregate quality metrics."""
    if settings.openrouter_api_key is None:
        raise ValueError("OPENROUTER_API_KEY is required for AI evaluation")
    with dataset.open(encoding="utf-8") as file:
        fixtures = json.load(file)
    reports = []
    all_expected: list[dict[str, Any]] = []
    all_actual: list[dict[str, Any]] = []
    all_retrieved: set[str] = set()
    all_targets: list[str] = []
    for item in fixtures:
        root = (dataset.parent / item["fixture"]).resolve()
        result = await run_fixture(root, settings, require_llm=True)
        expected_file = root / "expected.json"
        expected_data = json.loads(expected_file.read_text(encoding="utf-8"))
        actual = [finding.model_dump() for finding in result.findings]
        metrics = calculate_metrics(
            expected_data["findings"],
            actual,
            retrieved_paths=result.retrieved_paths,
            retrieval_targets=expected_data.get("retrieval_targets", []),
        )
        reports.append({"fixture": item["fixture"], **metrics.as_dict()})
        all_expected.extend(expected_data["findings"])
        all_actual.extend(actual)
        all_retrieved.update(result.retrieved_paths)
        all_targets.extend(expected_data.get("retrieval_targets", []))
    aggregate = calculate_metrics(
        all_expected,
        all_actual,
        retrieved_paths=all_retrieved,
        retrieval_targets=all_targets,
    )
    report_payload = {"fixtures": reports, "aggregate": aggregate.as_dict()}
    report_directory = Path("evaluation/reports")
    report_directory.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report_payload, indent=2)
    (report_directory / "latest.json").write_text(serialized, encoding="utf-8")
    print(serialized)


def _parser() -> argparse.ArgumentParser:
    """Construct the argument parser and register supported subcommands."""
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="Index a GitHub repository snapshot")
    index.add_argument("repository", help="Repository in owner/name format")
    index.add_argument("--commit", help="Commit SHA or ref (defaults to the default branch head)")
    review = commands.add_parser("review", help="Review a GitHub pull request")
    review.add_argument("repository", help="Repository in owner/name format")
    review.add_argument("--pr", type=int, required=True, help="Pull request number")
    fixture = commands.add_parser("review-fixture", help="Run the local RAG fixture demo")
    fixture.add_argument("path", type=Path, help="Fixture repository directory")
    evaluate = commands.add_parser("evaluate", help="Run the labeled fixture evaluation")
    evaluate.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/default.json"),
    )
    return parser


async def _dispatch(arguments: argparse.Namespace, settings: Settings) -> None:
    """Dispatch parsed CLI arguments to the corresponding async command."""
    if arguments.command in {"index", "review"}:
        try:
            owner, repo = arguments.repository.split("/", maxsplit=1)
        except ValueError as error:
            raise ValueError("Repository must be in owner/name format") from error
        if not owner or not repo or "/" in repo:
            raise ValueError("Repository must be in owner/name format")
        if arguments.command == "index":
            await _index_repository(owner, repo, arguments.commit, settings)
        else:
            await _review_pull_request(owner, repo, arguments.pr, settings)
    elif arguments.command == "review-fixture":
        result = await run_fixture(arguments.path, settings)
        print("GitHub PR Review Agent - local fixture")
        print(f"Indexed chunks: {result.indexed_chunks}")
        print(f"Retrieved contexts: {result.retrieved_count}")
        print(f"Static findings: {len(result.static_findings)}")
        if result.ai_enabled:
            print("Verified AI findings:")
            for finding in result.findings:
                print(
                    f"- [{finding.severity.upper()}] {finding.title} "
                    f"({finding.file_path}:{finding.line_start}, {finding.confidence:.0%})"
                )
        else:
            print(
                "AI review skipped: set OPENROUTER_API_KEY to run the review and verifier agents."
            )
            print("RAG indexing, vector retrieval, and static analysis completed.")
    elif arguments.command == "evaluate":
        await _evaluate(arguments.dataset, settings)


def main() -> None:
    """Parse CLI arguments and dispatch the requested application operation."""
    logging.basicConfig(level=get_settings().log_level.upper(), format="%(message)s")
    try:
        asyncio.run(_dispatch(_parser().parse_args(), get_settings()))
    except (ValueError, RuntimeError) as error:
        logger.error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
