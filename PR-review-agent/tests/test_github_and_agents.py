"""Tests for GitHub API behavior, review tools, and agent validation."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr
from qdrant_client import QdrantClient

from app.agents.client import LLMClient
from app.agents.decisions_client import DecisionsClient
from app.agents.review_agent import ReviewAgent, ReviewResponse
from app.agents.tools import ReviewToolbox
from app.agents.verifier_agent import VerifierAgent
from app.analysis.diff_analyzer import DiffAnalysis, DiffAnalyzer
from app.config import Settings
from app.github.client import GitHubClient
from app.ingestion.chunker import CodeChunker
from app.ingestion.indexer import RepositoryIndexer
from app.models.review import PullRequestMetadata, ReviewFinding
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.embeddings import LocalFeatureEmbeddingProvider
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import QdrantVectorStore
from app.review.github_publisher import GitHubReviewPublisher


class FakeAuthenticator:
    def app_jwt(self) -> str:
        """Sign and return a short-lived JWT for GitHub App authentication."""
        return "app-jwt"


class FakeLLM:
    async def complete(self, **kwargs: Any) -> tuple[Any, int]:
        """Persist successful job findings and optional run metrics atomically."""
        schema = kwargs["response_model"]
        if schema is ReviewResponse:
            return (
                ReviewResponse(
                    findings=[
                        ReviewFinding(
                            category="security",
                            severity="high",
                            confidence=0.94,
                            title="SQL injection",
                            explanation="The query interpolates caller-controlled input.",
                            evidence=["The changed query uses an f-string."],
                            file_path="app/service.py",
                            line_start=5,
                            line_end=5,
                            suggestion="Use a bound parameter.",
                            rationale="The existing database helper binds query values.",
                        ),
                        ReviewFinding(
                            category="bug",
                            severity="high",
                            confidence=0.95,
                            title="Invented line",
                            explanation="This finding refers to code not in the patch.",
                            evidence=["Unsupported claim"],
                            file_path="app/service.py",
                            line_start=900,
                            line_end=900,
                            suggestion=None,
                            rationale="This is intentionally invalid.",
                        ),
                    ],
                    summary="User input must stay parameterized.",
                    positive_changes=[],
                    confidence=0.9,
                ),
                123,
            )
        raise AssertionError(f"Unexpected response model: {schema}")


class FakeDecisions:
    def __init__(self, probability: float = 0.9) -> None:
        self.probability = probability

    async def assess_findings(
        self, state: dict[str, Any], indices: list[int]
    ) -> tuple[dict[int, float], int]:
        assert state["repository_context_is_untrusted"] is True
        return {index: self.probability for index in indices}, 45


async def _fixture_pipeline() -> tuple[
    QdrantClient,
    HybridRetriever,
    PullRequestMetadata,
    DiffAnalysis,
    dict[str, bytes],
]:
    root = "fixtures/python_project"
    files = {
        "app/service.py": Path(f"{root}/app/service.py").read_bytes(),
        "app/safe_database.py": Path(f"{root}/app/safe_database.py").read_bytes(),
        "README.md": Path(f"{root}/README.md").read_bytes(),
    }
    commit_sha = "fixture-head-sha"
    client = QdrantClient(location=":memory:")
    vectors = QdrantVectorStore(
        ":memory:",
        "agent_tool_test_chunks",
        384,
        client=client,
        create_payload_indexes=False,
    )
    embeddings = LocalFeatureEmbeddingProvider(384)
    await RepositoryIndexer(CodeChunker(), embeddings, vectors).index_files(
        repository="1",
        commit_sha=commit_sha,
        files=files,
        full_reindex=True,
    )
    retriever = HybridRetriever(embeddings, vectors, top_k=8)
    diff = DiffAnalyzer().analyze(Path(f"{root}/sample.diff").read_text(encoding="utf-8"))
    metadata = PullRequestMetadata(
        repository="1",
        repository_id=1,
        owner="fixture",
        name="python_project",
        number=1,
        title="Use safer query input",
        base_branch="main",
        head_branch="feature",
        head_sha=commit_sha,
        author="fixture-author",
        files=diff.files,
    )
    return client, retriever, metadata, diff, files


def test_review_agent_filters_locations_and_verifier_accepts_evidence() -> None:
    async def run() -> None:
        """Run supported static-analysis tools and normalize their findings."""
        client, retriever, metadata, diff, files = await _fixture_pipeline()
        try:
            result = await ReviewAgent(
                FakeLLM(),
                retriever,
                ContextBuilder(4000),  # type: ignore[arg-type]
            ).review(
                metadata=metadata,
                diff=diff,
                repository_files=files,
                static_findings=[],
            )
            assert len(result.findings) == 1
            assert result.findings[0].line_start == 5
            assert result.retrieved_chunks > 0
            assert "<UNTRUSTED_REPOSITORY_CONTEXT>" in result.retrieved_context
            accepted, tokens = await VerifierAgent(FakeDecisions()).verify(
                result.findings,
                context=result.retrieved_context,
                diff=diff,
            )
            assert len(accepted) == 1
            assert accepted[0].verified is True
            assert tokens == 45
            rejected, _ = await VerifierAgent(FakeDecisions(0.79), minimum_probability=0.8).verify(
                result.findings,
                context=result.retrieved_context,
                diff=diff,
            )
            assert rejected == []
        finally:
            client.close()

    asyncio.run(run())


def test_openrouter_chat_client_uses_configured_model_and_endpoint() -> None:
    async def run() -> None:
        observed: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            observed.append(request)
            assert request.url.path == "/api/v1/chat/completions"
            assert request.headers["Authorization"] == "Bearer test-openrouter-key"
            payload = json.loads(request.read())
            output_schema = payload["response_format"]["json_schema"]
            assert payload["response_format"]["type"] == "json_schema"
            assert output_schema["strict"] is True
            finding_schema = output_schema["schema"]["$defs"]["ReviewFinding"]
            assert {
                "category",
                "severity",
                "confidence",
                "file_path",
                "line_start",
                "line_end",
            } <= set(finding_schema["required"])
            assert finding_schema["additionalProperties"] is False
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "findings": [],
                                        "summary": "No actionable defects found.",
                                        "positive_changes": [],
                                        "confidence": 0.9,
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"total_tokens": 21},
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            settings = Settings.model_construct(
                openrouter_api_key=SecretStr("test-openrouter-key"),
                llm_model="vendor/chat-model",
            )
            _, tokens = await LLMClient(settings, http_client=http).complete(
                system="Review",
                user="Changed code",
                response_model=ReviewResponse,
            )
        assert tokens == 21
        assert json.loads(observed[0].content)["model"] == "vendor/chat-model"

    asyncio.run(run())


def test_openrouter_decisions_api_returns_validated_probabilities() -> None:
    async def run() -> None:
        observed: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            observed.append(request)
            assert request.url.path == "/api/alpha/decisions"
            assert request.headers["Authorization"] == "Bearer test-openrouter-key"
            return httpx.Response(
                200,
                json={
                    "answers": {
                        "finding_2": {"noul": 0.93},
                        "finding_4": {"noul": 0.18},
                    },
                    "usage": {"total_tokens": 12},
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            settings = Settings.model_construct(openrouter_api_key=SecretStr("test-openrouter-key"))
            probabilities, tokens = await DecisionsClient(
                settings, http_client=http
            ).assess_findings({"candidate_findings": []}, [2, 4])
        assert probabilities == {2: 0.93, 4: 0.18}
        assert tokens == 12
        request_payload = json.loads(observed[0].content)
        assert request_payload["model"] == "openai/gpt-6-luna-decisions"
        assert set(request_payload["questions"]) == {"finding_2", "finding_4"}

    asyncio.run(run())


def test_review_tools_expose_typed_bounded_schemas() -> None:
    async def run() -> None:
        """Run supported static-analysis tools and normalize their findings."""
        client, retriever, _, _, files = await _fixture_pipeline()
        try:
            toolbox = ReviewToolbox(
                repository="1",
                commit_sha="fixture-head-sha",
                retriever=retriever,
                files=files,
                diff="diff",
                static_findings=[],
            )
            tools = toolbox.schemas()
            names = {tool["function"]["name"] for tool in tools}
            assert {"search_code", "get_file", "get_symbol", "get_callers"} <= names
            assert {"get_dependencies", "get_tests", "get_documentation"} <= names
            assert all(
                tool["function"]["parameters"]["additionalProperties"] is False for tool in tools
            )
            file_result = await toolbox.execute("get_file", '{"path":"app/safe_database.py"}')
            assert isinstance(file_result, dict)
            assert "execute" in file_result["content"]
        finally:
            client.close()

    asyncio.run(run())


def test_github_app_client_uses_installation_token_and_fetches_actual_data() -> None:
    async def run() -> None:
        """Run supported static-analysis tools and normalize their findings."""
        observed: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/app/installations/77/access_tokens":
                assert request.headers["Authorization"] == "Bearer app-jwt"
                return httpx.Response(201, json={"token": "installation-token"})
            observed.append(request.headers["Authorization"])
            return httpx.Response(200, json={"id": 1, "default_branch": "main"})

        transport = httpx.MockTransport(handler)
        http = httpx.AsyncClient(base_url="https://api.github.test", transport=transport)
        settings = Settings(_env_file=None, github_api_url="https://api.github.test")
        github = GitHubClient(
            settings,
            http_client=http,
            authenticator=FakeAuthenticator(),  # type: ignore[arg-type]
        )
        try:
            repository = await github.get_repository("octo", "demo", 77)
            assert repository["default_branch"] == "main"
            assert observed == ["Bearer installation-token"]
        finally:
            await github.close()
            await http.aclose()

    asyncio.run(run())


def test_github_publisher_posts_only_changed_added_lines_and_is_idempotent() -> None:
    class FakeGitHub:
        def __init__(self) -> None:
            self.reviews: list[dict[str, Any]] = []
            self.existing: list[dict[str, Any]] = []

        async def list_pull_request_reviews(self, *args: Any) -> list[dict[str, Any]]:
            """List existing reviews to support idempotent publication."""
            return self.existing

        async def create_pull_request_review(self, *args: Any, **kwargs: Any) -> dict[str, int]:
            """Create a pull-request review with an optional list of inline comments."""
            self.reviews.append(kwargs)
            return {"id": 42}

    async def run() -> None:
        """Run supported static-analysis tools and normalize their findings."""
        github = FakeGitHub()
        publisher = GitHubReviewPublisher(github)  # type: ignore[arg-type]
        fixture_client, _, _, diff, _ = await _fixture_pipeline()
        finding = ReviewFinding(
            category="security",
            severity="high",
            confidence=0.9,
            title="Unsafe query interpolation",
            explanation="Untrusted input is interpolated into SQL.",
            evidence=["The source line is an f-string."],
            file_path="app/service.py",
            line_start=5,
            line_end=5,
            suggestion="Bind the value as a query parameter.",
            rationale="The repository already uses bound parameters.",
            verified=True,
        )
        try:
            review_id = await publisher.publish(
                owner="octo",
                repo="demo",
                pr_number=7,
                head_sha="head123",
                installation_id=77,
                summary="## AI Code Review",
                findings=[finding],
                diff=diff,
            )
            assert review_id == 42
            review = github.reviews[0]
            assert review["comments"][0]["line"] == 5
            assert review["comments"][0]["side"] == "RIGHT"
            assert review["event"] == "COMMENT"
            github.existing = [{"body": "<!-- github-pr-review-agent:head123 -->"}]
            duplicate = await publisher.publish(
                owner="octo",
                repo="demo",
                pr_number=7,
                head_sha="head123",
                installation_id=77,
                summary="duplicate",
                findings=[finding],
                diff=diff,
            )
            assert duplicate == 0
            assert len(github.reviews) == 1
        finally:
            fixture_client.close()

    asyncio.run(run())
