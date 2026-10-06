"""Validated, bounded repository tools exposed to the review model."""

import inspect
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.models.review import StaticFinding
from app.retrieval.retriever import HybridRetriever


class ToolArguments(BaseModel):
    """Base schema that rejects undeclared tool arguments."""

    model_config = ConfigDict(extra="forbid")


class SearchCodeArguments(ToolArguments):
    """Validated query fields for semantic and lexical code search."""

    query: str = Field(min_length=2, max_length=1000)
    symbol: str | None = Field(default=None, max_length=300)
    top_k: int = Field(default=5, ge=1, le=8)


class GetFileArguments(ToolArguments):
    """Validated repository-relative path for source retrieval."""

    path: str = Field(min_length=1, max_length=1000)


class GetSymbolArguments(ToolArguments):
    """Validated symbol name for code-context lookup."""

    name: str = Field(min_length=1, max_length=300)


class GetDependencyArguments(ToolArguments):
    """Validated symbol name for dependency-context lookup."""

    name: str = Field(min_length=1, max_length=300)


class GetTestsArguments(ToolArguments):
    """Validated symbol name for relevant test retrieval."""

    symbol: str = Field(min_length=1, max_length=300)


class GetDocumentationArguments(ToolArguments):
    """Validated query for repository documentation retrieval."""

    query: str = Field(min_length=2, max_length=1000)


class EmptyArguments(ToolArguments):
    """Empty argument schema for tools that need no parameters."""

    pass


class ReviewToolbox:
    """Expose bounded repository search, source, diff, and analysis tools to the agent."""

    def __init__(
        self,
        *,
        repository: str,
        commit_sha: str,
        retriever: HybridRetriever,
        files: dict[str, bytes],
        diff: str,
        static_findings: list[StaticFinding],
        dependencies: dict[str, list[str]] | None = None,
    ) -> None:
        self.repository = repository
        self.commit_sha = commit_sha
        self.retriever = retriever
        self.files = files
        self.diff = diff[:30000]
        self.static_findings = static_findings
        self.dependencies = dependencies or {}

    async def execute(self, name: str, arguments: str) -> object:
        """Validate tool arguments and invoke the matching repository operation."""
        handlers: dict[str, tuple[type[ToolArguments], Any]] = {
            "search_code": (SearchCodeArguments, self.search_code),
            "get_file": (GetFileArguments, self.get_file),
            "get_symbol": (GetSymbolArguments, self.get_symbol),
            "get_callers": (GetSymbolArguments, self.get_callers),
            "get_dependencies": (GetDependencyArguments, self.get_dependencies),
            "get_tests": (GetTestsArguments, self.get_tests),
            "get_documentation": (GetDocumentationArguments, self.get_documentation),
            "get_diff": (EmptyArguments, self.get_diff),
            "run_static_analysis": (EmptyArguments, self.run_static_analysis),
        }
        if name not in handlers:
            raise ValueError(f"Unknown review tool: {name}")
        model, handler = handlers[name]
        parsed = model.model_validate_json(arguments)
        result = handler(**parsed.model_dump())
        return await result if inspect.isawaitable(result) else result

    async def search_code(self, query: str, symbol: str | None = None, top_k: int = 5) -> object:
        """Search indexed repository context for a query or changed symbol."""
        return [
            item.__dict__
            for item in await self.retriever.retrieve(
                repository=self.repository,
                commit_sha=self.commit_sha,
                query=query,
                symbol=symbol,
                top_k=top_k,
            )
        ]

    def get_file(self, path: str) -> object:
        """Return bounded decoded content for an indexable repository file."""
        if path in self.files:
            content = self.files[path].decode("utf-8", errors="replace")
            return {"path": path, "content": content[:12000]}
        chunks = self.retriever.chunks_for_file(self.repository, self.commit_sha, path)
        if not chunks:
            return {"error": "File is not present in the indexed repository snapshot"}
        return {"path": path, "chunks": chunks[:8]}

    async def get_symbol(self, name: str) -> object:
        """Retrieve indexed context associated with a named symbol."""
        return await self.search_code(name, symbol=name, top_k=5)

    async def get_callers(self, name: str) -> object:
        """Retrieve likely caller and reference context for a named symbol."""
        return await self.search_code(
            f"calls references to callers of {name}", symbol=name, top_k=5
        )

    async def get_dependencies(self, name: str) -> object:
        """Retrieve likely import and dependency context for a named symbol."""
        related = await self.search_code(
            f"imports and dependencies used by {name}", symbol=name, top_k=5
        )
        return {"imports": self.dependencies.get(name, []), "related_code": related}

    async def get_tests(self, symbol: str) -> object:
        """Find indexed test chunks related to a symbol."""
        return self.retriever.search_kind(
            self.repository,
            self.commit_sha,
            kind="test",
            query=symbol,
            limit=5,
        )

    async def get_documentation(self, query: str) -> object:
        """Find indexed documentation relevant to a query."""
        return self.retriever.search_kind(
            self.repository,
            self.commit_sha,
            kind="documentation",
            query=query,
            limit=5,
        )

    def get_diff(self) -> object:
        """Expose the current pull-request diff to the review agent."""
        return {"diff": self.diff}

    def run_static_analysis(self) -> object:
        """Expose deterministic static-analysis results to the review agent."""
        return [finding.model_dump() for finding in self.static_findings]

    def schemas(self) -> list[dict[str, Any]]:
        """Return strict JSON schemas and descriptions for available review tools."""
        definitions: list[tuple[str, str, Any]] = [
            (
                "search_code",
                "Search indexed repository code using semantic and lexical retrieval.",
                SearchCodeArguments,
            ),
            (
                "get_file",
                "Read a repository file from the indexed commit snapshot.",
                GetFileArguments,
            ),
            ("get_symbol", "Search for a symbol definition and related code.", GetSymbolArguments),
            ("get_callers", "Search for likely callers of a symbol.", GetSymbolArguments),
            (
                "get_dependencies",
                "Search for dependencies and imports associated with a symbol.",
                GetDependencyArguments,
            ),
            ("get_tests", "Find repository tests referencing a symbol.", GetTestsArguments),
            (
                "get_documentation",
                "Search repository documentation for a topic.",
                GetDocumentationArguments,
            ),
            ("get_diff", "Inspect the actual pull request diff.", EmptyArguments),
            (
                "run_static_analysis",
                "Inspect deterministic findings already produced for this review.",
                EmptyArguments,
            ),
        ]
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": _strict_schema(schema),
                },
            }
            for name, description, schema in definitions
        ]


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic argument model to a strict tool-call JSON schema."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema["additionalProperties"] = False
    schema["required"] = list(schema.get("properties", {}).keys())
    return schema


async def execute_tool(name: str, arguments: str) -> object:
    """Dispatch a model tool call through the request-scoped review toolbox."""
    from app.agents.review_agent import current_toolbox

    toolbox = current_toolbox.get()
    if toolbox is None:
        raise RuntimeError("Review tools are unavailable outside an active review")
    try:
        return await toolbox.execute(name, arguments)
    except ValidationError as error:
        raise ValueError(str(error)) from error


def encode_tool_result(value: object) -> str:
    """Serialize tool output safely for inclusion in the model conversation."""
    return json.dumps(value, ensure_ascii=True)
