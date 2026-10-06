"""Hybrid semantic and lexical retrieval over a commit-specific index."""

import re
from dataclasses import dataclass
from typing import Any

from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore


@dataclass(frozen=True)
class RetrievedContext:
    """Repository evidence returned with its source location, rank, and context type."""

    content: str
    file_path: str
    symbol: str
    symbol_type: str
    start_line: int
    end_line: int
    score: float
    context_type: str


class HybridRetriever:
    """Combine vector, lexical, file-local, test, and documentation search results."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vectors: QdrantVectorStore,
        *,
        top_k: int = 8,
    ) -> None:
        self.embeddings = embeddings
        self.vectors = vectors
        self.top_k = top_k

    async def retrieve(
        self,
        *,
        repository: str,
        commit_sha: str,
        query: str,
        symbol: str | None = None,
        file_path: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedContext]:
        """Rank relevant commit-scoped evidence, favoring symbol and local matches."""
        limit = top_k or self.top_k
        query_vector = (await self.embeddings.embed([query]))[0]
        semantic = self.vectors.search(repository, commit_sha, query_vector, limit * 4)
        terms = _tokens(query)
        lexical_candidates = self.vectors.lexical_search(
            repository, commit_sha, sorted(terms, key=len, reverse=True)[:8], limit * 4
        )
        local = self.vectors.chunks_for_file(repository, commit_sha, file_path) if file_path else []
        tests = self.vectors.search_kind(
            repository,
            commit_sha,
            kind="test",
            terms=sorted(terms, key=len, reverse=True)[:5],
            limit=limit,
        )
        documentation = self.vectors.search_kind(
            repository,
            commit_sha,
            kind="documentation",
            terms=sorted(terms, key=len, reverse=True)[:5],
            limit=max(1, limit // 2),
        )
        ranked: dict[str, tuple[float, dict[str, Any], str]] = {}

        for payload in semantic:
            key = _key(payload)
            score = float(payload.get("score", 0.0)) * 0.65
            ranked[key] = (score, payload, "semantic")
        for payload in [*lexical_candidates, *local, *tests, *documentation]:
            searchable = _tokens(
                " ".join(
                    str(payload.get(field, ""))
                    for field in ("symbol", "file_path", "content", "symbol_type")
                )
            )
            lexical_score = len(terms & searchable) / max(len(terms), 1)
            exact_symbol = bool(symbol and symbol.lower() in str(payload.get("symbol", "")).lower())
            docs_or_tests = _context_type(str(payload.get("file_path", "")))
            is_local = bool(file_path and payload.get("file_path") == file_path)
            score = lexical_score * 0.25 + (0.3 if exact_symbol else 0) + (0.2 if is_local else 0)
            if docs_or_tests in {"documentation", "test"} and lexical_score:
                score += 0.12
            key = _key(payload)
            if key not in ranked or score > ranked[key][0]:
                context_type = (
                    "symbol"
                    if exact_symbol
                    else "test"
                    if docs_or_tests == "test"
                    else "documentation"
                    if docs_or_tests == "documentation"
                    else "local"
                    if is_local
                    else "lexical"
                )
                ranked[key] = (score, payload, context_type)
        contexts = [
            RetrievedContext(
                content=str(payload.get("content", "")),
                file_path=str(payload.get("file_path", "")),
                symbol=str(payload.get("symbol", "")),
                symbol_type=str(payload.get("symbol_type", "")),
                start_line=int(payload.get("start_line", 1)),
                end_line=int(payload.get("end_line", 1)),
                score=score,
                context_type=context_type,
            )
            for score, payload, context_type in ranked.values()
            if score > 0
        ]
        return sorted(contexts, key=lambda context: context.score, reverse=True)[:limit]

    def chunks_for_file(
        self, repository: str, commit_sha: str, file_path: str
    ) -> list[dict[str, Any]]:
        """Return all indexed chunks for one file at the specified commit."""
        return self.vectors.chunks_for_file(repository, commit_sha, file_path)

    def all_chunks(self, repository: str, commit_sha: str) -> list[dict[str, Any]]:
        """Return all indexed chunks for a repository commit."""
        return self.vectors.all_for_commit(repository, commit_sha)

    def search_kind(
        self, repository: str, commit_sha: str, *, kind: str, query: str, limit: int
    ) -> list[dict[str, Any]]:
        """Search only the requested chunk kind, such as tests or documentation."""
        terms = sorted(_tokens(query), key=len, reverse=True)[:5]
        return self.vectors.search_kind(repository, commit_sha, kind=kind, terms=terms, limit=limit)


def _tokens(text: str) -> set[str]:
    """Extract normalized identifier-like terms for lexical retrieval."""
    return {token for token in re.findall(r"[A-Za-z_][A-Za-z_0-9]{1,}", text.lower())}


def _key(payload: dict[str, Any]) -> str:
    """Normalize retrieval text for deterministic lexical matching."""
    return f"{payload.get('file_path')}:{payload.get('symbol')}:{payload.get('start_line')}"


def _context_type(path: str) -> str:
    """Classify retrieved chunks for use in downstream review context."""
    lowered = path.lower()
    if any(part in lowered for part in ("readme", "docs/", "doc/")):
        return "documentation"
    if any(part in lowered for part in ("test", "spec")):
        return "test"
    return "lexical"
