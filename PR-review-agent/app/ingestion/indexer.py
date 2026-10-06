"""Incremental embedding and indexing of repository snapshots."""

import logging
from collections.abc import Mapping

from app.ingestion.chunker import CodeChunker
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class RepositoryIndexer:
    """Coordinate chunking, embeddings, and immutable vector snapshots."""

    def __init__(
        self,
        chunker: CodeChunker,
        embeddings: EmbeddingProvider,
        vectors: QdrantVectorStore,
    ) -> None:
        self.chunker = chunker
        self.embeddings = embeddings
        self.vectors = vectors

    async def index_files(
        self,
        *,
        repository: str,
        commit_sha: str,
        files: Mapping[str, bytes],
        previous_commit_sha: str | None = None,
        full_reindex: bool = False,
        deleted_files: set[str] | None = None,
    ) -> int:
        """Index changed files and optionally reuse unchanged vectors from a prior commit."""
        if previous_commit_sha and not full_reindex and previous_commit_sha != commit_sha:
            self.vectors.clone_commit(repository, previous_commit_sha, commit_sha)
        for path in deleted_files or set():
            self.vectors.delete_file(repository, commit_sha, path)
        chunks = []
        for path, content in files.items():
            self.vectors.delete_file(repository, commit_sha, path)
            try:
                source = content.decode("utf-8")
            except UnicodeDecodeError:
                logger.info("Skipping non-UTF-8 source file: %s", path)
                continue
            chunks.extend(
                self.chunker.chunk(
                    source, repository=repository, commit_sha=commit_sha, file_path=path
                )
            )
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            vectors = await self.embeddings.embed([chunk.content for chunk in batch])
            self.vectors.upsert(batch, vectors)
        return len(chunks)
