"""Qdrant storage and filtering for immutable repository commit snapshots."""

import uuid
from typing import Any, cast

from qdrant_client import QdrantClient, models

from app.ingestion.chunker import CodeChunk


class QdrantVectorStore:
    """Store and query repository chunks partitioned by repository ID and commit SHA."""

    def __init__(
        self,
        url: str,
        collection: str,
        dimensions: int,
        *,
        client: QdrantClient | None = None,
        create_payload_indexes: bool = True,
    ) -> None:
        self.collection = collection
        self.dimensions = dimensions
        self.create_payload_indexes = create_payload_indexes
        self.client = client or QdrantClient(url=url, timeout=30)
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Create the collection or reject it when its configured vector size differs."""
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            configured_vectors = info.config.params.vectors
            if not isinstance(configured_vectors, models.VectorParams):
                raise ValueError("Qdrant collection must use one unnamed vector")
            configured_size = configured_vectors.size
            if configured_size != self.dimensions:
                raise ValueError(
                    f"Qdrant collection dimension is {configured_size}, expected {self.dimensions}"
                )
        else:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimensions, distance=models.Distance.COSINE
                ),
            )
        if not self.create_payload_indexes:
            return
        for field_name, field_type in [
            ("repository", models.PayloadSchemaType.KEYWORD),
            ("commit_sha", models.PayloadSchemaType.KEYWORD),
            ("file_path", models.PayloadSchemaType.KEYWORD),
            ("symbol", models.PayloadSchemaType.KEYWORD),
            ("symbol_type", models.PayloadSchemaType.KEYWORD),
            ("is_test", models.PayloadSchemaType.BOOL),
            ("is_documentation", models.PayloadSchemaType.BOOL),
        ]:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=field_type,
            )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="content",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
                min_token_len=2,
                max_token_len=32,
            ),
        )

    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> None:
        """Store chunks and matching embeddings, requiring one vector per chunk."""
        if len(chunks) != len(vectors):
            raise ValueError("Each code chunk must have exactly one embedding")
        if not chunks:
            return
        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(id=chunk.chunk_id, vector=vector, payload=chunk.payload())
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
            wait=True,
        )

    def clone_commit(self, repository: str, old_sha: str, new_sha: str) -> int:
        """Copy one commit snapshot to a new SHA and return the number of copied points."""
        copied = 0
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=self._filter(repository, old_sha),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            upsert_points: list[models.PointStruct] = []
            for point in points:
                if not isinstance(point.vector, list) or not all(
                    isinstance(value, (int, float)) for value in point.vector
                ):
                    raise ValueError("Qdrant returned an indexed chunk without its vector")
                vector = cast(list[float], point.vector)
                payload = dict(point.payload or {})
                payload["commit_sha"] = new_sha
                identity = (
                    f"{repository}:{new_sha}:{payload.get('file_path')}:{payload.get('symbol')}:"
                    f"{payload.get('start_line')}:{payload.get('chunk_index', 0)}"
                )
                upsert_points.append(
                    models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
                        vector=vector,
                        payload=payload,
                    )
                )
            if upsert_points:
                self.client.upsert(collection_name=self.collection, points=upsert_points, wait=True)
                copied += len(upsert_points)
            if offset is None:
                return copied

    def delete_file(self, repository: str, commit_sha: str, file_path: str) -> None:
        """Delete all indexed chunks for one file in a commit snapshot."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repository", match=models.MatchValue(value=repository)
                        ),
                        models.FieldCondition(
                            key="commit_sha", match=models.MatchValue(value=commit_sha)
                        ),
                        models.FieldCondition(
                            key="file_path", match=models.MatchValue(value=file_path)
                        ),
                    ]
                )
            ),
            wait=True,
        )

    def delete_commit(self, repository: str, commit_sha: str) -> None:
        """Delete all vectors belonging to the specified repository commit."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=self._filter(repository, commit_sha)),
            wait=True,
        )

    def delete_repository(self, repository: str) -> None:
        """Delete every indexed commit snapshot for a repository."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=self._repository_filter(repository)),
            wait=True,
        )

    def search(
        self, repository: str, commit_sha: str, vector: list[float], limit: int
    ) -> list[dict[str, Any]]:
        """Return nearest vector matches restricted to one repository commit."""
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=self._filter(repository, commit_sha),
            limit=limit,
            with_payload=True,
        )
        return [{**(point.payload or {}), "score": point.score} for point in result.points]

    def lexical_search(
        self, repository: str, commit_sha: str, terms: list[str], limit: int
    ) -> list[dict[str, Any]]:
        """Return commit-scoped chunks matching lexical terms."""
        if not terms:
            return []
        result, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=self._filter_conditions(repository, commit_sha),
                should=[
                    models.FieldCondition(key="content", match=models.MatchText(text=term))
                    for term in terms
                ],
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [point.payload or {} for point in result]

    def search_kind(
        self,
        repository: str,
        commit_sha: str,
        *,
        kind: str,
        terms: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search only test or documentation chunks for the supplied terms."""
        field = {"test": "is_test", "documentation": "is_documentation"}.get(kind)
        if field is None or not terms:
            return []
        result, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    *self._filter_conditions(repository, commit_sha),
                    models.FieldCondition(key=field, match=models.MatchValue(value=True)),
                ],
                should=[
                    models.FieldCondition(key="content", match=models.MatchText(text=term))
                    for term in terms
                ],
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [point.payload or {} for point in result]

    def chunks_for_file(
        self, repository: str, commit_sha: str, file_path: str
    ) -> list[dict[str, Any]]:
        """Return every stored chunk for one file and commit."""
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    *self._filter_conditions(repository, commit_sha),
                    models.FieldCondition(
                        key="file_path", match=models.MatchValue(value=file_path)
                    ),
                ]
            ),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        return [point.payload or {} for point in points]

    def all_for_commit(self, repository: str, commit_sha: str) -> list[dict[str, Any]]:
        """Return all stored chunk payloads for one repository commit."""
        results: list[dict[str, Any]] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=self._filter(repository, commit_sha),
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            results.extend(point.payload or {} for point in points)
            if offset is None:
                return results

    @staticmethod
    def _repository_filter(repository: str) -> models.Filter:
        """Build the mandatory stable repository and commit filter."""
        return models.Filter(
            must=[
                models.FieldCondition(key="repository", match=models.MatchValue(value=repository))
            ]
        )

    @staticmethod
    def _filter_conditions(repository: str, commit_sha: str) -> list[models.Condition]:
        """Add optional file or chunk-kind conditions to a vector-store filter."""
        return [
            models.FieldCondition(key="repository", match=models.MatchValue(value=repository)),
            models.FieldCondition(key="commit_sha", match=models.MatchValue(value=commit_sha)),
        ]

    @classmethod
    def _filter(cls, repository: str, commit_sha: str) -> models.Filter:
        """Combine repository identity, commit, and optional query constraints."""
        return models.Filter(must=cls._filter_conditions(repository, commit_sha))
