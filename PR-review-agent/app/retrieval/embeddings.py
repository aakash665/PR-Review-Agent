"""Embedding provider contracts and cached OpenRouter embeddings."""

import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Protocol

import httpx

from app.config import Settings
from app.database.session import Database


@asynccontextmanager
async def _http_client(
    client: httpx.AsyncClient | None,
    timeout: httpx.Timeout,
    default_headers: dict[str, str],
) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an injected HTTP client or a request-scoped client."""
    if client is not None:
        client.headers.update(default_headers)
        yield client
    else:
        async with httpx.AsyncClient(timeout=timeout, headers=default_headers) as owned_client:
            yield owned_client


class EmbeddingProvider(Protocol):
    """Contract for embedding batches into vectors of a consistent dimension."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector for each input text, preserving input order."""
        ...


class EmbeddingError(RuntimeError):
    """Raised when an embedding request fails or returns invalid vectors."""

    pass


class OpenRouterEmbeddingProvider:
    """Fetch and cache vectors through OpenRouter's embeddings endpoint."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if settings.openrouter_api_key is None:
            raise ValueError("OPENROUTER_API_KEY is required for hosted embeddings")
        self.settings = settings
        self.api_key = settings.openrouter_api_key.get_secret_value()
        self.database = database
        self.http_client = http_client
        self.total_latency_seconds = 0.0
        self.total_tokens = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return cached or newly generated vectors in input order."""
        if not texts:
            return []
        cache_keys = [self._cache_key(text) for text in texts]
        vectors: dict[str, list[float]] = {}
        missing: list[tuple[str, str]] = []
        with self.database.connect() as connection:
            for key, text in zip(cache_keys, texts, strict=True):
                row = connection.execute(
                    "SELECT vector_json FROM embedding_cache WHERE cache_key=?", (key,)
                ).fetchone()
                if row:
                    vectors[key] = json.loads(row["vector_json"])
                else:
                    missing.append((key, text))
        for batch_start in range(0, len(missing), 64):
            batch = missing[batch_start : batch_start + 64]
            started = time.monotonic()
            try:
                async with _http_client(
                    self.http_client,
                    httpx.Timeout(45, connect=10),
                    self._attribution_headers(),
                ) as client:
                    response = await client.post(
                        f"{self.settings.openrouter_base_url.rstrip('/')}/api/v1/embeddings",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.settings.embedding_model,
                            "input": [text for _, text in batch],
                            "dimensions": self.settings.embedding_dimensions,
                        },
                    )
            finally:
                self.total_latency_seconds += time.monotonic() - started
            if response.is_error:
                raise EmbeddingError(f"Embedding API returned HTTP {response.status_code}")
            try:
                payload = response.json()
                embedded = payload["data"]
                self.total_tokens += int(payload.get("usage", {}).get("total_tokens", 0))
                if len(embedded) != len(batch):
                    raise ValueError("embedding response count mismatch")
                for item, (key, _) in zip(embedded, batch, strict=True):
                    vector = item["embedding"]
                    if len(vector) != self.settings.embedding_dimensions:
                        raise ValueError("embedding dimension does not match configuration")
                    vectors[key] = vector
            except (KeyError, TypeError, ValueError) as error:
                raise EmbeddingError("Malformed embedding API response") from error
            with self.database.connect() as connection:
                connection.executemany(
                    """INSERT OR REPLACE INTO embedding_cache(cache_key, model, vector_json)
                       VALUES (?, ?, ?)""",
                    [
                        (key, self.settings.embedding_model, json.dumps(vectors[key]))
                        for key, _ in batch
                    ],
                )
        return [vectors[key] for key in cache_keys]

    def _cache_key(self, text: str) -> str:
        """Create a stable key for a normalized embedding request."""
        return hashlib.sha256(f"{self.settings.embedding_model}\0{text}".encode()).hexdigest()

    def _attribution_headers(self) -> dict[str, str]:
        """Return optional application attribution headers supported by OpenRouter."""
        headers = {"X-OpenRouter-Title": self.settings.openrouter_app_name}
        if self.settings.openrouter_site_url:
            headers["HTTP-Referer"] = self.settings.openrouter_site_url
        return headers


class LocalFeatureEmbeddingProvider:
    """Offline deterministic feature hashing for fixtures, not hosted embeddings."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate normalized deterministic feature-hash vectors for fixture text."""
        vectors: list[list[float]] = []
        for text in texts:
            tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]{1,}", text.lower())
            counts = Counter(tokens)
            vector = [0.0] * self.dimensions
            for token, frequency in counts.items():
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign * (1 + math.log(frequency))
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors
