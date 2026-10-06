"""Bounded retrieval of indexable files from a GitHub repository revision."""

import asyncio
import io
import zipfile
from collections.abc import Iterable
from pathlib import PurePosixPath

from app.github.client import GitHubClient
from app.ingestion.file_discovery import is_indexable


class GitHubRepositoryLoader:
    """Load only indexable repository files from a pinned GitHub commit."""

    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    async def load(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
        installation_id: int,
        paths: Iterable[str] | None = None,
    ) -> dict[str, bytes]:
        """Fetch requested files or a bounded archive at the specified commit."""
        requested = set(paths) if paths is not None else None
        contents: dict[str, bytes] = {}
        if requested is not None:
            if len(requested) > 3000:
                raise ValueError("Incremental index update exceeds the 3000-file safety limit")
            semaphore = asyncio.Semaphore(10)

            async def load_file(path: str) -> tuple[str, bytes]:
                """Fetch one indexable file and return its repository path and bytes."""
                if not is_indexable(path):
                    raise ValueError(f"Refusing to load unsupported repository path: {path}")
                async with semaphore:
                    content = await self.client.get_file_content(
                        owner, repo, path, commit_sha, installation_id
                    )
                if len(content) > 1_000_000:
                    raise ValueError(f"Repository file exceeds the 1 MiB indexing limit: {path}")
                return path, content

            for path, content in await asyncio.gather(*(load_file(path) for path in requested)):
                contents[path] = content
            return contents
        archive = await self.client.get_repository_archive(owner, repo, commit_sha, installation_id)
        total = 0
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            entries = zipped.infolist()
            if len(entries) > 20_000:
                raise ValueError("Repository archive exceeds the 20,000-entry indexing limit")
            for entry in entries:
                if entry.is_dir() or entry.file_size > 1_000_000:
                    continue
                parts = PurePosixPath(entry.filename).parts
                if len(parts) < 2:
                    continue
                relative = "/".join(parts[1:])
                if not is_indexable(relative, entry.file_size):
                    continue
                total += entry.file_size
                if total > 100 * 1024 * 1024:
                    raise ValueError("Repository source exceeds the 100 MiB indexing limit")
                contents[relative] = zipped.read(entry)
        return contents
