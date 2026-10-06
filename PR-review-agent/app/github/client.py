"""Authenticated GitHub API operations with response and download limits."""

import asyncio
import base64
import logging
import time
from typing import Any, cast
from urllib.parse import quote

import httpx

from app.config import Settings
from app.github.auth import GitHubAppAuthenticator

logger = logging.getLogger(__name__)


class GitHubAPIError(RuntimeError):
    """Raised when a GitHub API request fails or returns malformed data."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"GitHub API returned HTTP {status_code}: {message}")
        self.status_code = status_code


def _json_object(response: httpx.Response) -> dict[str, Any]:
    """Decode a response body that must contain a JSON object."""
    try:
        payload = response.json()
    except ValueError as error:
        raise GitHubAPIError(502, "GitHub returned malformed JSON") from error
    if not isinstance(payload, dict):
        raise GitHubAPIError(502, "GitHub returned an unexpected JSON object")
    return cast(dict[str, Any], payload)


def _json_array(response: httpx.Response) -> list[dict[str, Any]]:
    """Decode a response body that must contain a JSON array."""
    try:
        payload = response.json()
    except ValueError as error:
        raise GitHubAPIError(502, "GitHub returned malformed JSON") from error
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise GitHubAPIError(502, "GitHub returned an unexpected JSON list")
    return cast(list[dict[str, Any]], payload)


class GitHubClient:
    """Perform authenticated GitHub API requests with bounded response handling."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
        authenticator: GitHubAppAuthenticator | None = None,
    ) -> None:
        if authenticator is None:
            if not settings.github_app_id or not settings.github_private_key_path:
                raise ValueError("GitHub App ID and private key path are required")
            authenticator = GitHubAppAuthenticator(
                settings.github_app_id, settings.github_private_key_path
            )
        self.settings = settings
        self.authenticator = authenticator
        self.http = http_client or httpx.AsyncClient(
            base_url=settings.github_api_url.rstrip("/"),
            timeout=httpx.Timeout(30, connect=10),
            follow_redirects=True,
            headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
        )
        self._owns_client = http_client is None
        self._tokens: dict[int, tuple[str, float]] = {}

    async def close(self) -> None:
        """Close the HTTP client only when this instance owns it."""
        if self._owns_client:
            await self.http.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        installation_id: int,
        accept: str | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Perform a retried authenticated API request with bounded JSON handling."""
        token = await self._installation_token(installation_id)
        headers = {"Authorization": f"Bearer {token}"}
        if accept:
            headers["Accept"] = accept
        for attempt in range(4):
            response = await self.http.request(method, path, headers=headers, **kwargs)
            if response.status_code < 400:
                return response
            retryable = response.status_code in {429, 500, 502, 503, 504}
            if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
                retryable = True
            if not retryable or attempt == 3:
                logger.warning(
                    "GitHub API request failed: method=%s status=%s", method, response.status_code
                )
                raise GitHubAPIError(response.status_code, "request failed")
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
            await asyncio.sleep(min(delay, 30))
        raise RuntimeError("GitHub API retry loop exited unexpectedly")

    async def _download(
        self,
        path: str,
        *,
        installation_id: int,
        accept: str,
        max_bytes: int,
    ) -> bytes:
        """Stream a GitHub download while enforcing the configured byte limit."""
        token = await self._installation_token(installation_id)
        headers = {"Authorization": f"Bearer {token}", "Accept": accept}
        for attempt in range(4):
            async with self.http.stream("GET", path, headers=headers) as response:
                if response.status_code >= 400:
                    retryable = response.status_code in {429, 500, 502, 503, 504}
                    if (
                        response.status_code == 403
                        and response.headers.get("X-RateLimit-Remaining") == "0"
                    ):
                        retryable = True
                    if not retryable or attempt == 3:
                        logger.warning("GitHub download failed: status=%s", response.status_code)
                        raise GitHubAPIError(response.status_code, "download request failed")
                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                    )
                else:
                    content_length = response.headers.get("Content-Length")
                    if (
                        content_length
                        and content_length.isdigit()
                        and int(content_length) > max_bytes
                    ):
                        raise GitHubAPIError(
                            413, "GitHub download exceeds the configured size limit"
                        )
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            raise GitHubAPIError(
                                413, "GitHub download exceeds the configured size limit"
                            )
                    return bytes(body)
            await asyncio.sleep(min(delay, 30))
        raise RuntimeError("GitHub download retry loop exited unexpectedly")

    async def _installation_token(self, installation_id: int) -> str:
        """Cache or request a short-lived GitHub App installation token."""
        cached = self._tokens.get(installation_id)
        if cached and cached[1] > time.monotonic():
            return cached[0]
        response = await self.http.post(
            f"/app/installations/{installation_id}/access_tokens",
            headers={"Authorization": f"Bearer {self.authenticator.app_jwt()}"},
        )
        if response.is_error:
            raise GitHubAPIError(response.status_code, "installation token request failed")
        token = _json_object(response).get("token")
        if not isinstance(token, str) or not token:
            raise GitHubAPIError(502, "GitHub returned an invalid installation token")
        self._tokens[installation_id] = (token, time.monotonic() + 3000)
        return token

    async def get_pull_request(
        self, owner: str, repo: str, number: int, installation_id: int
    ) -> dict[str, Any]:
        """Fetch pull-request metadata through the GitHub REST API."""
        response = await self._request(
            "GET", f"/repos/{owner}/{repo}/pulls/{number}", installation_id=installation_id
        )
        return _json_object(response)

    async def get_repository(self, owner: str, repo: str, installation_id: int) -> dict[str, Any]:
        """Fetch repository metadata through the GitHub REST API."""
        response = await self._request(
            "GET", f"/repos/{owner}/{repo}", installation_id=installation_id
        )
        return _json_object(response)

    async def get_branch(
        self, owner: str, repo: str, branch: str, installation_id: int
    ) -> dict[str, Any]:
        """Fetch the requested branch metadata, escaping the branch path segment."""
        response = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/branches/{quote(branch, safe='')}",
            installation_id=installation_id,
        )
        return _json_object(response)

    async def get_pull_request_diff(
        self, owner: str, repo: str, number: int, installation_id: int
    ) -> str:
        """Download the textual unified diff for a pull request."""
        payload = await self._download(
            f"/repos/{owner}/{repo}/pulls/{number}",
            installation_id=installation_id,
            accept="application/vnd.github.v3.diff",
            max_bytes=10 * 1024 * 1024,
        )
        return payload.decode("utf-8", errors="replace")

    async def get_pull_request_files(
        self, owner: str, repo: str, number: int, installation_id: int
    ) -> list[dict[str, Any]]:
        """Collect pull-request file metadata across GitHub API pages."""
        results: list[dict[str, Any]] = []
        for page in range(1, 31):
            response = await self._request(
                "GET",
                f"/repos/{owner}/{repo}/pulls/{number}/files",
                installation_id=installation_id,
                params={"per_page": 100, "page": page},
            )
            batch = _json_array(response)
            results.extend(batch)
            if len(batch) < 100:
                return results
        raise GitHubAPIError(422, "Pull request contains more than 3000 changed files")

    async def get_commit_comparison(
        self, owner: str, repo: str, base: str, head: str, installation_id: int
    ) -> dict[str, Any]:
        """Fetch GitHub comparison metadata between base and head revisions."""
        path = f"/repos/{owner}/{repo}/compare/{base}...{head}"
        combined: dict[str, Any] | None = None
        files: list[dict[str, Any]] = []
        for page in range(1, 4):
            response = await self._request(
                "GET",
                path,
                installation_id=installation_id,
                params={"per_page": 100, "page": page},
            )
            result = _json_object(response)
            if combined is None:
                combined = result
            page_files = result.get("files") or []
            if not isinstance(page_files, list) or not all(
                isinstance(item, dict) for item in page_files
            ):
                raise GitHubAPIError(502, "GitHub returned malformed comparison files")
            files.extend(page_files)
            if len(result.get("files") or []) < 100:
                break
        assert combined is not None
        combined["files"] = files
        return combined

    async def get_repository_installation(self, owner: str, repo: str) -> int:
        """Resolve the GitHub App installation ID for a repository."""
        response = await self.http.get(
            f"/repos/{owner}/{repo}/installation",
            headers={"Authorization": f"Bearer {self.authenticator.app_jwt()}"},
        )
        if response.is_error:
            raise GitHubAPIError(response.status_code, "repository installation lookup failed")
        value = _json_object(response).get("id")
        if not isinstance(value, (str, int)):
            raise GitHubAPIError(502, "GitHub returned an invalid installation ID")
        return int(value)

    async def get_repository_archive(
        self, owner: str, repo: str, ref: str, installation_id: int
    ) -> bytes:
        """Download a repository archive for the requested revision."""
        return await self._download(
            f"/repos/{owner}/{repo}/zipball/{quote(ref, safe='')}",
            installation_id=installation_id,
            accept="application/vnd.github+json",
            max_bytes=100 * 1024 * 1024,
        )

    async def resolve_commit(self, owner: str, repo: str, ref: str, installation_id: int) -> str:
        """Resolve a branch, tag, or commit reference to a full commit SHA."""
        response = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/commits/{ref}",
            installation_id=installation_id,
        )
        sha = _json_object(response).get("sha")
        if not isinstance(sha, str):
            raise GitHubAPIError(502, "GitHub returned an invalid commit SHA")
        return sha

    async def get_repository_tree(
        self, owner: str, repo: str, commit_sha: str, installation_id: int
    ) -> list[dict[str, Any]]:
        """Fetch the recursive Git tree for a specific commit."""
        response = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/git/trees/{commit_sha}",
            installation_id=installation_id,
            params={"recursive": "1"},
        )
        if len(response.content) > 10 * 1024 * 1024:
            raise GitHubAPIError(413, "Repository tree exceeds the 10 MiB indexing limit")
        tree = _json_object(response).get("tree")
        if not isinstance(tree, list) or not all(isinstance(item, dict) for item in tree):
            raise GitHubAPIError(502, "GitHub returned an invalid repository tree")
        return cast(list[dict[str, Any]], tree)

    async def get_file_content(
        self, owner: str, repo: str, path: str, ref: str, installation_id: int
    ) -> bytes:
        """Fetch raw content for a repository path at a specific revision."""
        response = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{quote(path, safe='/')}",
            installation_id=installation_id,
            params={"ref": ref},
        )
        payload = _json_object(response)
        if payload.get("encoding") != "base64" or not isinstance(payload.get("content"), str):
            raise GitHubAPIError(422, "GitHub did not return an encoded file")
        return base64.b64decode(payload["content"], validate=True)

    async def create_pull_request_review(
        self,
        owner: str,
        repo: str,
        number: int,
        installation_id: int,
        *,
        body: str,
        event: str,
        comments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a pull-request review with an optional list of inline comments."""
        response = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{number}/reviews",
            installation_id=installation_id,
            json={"body": body, "event": event, "comments": comments},
        )
        return _json_object(response)

    async def list_pull_request_reviews(
        self, owner: str, repo: str, number: int, installation_id: int
    ) -> list[dict[str, Any]]:
        """List existing reviews to support idempotent publication."""
        response = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls/{number}/reviews",
            installation_id=installation_id,
            params={"per_page": 100},
        )
        return _json_array(response)
