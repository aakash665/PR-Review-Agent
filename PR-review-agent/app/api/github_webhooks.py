"""GitHub webhook schemas, signature validation, and pull-request job intake."""

import hashlib
import hmac
import json
import logging

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_WEBHOOK_BYTES = 5 * 1024 * 1024
REVIEW_ACTIONS = {"opened", "synchronize", "reopened"}
SUPPORTED_ACTIONS = REVIEW_ACTIONS | {"closed"}


class RepositoryPayload(BaseModel):
    """Validated repository identity and default-branch fields from a webhook."""

    model_config = ConfigDict(extra="ignore")

    id: int = Field(gt=0)
    full_name: str = Field(pattern=r"^[^/]+/[^/]+$")
    name: str
    default_branch: str


class UserPayload(BaseModel):
    """Validated GitHub account identity included in a webhook payload."""

    login: str


class RepositoryRefPayload(BaseModel):
    """Validated repository reference and stable GitHub repository ID."""

    id: int = Field(gt=0)


class BranchRefPayload(BaseModel):
    """Validated branch or commit reference from a webhook payload."""

    ref: str
    sha: str = Field(min_length=7, max_length=64)
    repo: RepositoryRefPayload | None = None


class PullRequestPayload(BaseModel):
    """Validated pull-request fields needed to enqueue review work."""

    number: int = Field(gt=0)
    title: str
    body: str | None = None
    user: UserPayload
    base: BranchRefPayload
    head: BranchRefPayload


class InstallationPayload(BaseModel):
    """Validated GitHub App installation identifier from a webhook."""

    id: int = Field(gt=0)


class WebhookPayload(BaseModel):
    """Validated subset of GitHub webhook event fields consumed by the service."""

    model_config = ConfigDict(extra="ignore")

    action: str
    repository: RepositoryPayload
    pull_request: PullRequestPayload
    installation: InstallationPayload | None = None


def verify_signature(body: bytes, signature: str | None, secret: str) -> bool:
    """Verify the GitHub SHA-256 HMAC signature using constant-time comparison."""
    if not signature or not signature.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/webhooks/github", status_code=status.HTTP_202_ACCEPTED)
async def github_webhook(request: Request) -> dict[str, int | bool | str]:
    """Authenticate supported pull-request events and enqueue durable review jobs."""
    settings = get_settings()
    if settings.github_webhook_secret is None:
        raise HTTPException(status_code=503, detail="GitHub webhook secret is not configured")
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_WEBHOOK_BYTES:
                raise HTTPException(status_code=413, detail="Webhook payload is too large")
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid content length") from error
    body_chunks = []
    body_size = 0
    async for chunk in request.stream():
        body_size += len(chunk)
        if body_size > MAX_WEBHOOK_BYTES:
            raise HTTPException(status_code=413, detail="Webhook payload is too large")
        body_chunks.append(chunk)
    body = b"".join(body_chunks)
    signature = request.headers.get("x-hub-signature-256")
    if not verify_signature(body, signature, settings.github_webhook_secret.get_secret_value()):
        raise HTTPException(status_code=401, detail="Invalid GitHub webhook signature")
    if request.headers.get("x-github-event") != "pull_request":
        return {"accepted": True, "queued": False, "reason": "unsupported event"}
    try:
        payload = WebhookPayload.model_validate_json(body)
    except (ValidationError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=400, detail="Malformed pull request webhook payload"
        ) from error
    if payload.action not in SUPPORTED_ACTIONS:
        return {"accepted": True, "queued": False, "reason": "unsupported pull request action"}
    try:
        repository = payload.repository
        if (
            payload.pull_request.base.repo is not None
            and payload.pull_request.base.repo.id != repository.id
        ):
            raise ValueError("Pull request base repository does not match webhook repository")
        owner, name = repository.full_name.split("/", maxsplit=1)
        if payload.action == "closed":
            request.app.state.jobs.close_pull_request(
                github_repo_id=repository.id,
                pr_number=payload.pull_request.number,
            )
            return {"accepted": True, "queued": False, "reason": "pull request closed"}
        job_id, created = request.app.state.jobs.enqueue(
            github_repo_id=repository.id,
            owner=owner,
            name=name,
            default_branch=repository.default_branch,
            pr_number=payload.pull_request.number,
            head_sha=payload.pull_request.head.sha,
            installation_id=(payload.installation.id if payload.installation is not None else None),
            title=payload.pull_request.title,
            author=payload.pull_request.user.login,
            head_branch=payload.pull_request.head.ref,
        )
    except (ValueError, KeyError) as error:
        raise HTTPException(
            status_code=400, detail="Invalid repository or pull request metadata"
        ) from error
    except Exception:
        logger.exception("Could not enqueue GitHub pull request job")
        raise HTTPException(status_code=503, detail="Could not persist review job") from None
    return {"accepted": True, "queued": created, "job_id": job_id}
