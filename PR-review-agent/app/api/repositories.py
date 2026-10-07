"""Repository registration and manual review endpoints for the web dashboard."""

import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.auth import require_dashboard_access
from app.config import Settings
from app.database.repositories import JobRepository
from app.github.client import GitHubAPIError, GitHubClient

router = APIRouter(prefix="/repositories", tags=["repositories"])
REPOSITORY_NAME = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class RepositoryRegistration(BaseModel):
    """Repository full name submitted from the dashboard."""

    full_name: str = Field(min_length=3, max_length=200, pattern=REPOSITORY_NAME.pattern)


class ManualReviewRequest(BaseModel):
    """Pull-request number submitted to start a dashboard review."""

    pr_number: int = Field(gt=0)


@router.get("")
async def list_repositories(
    request: Request,
    _: None = Depends(require_dashboard_access),
) -> list[dict[str, object]]:
    """Return registered repositories and their aggregate review counts."""
    jobs: JobRepository = request.app.state.jobs
    return jobs.list_repositories()


@router.post("", status_code=status.HTTP_201_CREATED)
async def register_repository(
    registration: RepositoryRegistration,
    request: Request,
    _: None = Depends(require_dashboard_access),
) -> dict[str, object]:
    """Validate GitHub App access and persist a repository for webhook reviews."""
    settings: Settings = request.app.state.settings
    if settings.github_app_id is None or settings.github_private_key_path is None:
        raise HTTPException(status_code=503, detail="GitHub App credentials are not configured")
    owner, name = registration.full_name.split("/", maxsplit=1)
    github = GitHubClient(settings)
    try:
        installation_id = await github.get_repository_installation(owner, name)
        repository = await github.get_repository(owner, name, installation_id)
        actual_name = str(repository.get("full_name", "")).casefold()
        if actual_name != registration.full_name.casefold():
            raise HTTPException(status_code=502, detail="GitHub repository identity did not match")
        request.app.state.jobs.upsert_repository(
            github_repo_id=int(repository["id"]),
            owner=str(repository["owner"]["login"]),
            name=str(repository["name"]),
            default_branch=str(repository["default_branch"]),
            installation_id=installation_id,
        )
        return {
            "repository": f"{repository['owner']['login']}/{repository['name']}",
            "github_repo_id": int(repository["id"]),
            "default_branch": str(repository["default_branch"]),
        }
    except GitHubAPIError as error:
        if error.status_code in {403, 404}:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Repository is unavailable to this GitHub App; install the App "
                    "with repository access"
                ),
            ) from error
        raise HTTPException(status_code=502, detail="GitHub repository lookup failed") from error
    finally:
        await github.close()


@router.post("/{repository_id}/reviews", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_manual_review(
    repository_id: int,
    body: ManualReviewRequest,
    request: Request,
    _: None = Depends(require_dashboard_access),
) -> dict[str, int | bool]:
    """Fetch the authoritative pull-request head and queue it for the worker."""
    repository = request.app.state.jobs.repository(repository_id)
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository is not registered")
    installation_id = repository["installation_id"]
    if installation_id is None:
        raise HTTPException(status_code=409, detail="Repository has no GitHub App installation")
    github = GitHubClient(request.app.state.settings)
    try:
        pr = await github.get_pull_request(
            str(repository["owner"]),
            str(repository["name"]),
            body.pr_number,
            int(installation_id),
        )
        base_repository_id = int(pr.get("base", {}).get("repo", {}).get("id", 0))
        if base_repository_id != int(repository["github_repo_id"]):
            raise HTTPException(
                status_code=400, detail="Pull request belongs to another repository"
            )
        if pr.get("state") != "open":
            raise HTTPException(status_code=409, detail="Only open pull requests can be reviewed")
        job_id, created = request.app.state.jobs.enqueue(
            github_repo_id=int(repository["github_repo_id"]),
            owner=str(repository["owner"]),
            name=str(repository["name"]),
            default_branch=str(repository["default_branch"]),
            pr_number=body.pr_number,
            head_sha=str(pr["head"]["sha"]),
            installation_id=int(installation_id),
            title=str(pr.get("title") or ""),
            author=str(pr.get("user", {}).get("login") or ""),
            head_branch=str(pr["head"]["ref"]),
        )
        return {"job_id": job_id, "queued": created}
    except GitHubAPIError as error:
        if error.status_code == 404:
            raise HTTPException(status_code=404, detail="Pull request was not found") from error
        raise HTTPException(status_code=502, detail="GitHub pull-request lookup failed") from error
    finally:
        await github.close()
