"""Public dashboard API, repository management, and review status tests."""

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.repositories import router as repositories_router
from app.api.reviews import router as reviews_router
from app.config import Settings
from app.database.repositories import JobRepository
from app.database.session import Database


class FakeGitHub:
    """Stub GitHub App lookups for repository registration and manual PR review."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.closed = False

    async def get_repository_installation(self, owner: str, repo: str) -> int:
        assert (owner, repo) == ("octo", "demo")
        return 991

    async def get_repository(self, owner: str, repo: str, installation_id: int) -> dict[str, Any]:
        assert (owner, repo, installation_id) == ("octo", "demo", 991)
        return {
            "id": 321,
            "full_name": "octo/demo",
            "owner": {"login": "octo"},
            "name": "demo",
            "default_branch": "main",
        }

    async def get_pull_request(
        self, owner: str, repo: str, number: int, installation_id: int
    ) -> dict[str, Any]:
        assert (owner, repo, number, installation_id) == ("octo", "demo", 7, 991)
        return {
            "state": "open",
            "title": "Improve service validation",
            "user": {"login": "contributor"},
            "base": {"repo": {"id": 321}},
            "head": {"sha": "head-sha-123", "ref": "feature"},
        }

    async def close(self) -> None:
        self.closed = True


def _application(tmp_path: Path) -> tuple[FastAPI, JobRepository]:
    database = Database(tmp_path / "dashboard.sqlite3")
    database.initialize()
    jobs = JobRepository(database)
    application = FastAPI()
    application.state.settings = Settings.model_construct(
        github_app_id="123",
        github_private_key_path=tmp_path / "github.pem",
    )
    application.state.jobs = jobs
    application.include_router(repositories_router)
    application.include_router(reviews_router)
    return application, jobs


def test_dashboard_apis_are_public_without_a_browser_key(tmp_path: Path) -> None:
    application, _ = _application(tmp_path)
    with TestClient(application) as client:
        assert client.get("/repositories").status_code == 200
        assert client.get("/repositories").json() == []
        assert client.get("/reviews").status_code == 200


def test_public_dashboard_registers_repository_and_queues_pr(tmp_path: Path, monkeypatch) -> None:
    import app.api.repositories as repositories_api

    monkeypatch.setattr(repositories_api, "GitHubClient", FakeGitHub)
    application, jobs = _application(tmp_path)
    with TestClient(application) as client:
        registered = client.post(
            "/repositories",
            json={"full_name": "octo/demo"},
        )
        assert registered.status_code == 201
        assert registered.json() == {
            "repository": "octo/demo",
            "github_repo_id": 321,
            "default_branch": "main",
        }
        repository_id = client.get("/repositories").json()[0]["id"]
        queued = client.post(
            f"/repositories/{repository_id}/reviews",
            json={"pr_number": 7},
        )
        assert queued.status_code == 202
        assert queued.json() == {"job_id": 1, "queued": True}
        reviews = client.get("/reviews").json()
        assert reviews["total"] == 1
        assert reviews["items"][0]["title"] == "Improve service validation"
        assert reviews["items"][0]["head_branch"] == "feature"
        detail = client.get("/reviews/1")
        assert detail.status_code == 200
        assert detail.json()["job"]["status"] == "queued"
    assert jobs.list_repositories()[0]["reviews"] == 1


def test_review_summary_is_persisted_for_dashboard(tmp_path: Path) -> None:
    database = Database(tmp_path / "summary.sqlite3")
    database.initialize()
    jobs = JobRepository(database)
    job_id, _ = jobs.enqueue(
        github_repo_id=321,
        owner="octo",
        name="demo",
        default_branch="main",
        pr_number=7,
        head_sha="head-sha-123",
        title="Improve service validation",
    )
    assert jobs.claim_by_id(job_id, 3) is not None
    jobs.complete(job_id, [], summary="## AI Code Review\nNo validated findings.")
    detail = jobs.review_status(job_id)
    assert detail is not None
    assert detail["job"]["summary"] == "## AI Code Review\nNo validated findings."
    assert detail["job"]["status"] == "completed"


def test_dashboard_static_assets_are_served(tmp_path: Path) -> None:
    from app.main import app

    with TestClient(app) as client:
        dashboard = client.get("/")
        javascript = client.get("/static/app.js")
        stylesheet = client.get("/static/styles.css")
        metrics = client.get("/metrics")
    assert dashboard.status_code == 200
    assert "Pull request reviews" in dashboard.text
    assert javascript.status_code == 200
    assert "reviewops_dashboard_key" not in javascript.text
    assert stylesheet.status_code == 200
    assert metrics.status_code == 200
