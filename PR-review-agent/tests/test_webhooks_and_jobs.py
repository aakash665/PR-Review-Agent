"""Tests for webhook authentication and durable pull-request job lifecycle."""

import hashlib
import hmac
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.github_webhooks import router as webhook_router
from app.api.github_webhooks import verify_signature
from app.config import get_settings
from app.database.repositories import JobRepository
from app.database.session import Database


class FakeJobs:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed: list[dict[str, object]] = []

    def enqueue(self, **values: object) -> tuple[int, bool]:
        """Upsert repository and pull-request state, then enqueue the requested head commit."""
        self.calls.append(values)
        return 12, True

    def close_pull_request(self, **values: object) -> bool:
        """Mark a pull request closed and cancel its unfinished review jobs."""
        self.closed.append(values)
        return True


def _payload(action: str = "opened") -> bytes:
    return json.dumps(
        {
            "action": action,
            "repository": {
                "id": 123,
                "full_name": "octo/demo",
                "name": "demo",
                "default_branch": "main",
            },
            "pull_request": {
                "number": 7,
                "title": "Update service",
                "body": "Description",
                "user": {"login": "contributor", "id": 456},
                "base": {"ref": "main", "sha": "abcdef123", "repo": {"id": 123}},
                "head": {"ref": "feature", "sha": "123456789abcdef"},
            },
            "installation": {"id": 999, "other": "ignored"},
        }
    ).encode()


def _signature(body: bytes, secret: str = "test-secret") -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_webhook_signature_is_constant_time_hmac_contract() -> None:
    body = b'{"action":"opened"}'
    assert verify_signature(body, _signature(body), "test-secret")
    assert not verify_signature(body, "sha1=wrong", "test-secret")
    assert not verify_signature(body + b" ", _signature(body), "test-secret")
    assert not verify_signature(body, None, "test-secret")


def test_webhook_validates_then_enqueues_supported_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "test-secret")
    get_settings.cache_clear()
    application = FastAPI()
    application.state.jobs = FakeJobs()
    application.include_router(webhook_router)
    body = _payload()
    with TestClient(application) as client:
        invalid = client.post(
            "/webhooks/github",
            content=body,
            headers={"x-github-event": "pull_request", "x-hub-signature-256": "sha256=wrong"},
        )
        assert invalid.status_code == 401
        accepted = client.post(
            "/webhooks/github",
            content=body,
            headers={
                "x-github-event": "pull_request",
                "x-hub-signature-256": _signature(body),
            },
        )
        assert accepted.status_code == 202
        assert accepted.json() == {"accepted": True, "queued": True, "job_id": 12}
        assert application.state.jobs.calls[0]["installation_id"] == 999
        assert application.state.jobs.calls[0]["head_sha"] == "123456789abcdef"
        assert application.state.jobs.calls[0]["title"] == "Update service"
        assert application.state.jobs.calls[0]["author"] == "contributor"
        assert application.state.jobs.calls[0]["head_branch"] == "feature"
        ignored_body = _payload(action="closed")
        ignored = client.post(
            "/webhooks/github",
            content=ignored_body,
            headers={
                "x-github-event": "pull_request",
                "x-hub-signature-256": _signature(ignored_body),
            },
        )
        assert ignored.status_code == 202
        assert ignored.json() == {
            "accepted": True,
            "queued": False,
            "reason": "pull request closed",
        }
        assert len(application.state.jobs.calls) == 1
        assert application.state.jobs.closed == [{"github_repo_id": 123, "pr_number": 7}]
    get_settings.cache_clear()


def test_webhook_rejects_base_repository_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "test-secret")
    get_settings.cache_clear()
    application = FastAPI()
    application.state.jobs = FakeJobs()
    application.include_router(webhook_router)
    payload = json.loads(_payload())
    payload["pull_request"]["base"]["repo"]["id"] = 789
    body = json.dumps(payload).encode()
    with TestClient(application) as client:
        response = client.post(
            "/webhooks/github",
            content=body,
            headers={
                "x-github-event": "pull_request",
                "x-hub-signature-256": _signature(body),
            },
        )
    assert response.status_code == 400
    assert application.state.jobs.calls == []
    get_settings.cache_clear()


def test_job_creation_is_idempotent_and_claim_is_durable(tmp_path) -> None:
    database = Database(tmp_path / "jobs.sqlite3")
    database.initialize()
    jobs = JobRepository(database)
    parameters = {
        "github_repo_id": 321,
        "owner": "octo",
        "name": "demo",
        "default_branch": "main",
        "pr_number": 7,
        "head_sha": "head-sha-123",
        "installation_id": 999,
    }
    first_id, created = jobs.enqueue(**parameters)
    duplicate_id, duplicate_created = jobs.enqueue(**parameters)
    assert created is True
    assert duplicate_created is False
    assert first_id == duplicate_id
    claimed = jobs.claim_next(3)
    assert claimed is not None
    assert claimed["id"] == first_id
    assert jobs.claim_next(3) is None
    jobs.fail(first_id, "transient", retry=True)
    retried = jobs.claim_by_id(first_id, 3)
    assert retried is not None
    assert retried["attempts"] == 1
    jobs.fail(first_id, "exhausted", retry=False)
    retry_id, retry_created = jobs.enqueue(**parameters)
    assert retry_id == first_id
    assert retry_created is True
    assert jobs.claim_by_id(first_id, 3) is not None


def test_stale_worker_lease_is_requeued_or_failed_at_retry_limit(tmp_path) -> None:
    database = Database(tmp_path / "leases.sqlite3")
    database.initialize()
    jobs = JobRepository(database)
    job_id, _ = jobs.enqueue(
        github_repo_id=321,
        owner="octo",
        name="demo",
        default_branch="main",
        pr_number=7,
        head_sha="stale-sha",
        installation_id=999,
    )
    assert jobs.claim_by_id(job_id, 3) is not None
    with database.connect() as connection:
        connection.execute(
            "UPDATE review_jobs SET started_at='2000-01-01 00:00:00' WHERE id=?",
            (job_id,),
        )
    assert jobs.requeue_stale(max_attempts=3, stale_seconds=60) == 1
    assert jobs.claim_by_id(job_id, 3) is not None
    with database.connect() as connection:
        connection.execute(
            "UPDATE review_jobs SET attempts=3, started_at='2000-01-01 00:00:00' WHERE id=?",
            (job_id,),
        )
    assert jobs.requeue_stale(max_attempts=3, stale_seconds=60) == 1
    assert jobs.claim_by_id(job_id, 3) is None


def test_closing_pr_cancels_job_and_reopening_requeues_same_commit(tmp_path) -> None:
    database = Database(tmp_path / "closed-pr.sqlite3")
    database.initialize()
    jobs = JobRepository(database)
    parameters = {
        "github_repo_id": 321,
        "owner": "octo",
        "name": "demo",
        "default_branch": "main",
        "pr_number": 7,
        "head_sha": "head-sha-123",
        "installation_id": 999,
    }
    job_id, created = jobs.enqueue(**parameters)
    assert created
    assert jobs.close_pull_request(github_repo_id=321, pr_number=7)
    assert jobs.claim_next(3) is None
    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT status FROM pull_requests WHERE github_pr_number=7"
            ).fetchone()["status"]
            == "closed"
        )
        assert (
            connection.execute("SELECT status FROM review_jobs WHERE id=?", (job_id,)).fetchone()[
                "status"
            ]
            == "cancelled"
        )
    reopened_id, requeued = jobs.enqueue(**parameters)
    assert reopened_id == job_id
    assert requeued
    assert jobs.claim_by_id(job_id, 3) is not None
