"""Transactional persistence for repositories, review jobs, findings, and metrics."""

import json
import sqlite3
from typing import cast

from app.database.session import Database
from app.models.review import ReviewFinding


class JobRepository:
    """Persist and coordinate idempotent review jobs and repository snapshots."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def enqueue(
        self,
        *,
        github_repo_id: int,
        owner: str,
        name: str,
        default_branch: str,
        pr_number: int,
        head_sha: str,
        installation_id: int | None = None,
    ) -> tuple[int, bool]:
        """Upsert repository and pull-request state, then enqueue the requested head commit."""
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO repositories(
                   github_repo_id, owner, name, default_branch, installation_id
                   ) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(github_repo_id) DO UPDATE SET
                   owner=excluded.owner, name=excluded.name, default_branch=excluded.default_branch,
                   installation_id=COALESCE(excluded.installation_id, repositories.installation_id),
                   updated_at=CURRENT_TIMESTAMP""",
                (github_repo_id, owner, name, default_branch, installation_id),
            )
            repository = connection.execute(
                "SELECT id FROM repositories WHERE github_repo_id = ?", (github_repo_id,)
            ).fetchone()
            assert repository is not None
            connection.execute(
                """INSERT INTO pull_requests(repository_id, github_pr_number, head_sha)
                   VALUES (?, ?, ?)
                   ON CONFLICT(repository_id, github_pr_number) DO UPDATE SET
                   head_sha=excluded.head_sha, updated_at=CURRENT_TIMESTAMP""",
                (repository["id"], pr_number, head_sha),
            )
            pull_request = connection.execute(
                "SELECT id FROM pull_requests WHERE repository_id = ? AND github_pr_number = ?",
                (repository["id"], pr_number),
            ).fetchone()
            assert pull_request is not None
            cursor = connection.execute(
                """INSERT OR IGNORE INTO review_jobs(pull_request_id, commit_sha)
                   VALUES (?, ?)""",
                (pull_request["id"], head_sha),
            )
            job = connection.execute(
                "SELECT id, status FROM review_jobs WHERE pull_request_id = ? AND commit_sha = ?",
                (pull_request["id"], head_sha),
            ).fetchone()
            assert job is not None
            created = cursor.rowcount == 1
            if not created and job["status"] in {"failed", "cancelled"}:
                connection.execute(
                    """UPDATE review_jobs SET status='queued', attempts=0, started_at=NULL,
                       completed_at=NULL, error=NULL WHERE id=?""",
                    (job["id"],),
                )
                created = True
            if created:
                connection.execute(
                    "UPDATE pull_requests SET status='queued' WHERE id=?",
                    (pull_request["id"],),
                )
            else:
                connection.execute(
                    "UPDATE pull_requests SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (job["status"], pull_request["id"]),
                )
            return int(job["id"]), created

    def close_pull_request(self, *, github_repo_id: int, pr_number: int) -> bool:
        """Mark a pull request closed and cancel its unfinished review jobs."""
        with self.database.connect() as connection:
            cursor = connection.execute(
                """UPDATE pull_requests SET status='closed', updated_at=CURRENT_TIMESTAMP
                   WHERE github_pr_number=? AND repository_id=(
                       SELECT id FROM repositories WHERE github_repo_id=?
                   )""",
                (pr_number, github_repo_id),
            )
            if cursor.rowcount == 0:
                return False
            connection.execute(
                """UPDATE review_jobs SET status='cancelled',
                   completed_at=CURRENT_TIMESTAMP, error='Pull request closed'
                   WHERE pull_request_id=(SELECT id FROM pull_requests
                       WHERE github_pr_number=? AND repository_id=(
                           SELECT id FROM repositories WHERE github_repo_id=?
                       ))
                   AND status IN ('queued', 'running')""",
                (pr_number, github_repo_id),
            )
            return True

    def cancel(self, job_id: int, reason: str) -> None:
        """Mark an active or pending job as cancelled with the supplied reason."""
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE review_jobs SET status='cancelled', error=?,
                   completed_at=CURRENT_TIMESTAMP
                   WHERE id=? AND status IN ('queued', 'running')""",
                (reason[:2000], job_id),
            )

    def upsert_repository(
        self,
        *,
        github_repo_id: int,
        owner: str,
        name: str,
        default_branch: str,
        installation_id: int | None,
    ) -> None:
        """Insert or refresh repository metadata and its installation association."""
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO repositories(
                   github_repo_id, owner, name, default_branch, installation_id
                   ) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(github_repo_id) DO UPDATE SET
                   owner=excluded.owner, name=excluded.name, default_branch=excluded.default_branch,
                   installation_id=COALESCE(excluded.installation_id, repositories.installation_id),
                   updated_at=CURRENT_TIMESTAMP""",
                (github_repo_id, owner, name, default_branch, installation_id),
            )

    def claim_next(self, max_attempts: int) -> sqlite3.Row | None:
        """Atomically lease the next eligible queued review job."""
        return self._claim(max_attempts)

    def claim_by_id(self, job_id: int, max_attempts: int) -> sqlite3.Row | None:
        """Atomically lease a specific eligible review job by its ID."""
        return self._claim(max_attempts, job_id=job_id)

    def heartbeat(self, job_id: int) -> None:
        """Refresh the lease timestamp for a job currently processed by a worker."""
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE review_jobs SET started_at=CURRENT_TIMESTAMP
                   WHERE id=? AND status='running'""",
                (job_id,),
            )

    def requeue_stale(self, max_attempts: int, stale_seconds: int = 1800) -> int:
        """Recover expired worker leases, retrying jobs within the attempt limit."""
        with self.database.connect() as connection:
            stale = connection.execute(
                """SELECT j.pull_request_id, j.commit_sha, j.attempts, p.head_sha
                   FROM review_jobs j JOIN pull_requests p ON p.id=j.pull_request_id
                   WHERE j.status='running' AND j.started_at < datetime('now', ?)""",
                (f"-{stale_seconds} seconds",),
            ).fetchall()
            cursor = connection.execute(
                """UPDATE review_jobs
                   SET status=CASE WHEN attempts < ? THEN 'queued' ELSE 'failed' END,
                   completed_at=CASE WHEN attempts < ? THEN NULL ELSE CURRENT_TIMESTAMP END,
                   error=CASE WHEN attempts < ? THEN 'Recovered stale worker lease'
                   ELSE 'Worker lease expired after retry limit' END
                   WHERE status='running'
                   AND started_at < datetime('now', ?)""",
                (
                    max_attempts,
                    max_attempts,
                    max_attempts,
                    f"-{stale_seconds} seconds",
                ),
            )
            for row in stale:
                if row["head_sha"] == row["commit_sha"]:
                    status = "queued" if int(row["attempts"]) < max_attempts else "failed"
                    connection.execute(
                        """UPDATE pull_requests SET status=?, updated_at=CURRENT_TIMESTAMP
                           WHERE id=?""",
                        (status, row["pull_request_id"]),
                    )
            return cursor.rowcount

    def _claim(self, max_attempts: int, *, job_id: int | None = None) -> sqlite3.Row | None:
        """Lease a queued or retryable job atomically, optionally constrained by ID."""
        with self.database.connect() as connection:
            job_filter = "AND j.id=?" if job_id is not None else ""
            params: tuple[object, ...] = (
                (max_attempts, job_id) if job_id is not None else (max_attempts,)
            )
            row = cast(
                sqlite3.Row | None,
                connection.execute(
                    """SELECT j.id, j.pull_request_id, j.commit_sha, j.attempts,
                          p.github_pr_number, r.github_repo_id, r.owner, r.name, r.installation_id
                   FROM review_jobs j
                   JOIN pull_requests p ON p.id=j.pull_request_id
                   JOIN repositories r ON r.id=p.repository_id
                   WHERE j.status='queued' AND j.attempts < ? """
                    + job_filter
                    + """
                   ORDER BY j.created_at LIMIT 1""",
                    params,
                ).fetchone(),
            )
            if row is None:
                return None
            connection.execute(
                """UPDATE review_jobs SET status='running', attempts=attempts+1,
                   started_at=CURRENT_TIMESTAMP, error=NULL WHERE id=?""",
                (row["id"],),
            )
            return row

    def complete(
        self,
        job_id: int,
        findings: list[ReviewFinding],
        metrics: dict[str, float | int | None] | None = None,
    ) -> None:
        """Persist successful job findings and optional run metrics atomically."""
        with self.database.connect() as connection:
            cursor = connection.execute(
                """UPDATE review_jobs SET status='completed',
                   completed_at=CURRENT_TIMESTAMP WHERE id=? AND status='running'""",
                (job_id,),
            )
            if cursor.rowcount == 0:
                return
            connection.execute("DELETE FROM findings WHERE review_job_id = ?", (job_id,))
            for finding in findings:
                connection.execute(
                    """INSERT INTO findings(
                       review_job_id, file_path, line_start, line_end, category, severity,
                       confidence, title, explanation, suggestion, evidence, status
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        job_id,
                        finding.file_path,
                        finding.line_start,
                        finding.line_end,
                        finding.category,
                        finding.severity,
                        finding.confidence,
                        finding.title,
                        finding.explanation,
                        finding.suggestion,
                        json.dumps(finding.evidence),
                        "validated" if finding.verified else "unverified",
                    ),
                )
            connection.execute(
                """UPDATE pull_requests SET status='completed', updated_at=CURRENT_TIMESTAMP
                   WHERE id=(SELECT pull_request_id FROM review_jobs WHERE id=?)
                   AND head_sha=(SELECT commit_sha FROM review_jobs WHERE id=?)
                   AND status != 'closed'""",
                (job_id, job_id),
            )
            if metrics is not None:
                self._insert_metrics(connection, job_id, metrics)

    @staticmethod
    def _insert_metrics(
        connection: sqlite3.Connection,
        job_id: int,
        metrics: dict[str, float | int | None],
    ) -> None:
        """Insert normalized metric values using the active SQLite transaction."""
        connection.execute(
            """INSERT OR REPLACE INTO review_metrics(
               review_job_id, retrieval_latency, llm_latency, embedding_latency,
               static_analysis_latency, number_of_chunks, number_of_findings,
               number_of_rejected_findings, tokens_used, estimated_cost_usd
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                metrics["retrieval_latency"],
                metrics["llm_latency"],
                metrics["embedding_latency"],
                metrics["static_analysis_latency"],
                metrics["number_of_chunks"],
                metrics["number_of_findings"],
                metrics["number_of_rejected_findings"],
                metrics["tokens_used"],
                metrics["estimated_cost_usd"],
            ),
        )

    def record_metrics(self, job_id: int, metrics: dict[str, float | int | None]) -> None:
        """Persist usage and timing metrics for a review job."""
        with self.database.connect() as connection:
            self._insert_metrics(connection, job_id, metrics)

    def fail(self, job_id: int, error: str, *, retry: bool) -> None:
        """Record a job failure and either schedule a retry or mark it terminal."""
        with self.database.connect() as connection:
            cursor = connection.execute(
                """UPDATE review_jobs SET status=?, error=?, completed_at=CASE
                   WHEN ? THEN NULL ELSE CURRENT_TIMESTAMP END WHERE id=? AND status='running'""",
                ("queued" if retry else "failed", error[:2000], retry, job_id),
            )
            if cursor.rowcount:
                connection.execute(
                    """UPDATE pull_requests SET status=?, updated_at=CURRENT_TIMESTAMP
                       WHERE id=(SELECT pull_request_id FROM review_jobs WHERE id=?)
                       AND head_sha=(SELECT commit_sha FROM review_jobs WHERE id=?)
                       AND status != 'closed'""",
                    ("queued" if retry else "failed", job_id, job_id),
                )

    def metrics(self) -> dict[str, int | float]:
        """Expose aggregate review-job metrics from the durable job repository."""
        with self.database.connect() as connection:
            jobs = connection.execute(
                "SELECT status, COUNT(*) AS count FROM review_jobs GROUP BY status"
            ).fetchall()
            metric_rows = connection.execute("SELECT * FROM review_metrics").fetchall()
            findings = connection.execute("SELECT COUNT(*) FROM findings").fetchone()
        metrics = {
            key: sum(float(row[key]) for row in metric_rows)
            for key in ("tokens_used", "number_of_findings", "number_of_rejected_findings")
        }
        return {
            **{row["status"]: int(row["count"]) for row in jobs},
            "findings": int(findings[0]),
            **metrics,
        }

    def repository_state(self, github_repo_id: int) -> sqlite3.Row | None:
        """Return stored indexing and repository metadata for a GitHub repository."""
        with self.database.connect() as connection:
            return cast(
                sqlite3.Row | None,
                connection.execute(
                    "SELECT id, last_indexed_sha FROM repositories WHERE github_repo_id=?",
                    (github_repo_id,),
                ).fetchone(),
            )

    def record_indexed_sha(self, github_repo_id: int, commit_sha: str, retention: int) -> list[str]:
        """Record a completed commit snapshot and prune snapshots beyond retention."""
        with self.database.connect() as connection:
            repository = connection.execute(
                "SELECT id FROM repositories WHERE github_repo_id=?", (github_repo_id,)
            ).fetchone()
            if repository is None:
                raise ValueError(f"Repository {github_repo_id} is not registered")
            repository_id = int(repository["id"])
            connection.execute(
                """INSERT INTO repository_indexes(repository_id, commit_sha, indexed_at)
                   VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                   ON CONFLICT(repository_id, commit_sha) DO UPDATE
                   SET indexed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
                (repository_id, commit_sha),
            )
            connection.execute(
                """UPDATE repositories SET last_indexed_sha=?, updated_at=CURRENT_TIMESTAMP
                   WHERE id=?""",
                (commit_sha, repository_id),
            )
            old = connection.execute(
                """SELECT commit_sha FROM repository_indexes WHERE repository_id=?
                   ORDER BY indexed_at DESC LIMIT -1 OFFSET ?""",
                (repository_id, retention),
            ).fetchall()
            old_shas = [str(row["commit_sha"]) for row in old]
            if old_shas:
                connection.executemany(
                    "DELETE FROM repository_indexes WHERE repository_id=? AND commit_sha=?",
                    [(repository_id, sha) for sha in old_shas],
                )
            return old_shas

    def review_status(self, job_id: int) -> dict[str, object] | None:
        """Return a serialized job status with findings and available metrics."""
        with self.database.connect() as connection:
            job = connection.execute(
                """SELECT j.id, j.commit_sha, j.status, j.attempts, j.error, j.created_at,
                          j.completed_at, p.github_pr_number, r.owner, r.name
                   FROM review_jobs j JOIN pull_requests p ON p.id=j.pull_request_id
                   JOIN repositories r ON r.id=p.repository_id WHERE j.id=?""",
                (job_id,),
            ).fetchone()
            if job is None:
                return None
            findings = connection.execute(
                """SELECT file_path, line_start, line_end, category, severity, confidence,
                          title, explanation, suggestion, evidence, status
                   FROM findings WHERE review_job_id=? ORDER BY severity, confidence DESC""",
                (job_id,),
            ).fetchall()
            metrics = connection.execute(
                "SELECT * FROM review_metrics WHERE review_job_id=?", (job_id,)
            ).fetchone()
            return {
                "job": dict(job),
                "metrics": dict(metrics) if metrics else None,
                "findings": [
                    {**dict(finding), "evidence": json.loads(finding["evidence"])}
                    for finding in findings
                ],
            }
