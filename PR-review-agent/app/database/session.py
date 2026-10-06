"""SQLite connection lifecycle and schema initialization."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS repositories (
    id INTEGER PRIMARY KEY,
    github_repo_id INTEGER NOT NULL UNIQUE,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    default_branch TEXT NOT NULL,
    installation_id INTEGER,
    last_indexed_sha TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS pull_requests (
    id INTEGER PRIMARY KEY,
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    github_pr_number INTEGER NOT NULL,
    head_sha TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(repository_id, github_pr_number)
);
CREATE TABLE IF NOT EXISTS review_jobs (
    id INTEGER PRIMARY KEY,
    pull_request_id INTEGER NOT NULL REFERENCES pull_requests(id) ON DELETE CASCADE,
    commit_sha TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    completed_at TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pull_request_id, commit_sha)
);
CREATE INDEX IF NOT EXISTS review_jobs_status_created ON review_jobs(status, created_at);
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY,
    review_job_id INTEGER NOT NULL REFERENCES review_jobs(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    title TEXT NOT NULL,
    explanation TEXT NOT NULL,
    suggestion TEXT,
    evidence TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'validated'
);
CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS review_metrics (
    review_job_id INTEGER PRIMARY KEY REFERENCES review_jobs(id) ON DELETE CASCADE,
    retrieval_latency REAL NOT NULL DEFAULT 0,
    llm_latency REAL NOT NULL DEFAULT 0,
    embedding_latency REAL NOT NULL DEFAULT 0,
    static_analysis_latency REAL NOT NULL DEFAULT 0,
    number_of_chunks INTEGER NOT NULL DEFAULT 0,
    number_of_findings INTEGER NOT NULL DEFAULT 0,
    number_of_rejected_findings INTEGER NOT NULL DEFAULT 0,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL
);
CREATE TABLE IF NOT EXISTS repository_indexes (
    repository_id INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    commit_sha TEXT NOT NULL,
    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(repository_id, commit_sha)
);
CREATE INDEX IF NOT EXISTS repository_indexes_latest
ON repository_indexes(repository_id, indexed_at DESC);
"""


class Database:
    """Manage SQLite database paths, schema setup, and transactional connections."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        """Create the database directory and initialize the durable job schema."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(repositories)").fetchall()
            }
            if "installation_id" not in columns:
                connection.execute("ALTER TABLE repositories ADD COLUMN installation_id INTEGER")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a configured SQLite connection as a transaction-scoped context manager."""
        connection = sqlite3.connect(self.path, timeout=30, isolation_level="IMMEDIATE")
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
