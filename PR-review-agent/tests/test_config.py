"""Runtime configuration tests."""

from pathlib import Path

from app.config import Settings


def test_vercel_uses_writable_temporary_database_path(monkeypatch) -> None:
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("DATABASE_PATH", raising=False)

    settings = Settings(_env_file=None)

    assert settings.database_path == Path("/tmp/reviews.sqlite3")


def test_configured_database_path_takes_precedence_on_vercel(monkeypatch) -> None:
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("DATABASE_PATH", "/mnt/data/reviews.sqlite3")

    settings = Settings(_env_file=None)

    assert settings.database_path == Path("/mnt/data/reviews.sqlite3")
