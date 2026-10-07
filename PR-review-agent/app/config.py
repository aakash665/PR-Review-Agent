"""Validated application settings loaded from environment variables and .env files."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime configuration for GitHub, LLM, storage, and review limits."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    database_path: Path = Path("data/reviews.sqlite3")
    github_app_id: str | None = None
    github_private_key_path: Path | None = None
    github_webhook_secret: SecretStr | None = None
    dashboard_api_key: SecretStr | None = None
    github_api_url: str = "https://api.github.com"
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai"
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "GitHub PR Review Agent"
    llm_model: str = "openai/gpt-4o-mini"
    decisions_model: str = "openai/gpt-6-luna-decisions"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "repository_chunks"
    max_context_tokens: int = Field(default=12000, gt=1500)
    top_k: int = Field(default=8, gt=0, le=50)
    confidence_threshold: float = Field(default=0.80, ge=0, le=1)
    summary_confidence_threshold: float = Field(default=0.60, ge=0, le=1)
    verification_threshold: float = Field(default=0.80, ge=0, le=1)
    worker_poll_seconds: float = Field(default=2, gt=0)
    max_retries: int = Field(default=3, ge=0)
    cost_per_million_tokens: float | None = Field(default=None, ge=0)
    index_snapshot_retention: int = Field(default=20, ge=1)
    log_level: str = "INFO"

    @field_validator(
        "github_app_id",
        "github_private_key_path",
        "github_webhook_secret",
        "dashboard_api_key",
        "openrouter_api_key",
        "cost_per_million_tokens",
        mode="before",
    )
    @classmethod
    def empty_credentials_are_unset(cls, value: object) -> object:
        """Normalize empty optional credential strings to unset values."""
        return None if value == "" else value

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Settings":
        """Require summary confidence to be no stricter than finding confidence."""
        if self.summary_confidence_threshold > self.confidence_threshold:
            raise ValueError("summary confidence threshold must not exceed inline threshold")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached, environment-backed application settings."""
    return Settings()
