"""FastAPI application construction, lifecycle management, and health endpoints."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.github_webhooks import router as webhook_router
from app.api.repositories import router as repositories_router
from app.api.reviews import router as reviews_router
from app.config import get_settings
from app.database.repositories import JobRepository
from app.database.session import Database


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize shared application services and release their resources on shutdown."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    database = Database(settings.database_path)
    database.initialize()
    application.state.database = database
    application.state.jobs = JobRepository(database)
    yield


app = FastAPI(
    title="GitHub PR Review Agent",
    description="Evidence-grounded pull request analysis using repository retrieval.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(webhook_router)
app.include_router(repositories_router)
app.include_router(reviews_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Report whether the API process is responsive."""
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> dict[str, int | float]:
    """Expose aggregate review-job metrics from the durable job repository."""
    jobs: JobRepository | None = getattr(app.state, "jobs", None)
    return jobs.metrics() if jobs else {}
