"""Polling worker entry point with durable job leases and heartbeats."""

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress

from app.config import get_settings
from app.database.repositories import JobRepository
from app.database.session import Database
from app.workflow.review_processor import ReviewProcessor

logger = logging.getLogger(__name__)
HEARTBEAT_SECONDS = 300


async def run_worker(
    *,
    once: bool = False,
    processor_factory: Callable[[], ReviewProcessor] | None = None,
) -> None:
    """Poll durable jobs and process them until stopped or run-once work is complete."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    database = Database(settings.database_path)
    database.initialize()
    jobs = JobRepository(database)
    if processor_factory is None:

        def create_processor() -> ReviewProcessor:
            """Construct a review processor with shared database and GitHub clients."""
            return ReviewProcessor(settings, database)

        processor_factory = create_processor
    while True:
        jobs.requeue_stale(max_attempts=max(1, settings.max_retries))
        job = jobs.claim_next(max(1, settings.max_retries))
        if job is None:
            if once:
                return
            await asyncio.sleep(settings.worker_poll_seconds)
            continue
        processor: ReviewProcessor | None = None
        heartbeat = asyncio.create_task(_heartbeat(jobs, int(job["id"])))
        try:
            processor = processor_factory()
            await processor.process(job)
        except Exception as error:
            attempt = int(job["attempts"]) + 1
            retry = attempt < max(1, settings.max_retries)
            logger.exception(
                "Review job failed review_id=%s attempt=%s retry=%s",
                job["id"],
                attempt,
                retry,
            )
            jobs.fail(int(job["id"]), f"{type(error).__name__}: {error}", retry=retry)
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            if processor is not None:
                await processor.github.close()
        if once:
            return


async def _run() -> None:
    """Execute the worker polling loop while recovering stale job leases."""
    await run_worker()


async def _heartbeat(jobs: JobRepository, job_id: int) -> None:
    """Refresh a job lease periodically until processing completes."""
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        jobs.heartbeat(job_id)


if __name__ == "__main__":
    asyncio.run(_run())
