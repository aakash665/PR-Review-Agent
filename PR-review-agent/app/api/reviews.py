"""HTTP endpoint for retrieving durable review-job status and results."""

from fastapi import APIRouter, HTTPException, Request

from app.database.repositories import JobRepository

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("/{job_id}")
async def get_review(job_id: int, request: Request) -> dict[str, object]:
    """Return the persisted status and findings for a review job, if it exists."""
    repository: JobRepository = request.app.state.jobs
    result = repository.review_status(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Review job not found")
    return result
