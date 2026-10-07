"""HTTP endpoints for listing and retrieving durable review-job results."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.auth import require_dashboard_access
from app.database.repositories import JobRepository

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("")
async def list_reviews(
    request: Request,
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _: None = Depends(require_dashboard_access),
) -> dict[str, object]:
    """Return recent jobs for dashboard polling with bounded pagination."""
    repository: JobRepository = request.app.state.jobs
    return repository.list_reviews(limit=limit, offset=offset)


@router.get("/{job_id}")
async def get_review(
    job_id: int,
    request: Request,
    _: None = Depends(require_dashboard_access),
) -> dict[str, object]:
    """Return the persisted status and findings for a review job, if it exists."""
    repository: JobRepository = request.app.state.jobs
    result = repository.review_status(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Review job not found")
    return result
