"""HTTP endpoints for listing repositories known to the service."""

from fastapi import APIRouter, Request

from app.database.session import Database

router = APIRouter(prefix="/repositories", tags=["repositories"])


@router.get("")
async def list_repositories(request: Request) -> list[dict[str, object]]:
    """Return known repository metadata from the application database."""
    database: Database = request.app.state.database
    with database.connect() as connection:
        rows = connection.execute(
            """SELECT r.id, r.github_repo_id, r.owner, r.name, r.default_branch,
                      r.last_indexed_sha, r.created_at, r.updated_at,
                      COUNT(DISTINCT p.id) AS pull_requests,
                      COUNT(DISTINCT j.id) AS reviews
               FROM repositories r LEFT JOIN pull_requests p ON p.repository_id=r.id
               LEFT JOIN review_jobs j ON j.pull_request_id=p.id
               GROUP BY r.id ORDER BY r.updated_at DESC"""
        ).fetchall()
    return [dict(row) for row in rows]
