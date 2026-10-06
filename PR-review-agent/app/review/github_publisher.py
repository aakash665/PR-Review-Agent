"""Idempotent publication of inline findings to GitHub pull requests."""

from app.analysis.diff_analyzer import DiffAnalysis
from app.github.client import GitHubClient
from app.models.review import ReviewFinding
from app.review.github_formatter import format_inline_comment


class GitHubReviewPublisher:
    """Publish validated findings once per reviewed pull-request head SHA."""

    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    async def publish(
        self,
        *,
        owner: str,
        repo: str,
        pr_number: int,
        head_sha: str,
        installation_id: int,
        summary: str,
        findings: list[ReviewFinding],
        diff: DiffAnalysis,
    ) -> int:
        """Create a GitHub review with eligible added-line comments; return its review ID."""
        marker = f"<!-- github-pr-review-agent:{head_sha} -->"
        existing = await self.client.list_pull_request_reviews(
            owner, repo, pr_number, installation_id
        )
        if any(marker in str(review.get("body") or "") for review in existing):
            return 0
        comments = [
            {
                "path": finding.file_path,
                "line": finding.line_start,
                "side": "RIGHT",
                "body": format_inline_comment(finding),
            }
            for finding in findings[:50]
            if diff.is_commentable(finding.file_path, finding.line_start)
            and any(
                item.change_type == "added" and item.line == finding.line_start
                for file in diff.files
                if file.file_path == finding.file_path
                for item in file.changed_lines
            )
        ]
        response = await self.client.create_pull_request_review(
            owner,
            repo,
            pr_number,
            installation_id,
            body=f"{summary[:60000]}\n\n{marker}",
            event="COMMENT",
            comments=comments,
        )
        return int(response.get("id", 0))
