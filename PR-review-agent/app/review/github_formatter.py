"""Conversion of validated findings into GitHub review comments and summaries."""

from app.models.review import ReviewFinding


def format_inline_comment(finding: ReviewFinding) -> str:
    """Render one finding as a concise GitHub Markdown inline comment."""
    title = finding.category.replace("_", " ").title()
    body = [
        f"### ⚠️ {title} · {finding.severity.title()}: {finding.title}",
        "",
        finding.explanation,
        "",
        "**Evidence:**",
        *[f"- {evidence}" for evidence in finding.evidence],
    ]
    if finding.suggestion:
        language = _language(finding.file_path)
        body.extend(["", "**Suggested fix:**", f"```{language}", finding.suggestion, "```"])
    body.extend(["", f"Confidence: {finding.confidence:.0%}"])
    return "\n".join(body)


def format_review_summary(
    *,
    files_analyzed: int,
    additions: int,
    deletions: int,
    findings: list[ReviewFinding],
    summary_only: list[ReviewFinding],
    discarded_count: int,
    rejected_count: int,
    llm_summary: str,
    positive_changes: list[str],
    review_confidence: float,
    unmapped_count: int = 0,
) -> str:
    """Build a Markdown review summary with severity counts and quality indicators."""
    all_reportable = sorted(
        [*findings, *summary_only],
        key=lambda item: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}[item.severity],
            -item.confidence,
        ),
    )
    counts = {
        severity: sum(item.severity == severity for item in all_reportable)
        for severity in ("critical", "high", "medium", "low")
    }
    lines = [
        "## AI Code Review",
        "",
        f"Files analyzed: {files_analyzed} · Lines changed: +{additions} / -{deletions}",
        "",
        "### Findings",
        "",
        f"🔴 Critical: {counts['critical']} · 🟠 High: {counts['high']} · "
        f"🟡 Medium: {counts['medium']} · 🔵 Low: {counts['low']}",
        "",
    ]
    if all_reportable:
        lines.append("### Key Issues")
        lines.extend(
            f"{index}. **{item.title}** (`{item.file_path}:{item.line_start}`) — "
            f"{item.explanation.splitlines()[0]}"
            for index, item in enumerate(all_reportable[:5], start=1)
        )
        if len(all_reportable) > 5:
            lines.append(f"- And {len(all_reportable) - 5} more finding(s).")
        lines.append("")
    else:
        lines.extend(["No actionable findings met the configured confidence threshold.", ""])
    if positive_changes:
        lines.extend(
            ["### Positive Changes", "", *[f"- {item}" for item in positive_changes[:5]], ""]
        )
    if llm_summary.strip():
        lines.extend(["### Summary", "", llm_summary.strip(), ""])
    lines.extend(
        [
            f"Review confidence: {review_confidence:.0%}",
            f"Verified and reported: {len(all_reportable)} · "
            f"Rejected by verifier: {rejected_count}",
        ]
    )
    if discarded_count:
        lines.append(f"Low-confidence findings discarded: {discarded_count}")
    if unmapped_count:
        lines.append(
            "Findings not eligible for inline comments (not on a valid "
            f"changed line): {unmapped_count}"
        )
    return "\n".join(lines)


def _language(path: str) -> str:
    """Map a source path to a GitHub inline-comment language label."""
    extension = path.rsplit(".", maxsplit=1)[-1].lower() if "." in path else ""
    return {
        "py": "python",
        "js": "javascript",
        "jsx": "jsx",
        "ts": "typescript",
        "tsx": "tsx",
        "java": "java",
        "go": "go",
        "rs": "rust",
    }.get(extension, "")
