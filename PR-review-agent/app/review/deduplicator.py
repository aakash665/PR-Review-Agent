"""Normalization and merging of overlapping review findings."""

import re
from difflib import SequenceMatcher

from app.models.review import ReviewFinding


def deduplicate_findings(findings: list[ReviewFinding]) -> list[ReviewFinding]:
    """Merge overlapping same-category findings while retaining strongest evidence."""
    ordered = sorted(findings, key=lambda item: (-item.confidence, item.file_path, item.line_start))
    kept: list[ReviewFinding] = []
    for finding in ordered:
        duplicate_index = next(
            (index for index, existing in enumerate(kept) if _is_duplicate(finding, existing)),
            None,
        )
        if duplicate_index is None:
            kept.append(finding)
            continue
        existing = kept[duplicate_index]
        strongest = finding if _strength(finding) > _strength(existing) else existing
        evidence = list(dict.fromkeys([*existing.evidence, *finding.evidence]))[:10]
        kept[duplicate_index] = strongest.model_copy(update={"evidence": evidence})
    return sorted(kept, key=lambda item: (item.file_path, item.line_start, item.title))


def _is_duplicate(first: ReviewFinding, second: ReviewFinding) -> bool:
    """Compare finding location and normalized content for overlap."""
    if first.file_path != second.file_path:
        return False
    overlap = first.line_start <= second.line_end + 3 and second.line_start <= first.line_end + 3
    same_rule = first.rule_id is not None and first.rule_id == second.rule_id
    if not overlap or (first.category != second.category and not same_rule):
        return False
    first_text = _normalize(f"{first.title} {first.explanation}")
    second_text = _normalize(f"{second.title} {second.explanation}")
    words_first, words_second = set(first_text.split()), set(second_text.split())
    jaccard = len(words_first & words_second) / max(len(words_first | words_second), 1)
    ratio = SequenceMatcher(None, first_text, second_text).ratio()
    return same_rule or jaccard >= 0.55 or ratio >= 0.75


def _normalize(value: str) -> str:
    """Normalize finding text before duplicate comparison."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _strength(finding: ReviewFinding) -> tuple[int, float, int]:
    """Rank findings by confidence and evidence quality for merge selection."""
    severity = {"critical": 4, "high": 3, "medium": 2, "low": 1}[finding.severity]
    return severity, finding.confidence, len(finding.evidence)
