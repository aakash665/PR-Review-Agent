"""Unified-diff parsing and mapping of changed lines to source symbols."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from app.ingestion.parser import ParsedSymbol, parse_symbols
from app.models.review import ChangedFile, ChangedLine

FILE_HEADER = re.compile(r"^diff --git a/(.*?) b/(.*?)$")
HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class DiffFile:
    """Parsed per-file diff data, including changed lines and source coordinates."""

    path: str
    old_path: str
    status: str = "modified"
    additions: int = 0
    deletions: int = 0
    changed_lines: list[ChangedLine] = field(default_factory=list)
    commentable_lines: set[int] = field(default_factory=set)
    patch: list[str] = field(default_factory=list)

    def as_model(self, api_file: dict[str, object] | None = None) -> ChangedFile:
        """Combine parsed diff data with GitHub file metadata as a validated model."""
        metadata = api_file or {}
        additions = metadata.get("additions")
        deletions = metadata.get("deletions")
        return ChangedFile(
            file_path=self.path,
            status=str(metadata.get("status") or self.status),
            additions=additions if isinstance(additions, int) else self.additions,
            deletions=deletions if isinstance(deletions, int) else self.deletions,
            changed_lines=self.changed_lines,
            patch="\n".join(self.patch),
        )


@dataclass(frozen=True)
class DiffAnalysis:
    """Complete diff analysis and the set of lines eligible for inline comments."""

    files: list[ChangedFile]
    commentable_lines: dict[str, frozenset[int]]

    def is_commentable(self, path: str, line: int) -> bool:
        """Check whether a line is an added line suitable for an inline review comment."""
        return line in self.commentable_lines.get(path, frozenset())


class DiffAnalyzer:
    """Parse unified diffs and map added lines to commentable locations."""

    def analyze(self, diff: str, api_files: list[dict[str, object]] | None = None) -> DiffAnalysis:
        """Parse the diff and optional GitHub file metadata into changed-file records."""
        files: list[DiffFile] = []
        current: DiffFile | None = None
        old_line = 0
        new_line = 0
        in_hunk = False
        for line in diff.splitlines():
            header = FILE_HEADER.match(line)
            if header:
                current = DiffFile(path=header.group(2), old_path=header.group(1))
                files.append(current)
                in_hunk = False
                continue
            if current is None:
                continue
            if line.startswith("rename from "):
                current.status = "renamed"
                continue
            if line.startswith("new file mode"):
                current.status = "added"
                continue
            if line.startswith("deleted file mode"):
                current.status = "removed"
                continue
            if line.startswith("+++ b/"):
                current.path = line[6:]
                continue
            if line.startswith("+++ /dev/null"):
                current.status = "removed"
                continue
            if line.startswith("--- /dev/null"):
                current.status = "added"
                continue
            hunk = HUNK_HEADER.match(line)
            if hunk:
                old_line = int(hunk.group(1))
                new_line = int(hunk.group(3))
                in_hunk = True
                current.patch.append(line)
                continue
            if not in_hunk or line.startswith("\\ No newline"):
                continue
            current.patch.append(line)
            if line.startswith("+"):
                current.additions += 1
                current.changed_lines.append(
                    ChangedLine(line=new_line, content=line[1:], change_type="added")
                )
                current.commentable_lines.add(new_line)
                new_line += 1
            elif line.startswith("-"):
                current.deletions += 1
                current.changed_lines.append(
                    ChangedLine(line=old_line, content=line[1:], change_type="removed")
                )
                old_line += 1
            elif line.startswith(" "):
                current.commentable_lines.add(new_line)
                old_line += 1
                new_line += 1
        api_by_path = {str(item.get("filename")): item for item in (api_files or [])}
        analyzed_paths = {item.path for item in files}
        for item in api_files or []:
            path = str(item.get("filename", ""))
            if path and path not in analyzed_paths and str(item.get("status")) != "removed":
                files.append(
                    DiffFile(
                        path=path,
                        old_path=str(item.get("previous_filename") or path),
                        status=str(item.get("status") or "modified"),
                    )
                )
        return DiffAnalysis(
            files=[item.as_model(api_by_path.get(item.path)) for item in files],
            commentable_lines={item.path: frozenset(item.commentable_lines) for item in files},
        )


def identify_changed_symbols(
    diff: DiffAnalysis, repository_files: Mapping[str, bytes]
) -> dict[str, list[str]]:
    """Map changed lines to enclosing symbols using the corresponding file contents."""
    result: dict[str, list[str]] = {}
    for file in diff.files:
        source_bytes = repository_files.get(file.file_path)
        if source_bytes is None:
            continue
        try:
            source = source_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue
        symbols = parse_symbols(source, file.file_path)
        added_lines = {item.line for item in file.changed_lines if item.change_type == "added"}
        enclosing: dict[str, ParsedSymbol] = {}
        for line in added_lines:
            matches = [symbol for symbol in symbols if symbol.start_line <= line <= symbol.end_line]
            if matches:
                match = min(
                    matches,
                    key=lambda symbol: symbol.end_line - symbol.start_line,
                )
                enclosing[match.name] = match
        if enclosing:
            result[file.file_path] = list(enclosing)
    return result
