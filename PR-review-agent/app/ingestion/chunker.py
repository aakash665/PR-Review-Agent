"""Syntax-aware source chunking with stable repository metadata."""

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath

from app.ingestion.parser import ParsedSymbol, parse_symbols


@dataclass(frozen=True)
class CodeChunk:
    """A source or documentation segment with stable IDs, source bounds, and metadata."""

    chunk_id: str
    repository: str
    commit_sha: str
    file_path: str
    language: str
    symbol: str
    symbol_type: str
    chunk_index: int
    start_line: int
    end_line: int
    content_hash: str
    content: str

    def payload(self) -> dict[str, str | int]:
        """Build the Qdrant payload associated with this chunk and its source location."""
        return {
            "repository": self.repository,
            "repository_id": self.repository,
            "commit_sha": self.commit_sha,
            "file_path": self.file_path,
            "language": self.language,
            "symbol": self.symbol,
            "symbol_type": self.symbol_type,
            "chunk_index": self.chunk_index,
            "is_test": any(part in self.file_path.lower() for part in ("test", "spec")),
            "is_documentation": self.language in {"markdown", "text"},
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content_hash": self.content_hash,
            "content": self.content,
        }


class CodeChunker:
    """Split repository files into syntax-aware chunks bounded by line and character limits."""

    def __init__(self, max_lines: int = 180, max_chars: int = 12000) -> None:
        if max_lines < 1 or max_chars < 256:
            raise ValueError("Chunk line and character limits must be positive and bounded")
        self.max_lines = max_lines
        self.max_chars = max_chars

    def chunk(
        self, source: str, *, repository: str, commit_sha: str, file_path: str
    ) -> list[CodeChunk]:
        """Parse and split source into indexed chunks for the requested commit and file."""
        lines = source.splitlines()
        if not lines:
            return []
        language = _language(file_path)
        if language in {"markdown", "text"}:
            symbols = _document_sections(lines)
        else:
            symbols = parse_symbols(source, file_path)
            if not symbols:
                symbols = [ParsedSymbol(PurePosixPath(file_path).stem, "module", 1, len(lines))]
        imports = [line[:300] for line in lines[:100] if _is_import(line, language)][:20]
        chunks: list[CodeChunk] = []
        for symbol in symbols:
            start = max(symbol.start_line, 1)
            end = min(max(symbol.end_line, start), len(lines))
            parent = f"Parent: {symbol.parent}\n" if symbol.parent else ""
            prefix = f"{parent}Imports:\n" + "\n".join(imports) + "\n"
            for chunk_index, (chunk_start, chunk_end, body) in enumerate(
                self._segments(lines, start, end)
            ):
                content = f"File: {file_path}\nSymbol: {symbol.name}\n{prefix}{body}"
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                identity = (
                    f"{repository}:{commit_sha}:{file_path}:{symbol.name}:"
                    f"{chunk_start}:{chunk_index}"
                )
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        repository=repository,
                        commit_sha=commit_sha,
                        file_path=file_path,
                        language=language,
                        symbol=symbol.name,
                        symbol_type=symbol.symbol_type,
                        chunk_index=chunk_index,
                        start_line=chunk_start,
                        end_line=chunk_end,
                        content_hash=content_hash,
                        content=content,
                    )
                )
        return chunks

    def _segments(self, lines: list[str], start: int, end: int) -> list[tuple[int, int, str]]:
        """Partition oversized declarations without exceeding chunk-size limits."""
        segments: list[tuple[int, int, str]] = []
        current: list[str] = []
        current_start = start
        current_chars = 0

        def flush(last_line: int) -> None:
            """Emit the buffered segment and reset its working state."""
            nonlocal current_chars, current_start
            if current:
                segments.append((current_start, last_line, "\n".join(current)))
                current.clear()
                current_chars = 0

        for line_number in range(start, end + 1):
            line = lines[line_number - 1]
            if len(line) > self.max_chars:
                flush(line_number - 1)
                for offset in range(0, len(line), self.max_chars):
                    segments.append(
                        (
                            line_number,
                            line_number,
                            line[offset : offset + self.max_chars],
                        )
                    )
                current_start = line_number + 1
                continue
            next_size = current_chars + len(line) + (1 if current else 0)
            if current and (len(current) >= self.max_lines or next_size > self.max_chars):
                flush(line_number - 1)
                current_start = line_number
            current.append(line)
            current_chars += len(line) + (1 if len(current) > 1 else 0)
        flush(end)
        return segments


def _language(file_path: str) -> str:
    """Select the syntax parser language corresponding to a file extension."""
    suffix = PurePosixPath(file_path).suffix.lower()
    return {
        ".py": "python",
        ".pyi": "python",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".md": "markdown",
        ".rst": "text",
        ".txt": "text",
    }.get(suffix, suffix.lstrip(".") or "text")


def _is_import(line: str, language: str) -> bool:
    """Recognize import declarations used as chunk-level shared context."""
    stripped = line.strip()
    return (
        stripped.startswith(("import ", "from "))
        if language == "python"
        else stripped.startswith(("import ", "package "))
    )


def _document_sections(lines: list[str]) -> list[ParsedSymbol]:
    """Split documentation into bounded heading-based sections."""
    headings = [
        (index + 1, line.lstrip("#").strip())
        for index, line in enumerate(lines)
        if re.match(r"^\s{0,3}#{1,6}\s+", line)
    ]
    if not headings:
        return [ParsedSymbol("document", "documentation", 1, len(lines))]
    result: list[ParsedSymbol] = []
    for index, (start, heading) in enumerate(headings):
        end = headings[index + 1][0] - 1 if index + 1 < len(headings) else len(lines)
        result.append(ParsedSymbol(heading or "section", "documentation", start, end))
    return result
