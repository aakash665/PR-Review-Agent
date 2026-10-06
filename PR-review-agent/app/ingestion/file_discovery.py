"""Repository path filtering and deterministic local file discovery."""

from collections.abc import Iterator
from pathlib import Path

MAX_SOURCE_BYTES = 1_000_000
IGNORED_DIRECTORIES = {
    ".git",
    ".next",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "build",
    "dist",
    "coverage",
    "__pycache__",
    "vendor",
    "generated",
}
IGNORED_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "Cargo.lock",
    "go.sum",
    "Pipfile.lock",
    "expected.json",
}
SUPPORTED_EXTENSIONS = {
    ".py",
    ".pyi",
    ".java",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".rb",
    ".md",
    ".rst",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".jar", ".class", ".woff"}


def is_indexable(path: str | Path, size: int | None = None) -> bool:
    """Determine whether a repository path and optional size meet indexing policy."""
    normalized = str(path).replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute() or normalized.startswith("/") or ".." in normalized.split("/"):
        return False
    if any(part.lower() in IGNORED_DIRECTORIES for part in candidate.parts):
        return False
    if candidate.name in IGNORED_FILES or candidate.name.startswith("."):
        return False
    if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS | {".md"}:
        return False
    if candidate.suffix.lower() in BINARY_SUFFIXES:
        return False
    return size is None or size <= MAX_SOURCE_BYTES


def discover_files(root: Path) -> Iterator[Path]:
    """Yield indexable files below a local repository root in stable order."""
    root = root.resolve()
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        try:
            if is_indexable(relative, path.stat().st_size):
                yield path
        except OSError:
            continue
