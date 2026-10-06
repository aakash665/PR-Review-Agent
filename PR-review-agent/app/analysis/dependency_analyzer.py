"""Language-aware extraction of direct import dependencies."""

import ast
import re


def imported_modules(source: str, language: str) -> set[str]:
    """Extract direct import names for a supported source language."""
    if language == "python":
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return set()
        return {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
    if language in {"java", "javascript", "typescript"}:
        return set(re.findall(r"""(?:import|from)\s+["']?([\w./@-]+)""", source))
    return set()
