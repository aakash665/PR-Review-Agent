"""Cross-language extraction of source symbols and their enclosing scopes."""

import ast
from dataclasses import dataclass

from tree_sitter import Node
from tree_sitter_language_pack import get_parser


@dataclass(frozen=True)
class ParsedSymbol:
    """A named declaration with its qualified scope and source-line range."""

    name: str
    symbol_type: str
    start_line: int
    end_line: int
    parent: str | None = None


PYTHON_SUFFIXES = {".py", ".pyi"}
TREE_SITTER_LANGUAGES = {
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
}
DECLARATION_TYPES = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "function_declaration": "function",
    "method_declaration": "method",
    "constructor_declaration": "constructor",
    "lexical_declaration": "function",
    "function_definition": "function",
    "module": "module",
    "struct_item": "struct",
    "trait_item": "trait",
    "impl_item": "implementation",
    "method": "method",
    "class": "class",
    "module_function": "function",
}


def parse_symbols(source: str, file_path: str) -> list[ParsedSymbol]:
    """Extract language-specific declarations from a source file."""
    suffix = "." + file_path.rsplit(".", maxsplit=1)[-1].lower() if "." in file_path else ""
    if suffix in PYTHON_SUFFIXES:
        try:
            return _parse_python(source)
        except SyntaxError:
            return _parse_tree_sitter(source, suffix)
    if suffix in TREE_SITTER_LANGUAGES:
        return _parse_tree_sitter(source, suffix)
    return []


def _parse_python(source: str) -> list[ParsedSymbol]:
    """Extract Python declarations using the standard-library AST."""
    tree = ast.parse(source)
    symbols: list[ParsedSymbol] = []

    class Visitor(ast.NodeVisitor):
        """Collect nested Python declarations while tracking their qualified scope."""

        parents: list[str]

        def __init__(self) -> None:
            self.parents = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            """Record a Python class and visit declarations in its body."""
            qualified = ".".join([*self.parents, node.name])
            symbols.append(
                ParsedSymbol(qualified, "class", node.lineno, node.end_lineno or node.lineno)
            )
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            """Record a synchronous Python function or method."""
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            """Record an asynchronous Python function or method."""
            self._visit_function(node)

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            """Record a function and traverse it under its qualified parent scope."""
            qualified = ".".join([*self.parents, node.name])
            symbol_type = "method" if self.parents else "function"
            parent = ".".join(self.parents) if self.parents else None
            symbols.append(
                ParsedSymbol(
                    qualified, symbol_type, node.lineno, node.end_lineno or node.lineno, parent
                )
            )
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

    Visitor().visit(tree)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.If, ast.For, ast.Expr)):
            symbols.append(
                ParsedSymbol(
                    "module",
                    "module",
                    node.lineno,
                    node.end_lineno or node.lineno,
                )
            )
    return symbols


def _parse_tree_sitter(source: str, suffix: str) -> list[ParsedSymbol]:
    """Extract supported-language declarations with Tree-sitter."""
    language = TREE_SITTER_LANGUAGES.get(suffix)
    if language is None:
        return []
    parser = get_parser(language)
    tree = parser.parse(source.encode("utf-8"))
    symbols: list[ParsedSymbol] = []

    def walk(node: Node, parent_name: str | None = None) -> None:
        """Visit a syntax-tree node and its named descendants to collect declarations."""
        node_type = node.type
        children = node.named_children
        name_node = next(
            (
                child
                for child in children
                if child.type in {"identifier", "type_identifier", "property_identifier"}
            ),
            None,
        )
        name_text = name_node.text if name_node is not None else None
        name = name_text.decode("utf-8", errors="replace") if name_text else node_type
        kind = DECLARATION_TYPES.get(node_type)
        current_parent = parent_name
        if kind:
            start_line = int(node.start_point.row) + 1
            end_line = int(node.end_point.row) + 1
            qualified = f"{parent_name}.{name}" if parent_name else name
            symbols.append(ParsedSymbol(qualified, kind, start_line, end_line, parent_name))
            if kind in {"class", "interface", "struct", "trait", "implementation", "module"}:
                current_parent = name
        for child in children:
            walk(child, current_parent)

    walk(tree.root_node)
    return symbols
