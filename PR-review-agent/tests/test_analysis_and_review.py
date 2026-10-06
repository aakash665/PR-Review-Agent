"""Tests for diff mapping, chunking, finding quality gates, and evaluation metrics."""

from pathlib import Path

from app.analysis.diff_analyzer import DiffAnalyzer, identify_changed_symbols
from app.ingestion.chunker import CodeChunker
from app.ingestion.file_discovery import discover_files, is_indexable
from app.models.review import ReviewFinding
from app.review.confidence import filter_by_confidence
from app.review.deduplicator import deduplicate_findings
from app.review.github_formatter import format_inline_comment, format_review_summary
from evaluation.metrics import calculate_metrics


def _finding(
    *,
    confidence: float = 0.9,
    line: int = 5,
    title: str = "Missing input validation",
    explanation: str = "The value is passed into the query without validation.",
    severity: str = "high",
    category: str = "bug",
) -> ReviewFinding:
    return ReviewFinding(
        category=category,
        severity=severity,
        confidence=confidence,
        title=title,
        explanation=explanation,
        evidence=["The changed query interpolates the supplied value."],
        file_path="app/service.py",
        line_start=line,
        line_end=line,
        suggestion="Use a bound parameter.",
        rationale="The database API supports parameter binding.",
        verified=True,
    )


def test_diff_parser_maps_added_removed_and_commentable_lines() -> None:
    fixture = Path("fixtures/python_project/sample.diff").read_text(encoding="utf-8")
    analysis = DiffAnalyzer().analyze(fixture)
    assert len(analysis.files) == 1
    changed = analysis.files[0]
    added = [line for line in changed.changed_lines if line.change_type == "added"]
    removed = [line for line in changed.changed_lines if line.change_type == "removed"]
    assert [(line.line, line.content) for line in added] == [
        (5, '    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")')
    ]
    assert removed[0].line == 5
    assert analysis.is_commentable(changed.file_path, 5)
    assert not analysis.is_commentable(changed.file_path, 99)


def test_changed_line_is_mapped_to_its_enclosing_symbol() -> None:
    fixture = Path("fixtures/python_project")
    diff = DiffAnalyzer().analyze((fixture / "sample.diff").read_text(encoding="utf-8"))
    sources = {
        "app/service.py": (fixture / "app/service.py").read_bytes(),
    }
    assert identify_changed_symbols(diff, sources) == {"app/service.py": ["find_user"]}


def test_python_ast_chunking_preserves_parent_imports_and_source_lines() -> None:
    source = (
        "from app.errors import NotFound\n\n"
        "class UserService:\n"
        "    def get_user(self, user_id):\n"
        "        return self.repository.find(user_id)\n"
    )
    chunks = CodeChunker().chunk(
        source, repository="octo/repo", commit_sha="abc123", file_path="app/service.py"
    )
    method = next(chunk for chunk in chunks if chunk.symbol_type == "method")
    assert method.symbol == "UserService.get_user"
    assert method.start_line == 4
    assert "Parent: UserService" in method.content
    assert "from app.errors import NotFound" in method.content
    assert method.payload()["commit_sha"] == "abc123"


def test_chunker_bounds_single_oversized_source_lines() -> None:
    chunks = CodeChunker(max_lines=20, max_chars=1200).chunk(
        "#" + ("x" * 3000),
        repository="octo/repo",
        commit_sha="abc123",
        file_path="app/generated.py",
    )
    assert len(chunks) == 3
    assert all(len(chunk.content) <= 1400 for chunk in chunks)
    assert {chunk.start_line for chunk in chunks} == {1}


def test_tree_sitter_chunks_supported_language_declaration() -> None:
    chunks = CodeChunker().chunk(
        "public class UserService { public User find(String id) { return null; } }",
        repository="octo/repo",
        commit_sha="abc123",
        file_path="src/UserService.java",
    )
    assert any(chunk.symbol_type == "class" for chunk in chunks)
    assert any(chunk.symbol_type == "method" for chunk in chunks)


def test_tree_sitter_extracts_typescript_function_symbols() -> None:
    chunks = CodeChunker().chunk(
        "export function isAdmin(user: User): boolean { return user.role === 'admin'; }",
        repository="octo/repo",
        commit_sha="abc123",
        file_path="src/auth.ts",
    )
    assert any(chunk.symbol == "isAdmin" and chunk.symbol_type == "function" for chunk in chunks)


def test_discovery_excludes_build_artifacts_and_fixture_labels(tmp_path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "app" / "service.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "node_modules" / "pkg" / "code.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "expected.json").write_text("{}", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
    assert [path.name for path in discover_files(tmp_path)] == ["service.py"]
    assert not is_indexable("target/classes/User.class")


def test_confidence_thresholds_are_exact_and_configurable() -> None:
    result = filter_by_confidence(
        [_finding(confidence=0.8), _finding(confidence=0.6), _finding(confidence=0.59)],
        inline_threshold=0.8,
        summary_threshold=0.6,
    )
    assert len(result.inline) == 1
    assert len(result.summary_only) == 1
    assert len(result.discarded) == 1


def test_deduplicator_merges_duplicate_evidence_and_keeps_strongest() -> None:
    first = _finding(
        confidence=0.82,
        title="Unsafe SQL query input",
        explanation="User input is interpolated into SQL without binding.",
        category="security",
    )
    second = _finding(
        confidence=0.96,
        title="SQL injection via interpolation",
        explanation="User input is interpolated into the SQL query without parameter binding.",
        severity="critical",
        category="security",
    ).model_copy(update={"evidence": ["Bandit identified unsafe query interpolation."]})
    result = deduplicate_findings([first, second])
    assert len(result) == 1
    assert result[0].confidence == 0.96
    assert len(result[0].evidence) == 2


def test_github_formatting_and_summary_are_actionable() -> None:
    finding = _finding()
    inline = format_inline_comment(finding)
    assert "The value is passed into the query" in inline
    assert "Use a bound parameter." in inline
    summary = format_review_summary(
        files_analyzed=1,
        additions=1,
        deletions=1,
        findings=[finding],
        summary_only=[],
        discarded_count=0,
        rejected_count=2,
        llm_summary="A query input can be exploited.",
        positive_changes=[],
        review_confidence=0.92,
    )
    assert "Files analyzed: 1" in summary
    assert "Rejected by verifier: 2" in summary
    assert "SQL" not in summary


def test_evaluation_metrics_measure_precision_recall_lines_and_retrieval() -> None:
    expected = [
        {"file_path": "app/service.py", "line": 5, "category": "security"},
        {"file_path": "app/other.py", "line": 10, "category": "bug"},
    ]
    actual = [
        {"file_path": "app/service.py", "line_start": 5, "category": "security"},
        {"file_path": "app/extra.py", "line_start": 2, "category": "bug"},
    ]
    result = calculate_metrics(
        expected,
        actual,
        retrieved_paths={"app/safe_database.py"},
        retrieval_targets=["app/safe_database.py", "README.md"],
    )
    assert result.precision == 0.5
    assert result.recall == 0.5
    assert result.f1 == 0.5
    assert result.false_positive_rate == 0.5
    assert result.line_accuracy == 1
    assert result.retrieval_hit_rate == 0.5
