"""Orchestration of local static-analysis tools over pull-request files."""

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from app.models.review import StaticFinding

logger = logging.getLogger(__name__)
MAX_STATIC_ANALYSIS_SECONDS = 60


class StaticAnalyzer:
    """Run available deterministic analyzers against changed repository files."""

    def run(self, files: dict[str, bytes]) -> list[StaticFinding]:
        """Run supported static-analysis tools and normalize their findings."""
        results: list[StaticFinding] = []
        python_files = {
            path: content for path, content in files.items() if path.endswith((".py", ".pyi"))
        }
        js_files = {
            path: content
            for path, content in files.items()
            if path.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"))
        }
        with tempfile.TemporaryDirectory(prefix="pr-review-static-") as temporary:
            root = Path(temporary).resolve()
            materialized = self._materialize(root, files)
            if python_files:
                results.extend(self._ruff(python_files, materialized))
                results.extend(self._bandit(python_files, materialized))
            if js_files:
                results.extend(self._eslint(js_files))
            results.extend(self._semgrep(materialized, bool(python_files or js_files)))
        return results

    def _materialize(self, root: Path, files: dict[str, bytes]) -> dict[str, Path]:
        """Write analyzed virtual files into a temporary workspace for tool execution."""
        materialized: dict[str, Path] = {}
        for relative, contents in files.items():
            candidate = PurePosixPath(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                logger.warning("Skipping unsafe repository path during analysis")
                continue
            path = root.joinpath(*candidate.parts).resolve()
            if root not in path.parents:
                logger.warning("Skipping repository path escaping analysis directory")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            materialized[relative] = path
        return materialized

    def _run(self, command: list[str], *, stdin: bytes | None = None) -> str | None:
        """Run one local analyzer command with bounded time and captured output."""
        try:
            completed = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                check=False,
                timeout=MAX_STATIC_ANALYSIS_SECONDS,
                shell=False,
            )
        except FileNotFoundError:
            logger.info("Static analyzer is not installed: %s", command[0])
            return None
        except subprocess.TimeoutExpired:
            logger.warning("Static analyzer timed out: %s", command[0])
            return None
        output = completed.stdout.decode("utf-8", errors="replace")
        if completed.returncode not in (0, 1):
            logger.warning("Static analyzer failed: %s exit=%d", command[0], completed.returncode)
        return output

    def _ruff(self, files: dict[str, bytes], paths: dict[str, Path]) -> list[StaticFinding]:
        """Run Ruff on changed Python files and map diagnostics to review locations."""
        selected = [paths[path] for path in files if path in paths]
        if not selected:
            return []
        if shutil.which("ruff") is None:
            logger.info("Ruff is not installed; skipping Python lint analysis")
            return []
        output = self._run(
            ["ruff", "check", "--isolated", "--output-format", "json", *map(str, selected)]
        )
        if not output:
            return []
        try:
            findings = json.loads(output)
            return [
                StaticFinding(
                    tool="ruff",
                    rule_id=item["code"],
                    severity="warning",
                    file_path=self._relative(item["filename"], paths),
                    line=item["location"]["row"],
                    message=item["message"],
                )
                for item in findings
                if item.get("code") and item.get("location", {}).get("row")
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            logger.warning("Could not parse Ruff output: %s", error)
            return []

    def _bandit(self, files: dict[str, bytes], paths: dict[str, Path]) -> list[StaticFinding]:
        """Run Bandit on changed Python files and normalize its security findings."""
        selected = [paths[path] for path in files if path in paths]
        if not selected:
            return []
        if shutil.which("bandit") is None:
            logger.info("Bandit is not installed; skipping Python security analysis")
            return []
        output = self._run(["bandit", "-q", "-f", "json", *map(str, selected)])
        if not output:
            return []
        try:
            findings = json.loads(output).get("results", [])
            return [
                StaticFinding(
                    tool="bandit",
                    rule_id=item["test_id"],
                    severity=str(item.get("issue_severity", "medium")).lower(),
                    file_path=self._relative(item["filename"], paths),
                    line=item["line_number"],
                    message=item["issue_text"],
                )
                for item in findings
                if item.get("line_number")
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            logger.warning("Could not parse Bandit output: %s", error)
            return []

    def _eslint(self, files: dict[str, bytes]) -> list[StaticFinding]:
        """Run ESLint on changed JavaScript or TypeScript files when available."""
        if shutil.which("eslint") is None:
            if files:
                logger.info("ESLint is not installed; skipping JavaScript/TypeScript analysis")
            return []
        findings: list[StaticFinding] = []
        for path, content in files.items():
            output = self._run(
                [
                    "eslint",
                    "--no-config-lookup",
                    "--no-ignore",
                    "--stdin",
                    "--stdin-filename",
                    path,
                    "--format",
                    "json",
                ],
                stdin=content,
            )
            if not output:
                continue
            try:
                messages = json.loads(output)[0].get("messages", [])
                findings.extend(
                    StaticFinding(
                        tool="eslint",
                        rule_id=str(item.get("ruleId") or "eslint"),
                        severity="error" if item.get("severity") == 2 else "warning",
                        file_path=path,
                        line=item["line"],
                        message=item["message"],
                    )
                    for item in messages
                    if item.get("line")
                )
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                logger.warning("Could not parse ESLint output for %s: %s", path, error)
        return findings

    def _semgrep(self, paths: dict[str, Path], has_supported_source: bool) -> list[StaticFinding]:
        """Run configured Semgrep rules over the changed-file workspace."""
        executable = shutil.which("semgrep")
        rule_file = Path(__file__).with_name("semgrep_rules.yml")
        if not has_supported_source or not rule_file.is_file():
            return []
        if not executable:
            logger.info("Semgrep is not installed; skipping local generic rules")
            return []
        roots = [
            str(path.parent) for path in paths.values() if path.suffix in {".py", ".js", ".ts"}
        ]
        output = self._run([executable, "--config", str(rule_file), "--json", *sorted(set(roots))])
        if not output:
            return []
        try:
            findings = json.loads(output).get("results", [])
            return [
                StaticFinding(
                    tool="semgrep",
                    rule_id=item.get("check_id", "semgrep"),
                    severity=str(item.get("extra", {}).get("severity", "warning")).lower(),
                    file_path=self._relative(item["path"], paths),
                    line=item["start"]["line"],
                    message=item.get("extra", {}).get("message", "Static pattern matched"),
                )
                for item in findings
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            logger.warning("Could not parse Semgrep output: %s", error)
            return []

    @staticmethod
    def _relative(filename: str, paths: dict[str, Path]) -> str:
        """Convert a command result path to its repository-relative form."""
        resolved = Path(filename).resolve()
        for relative, path in paths.items():
            if path == resolved:
                return relative
        return Path(filename).name
