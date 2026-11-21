"""Probe one selected test against a patch without modifying the baseline source."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal, Sequence

from commentry import FileChange

PROBE_RESULT_PREFIX = "BLAST_RADIUS_PROBE_RESULT="


def _deleted_paths(patch_text: str) -> set[str]:
    lines = patch_text.splitlines()
    deleted = set()
    for index, line in enumerate(lines[:-1]):
        if line.startswith("--- a/") and lines[index + 1] == "+++ /dev/null":
            deleted.add(line[6:].split("\t", 1)[0])
    return deleted


def _write_changes(
    changes: Sequence[FileChange], patch_text: str, worktree: Path
) -> None:
    root = worktree.resolve()
    deleted = _deleted_paths(patch_text)
    for change in changes:
        destination = (root / change.path).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"Patch path escapes worktree: {change.path!r}")
        if change.path in deleted:
            destination.unlink(missing_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(change.modified, encoding="utf-8")


def _reports_failure(output: str, test_id: str) -> bool:
    output = output.replace("\\", "/")
    test_id = test_id.replace("\\", "/")
    if re.search(rf"(?m)^(?:FAILED|ERROR)\s+{re.escape(test_id)}(?:\s|$)", output):
        return True
    test_file = test_id.split("::", 1)[0]
    return f"ERROR collecting {test_file}" in output


PROBE_RUNNER = r"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import pytest

PREFIX = "BLAST_RADIUS_PROBE_RESULT="


class ProbeReports:
    def __init__(self):
        self.results = {}
        self.collection_failed = False

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            self.results.setdefault(report.nodeid, "passed")
        elif report.failed:
            self.results[report.nodeid] = "failed"
        elif report.skipped:
            self.results[report.nodeid] = "unknown"

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_failed = True


plugin = ProbeReports()
remaining = []
modules = {}
for node_id in sys.argv[1:]:
    parts = node_id.split("::")
    if len(parts) != 3 or "[" in parts[-1]:
        remaining.append(node_id)
        continue
    filename, class_name, method = parts
    path = Path(filename)
    if not path.is_file():
        remaining.append(node_id)
        continue
    try:
        module = modules.get(filename)
        if module is None:
            module_name = "probe_" + str(len(modules))
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            modules[filename] = module
        test_class = getattr(module, class_name)
        if not issubclass(test_class, unittest.TestCase):
            remaining.append(node_id)
            continue
        case = test_class(method)
        result = unittest.TestResult()
        unittest.TestSuite([case]).run(result)
        if result.errors or result.failures:
            plugin.results[node_id] = "failed"
        elif result.skipped:
            plugin.results[node_id] = "unknown"
        else:
            plugin.results[node_id] = "passed"
    except Exception:
        remaining.append(node_id)

code = 0
if remaining:
    code = pytest.main(
        [
            "-q",
            "--tb=no",
            "--color=no",
            "-p",
            "no:cacheprovider",
            *remaining,
        ],
        plugins=[plugin],
    )
print(
    PREFIX
    + json.dumps(
        {
            "results": plugin.results,
            "collection_failed": plugin.collection_failed,
            "exit_code": code,
        },
        separators=(",", ":"),
    )
)
"""


def _parse_probe_results(output: str, test_ids: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {test_id: "unknown" for test_id in test_ids}
    lines = output.splitlines()
    encoded = next(
        (
            line[len(PROBE_RESULT_PREFIX):]
            for line in reversed(lines)
            if line.startswith(PROBE_RESULT_PREFIX)
        ),
        None,
    )
    if encoded is None:
        return result
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError):
        return result
    if payload.get("collection_failed"):
        return result
    for test_id, outcome in payload.get("results", {}).items():
        if test_id in result and outcome in {"passed", "failed"}:
            result[test_id] = outcome
    return result


def probe_test_failures(
    source_root: Path,
    changes: Sequence[FileChange],
    patch_text: str,
    test_ids: Sequence[str],
    deadline: float,
    timeout: float,
) -> dict[str, Literal["failed", "passed", "unknown"]]:
    unique_ids = list(dict.fromkeys(test_ids))
    unknown: dict[str, Literal["failed", "passed", "unknown"]] = {
        test_id: "unknown" for test_id in unique_ids
    }
    if not unique_ids:
        return unknown
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return unknown

    with tempfile.TemporaryDirectory(prefix="blast-radius-probe-", dir="/tmp") as name:
        worktree = Path(name) / "source"
        shutil.copytree(source_root, worktree)
        _write_changes(changes, patch_text, worktree)
        runner = Path(name) / "probe_tests.py"
        runner.write_text(PROBE_RUNNER, encoding="utf-8")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return unknown
        environment = os.environ.copy()
        inherited_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(worktree)
            if not inherited_pythonpath
            else str(worktree) + os.pathsep + inherited_pythonpath
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["SKIP_INTEGRATION"] = "0"
        try:
            result = subprocess.run(
                [sys.executable, str(runner), *unique_ids],
                cwd=worktree,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=min(timeout, remaining),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return unknown
    return {
        test_id: outcome if outcome in {"failed", "passed", "unknown"} else "unknown"
        for test_id, outcome in _parse_probe_results(
            result.stdout or "", unique_ids
        ).items()
    }


def probe_test_failure(
    source_root: Path,
    changes: Sequence[FileChange],
    patch_text: str,
    test_id: str,
    deadline: float,
    per_test_timeout: float,
) -> Literal["failed", "passed", "unknown"]:
    """Report a conclusive single-test result, otherwise preserve selection."""
    return probe_test_failures(
        source_root,
        changes,
        patch_text,
        [test_id],
        deadline,
        per_test_timeout,
    )[test_id]
