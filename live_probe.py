"""Probe one selected test against a patch without modifying the baseline source."""

from __future__ import annotations

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


def probe_test_failure(
    source_root: Path,
    changes: Sequence[FileChange],
    patch_text: str,
    test_id: str,
    deadline: float,
    per_test_timeout: float,
) -> Literal["failed", "passed", "unknown"]:
    """Report a conclusive single-test result, otherwise preserve selection."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return "unknown"

    with tempfile.TemporaryDirectory(prefix="blast-radius-probe-", dir="/tmp") as name:
        worktree = Path(name) / "source"
        shutil.copytree(source_root, worktree)
        _write_changes(changes, patch_text, worktree)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "unknown"

        environment = os.environ.copy()
        inherited_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(worktree)
            if not inherited_pythonpath
            else str(worktree) + os.pathsep + inherited_pythonpath
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["SKIP_INTEGRATION"] = "0"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=no",
            "--color=no",
            "-x",
            "-p",
            "no:cacheprovider",
            test_id,
        ]
        try:
            result = subprocess.run(
                command,
                cwd=worktree,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=min(per_test_timeout, remaining),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "unknown"
        if result.returncode == 0:
            return "passed"
        if _reports_failure(result.stdout or "", test_id):
            return "failed"
        return "unknown"
