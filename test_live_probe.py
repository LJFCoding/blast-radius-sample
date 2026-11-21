import json
import difflib
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from commentry import file_changes
from live_probe import PROBE_RESULT_PREFIX, probe_test_failure, probe_test_failures
from selector import live_probe_candidate


class LiveProbeTests(unittest.TestCase):
    def test_candidate_requires_direct_evidence_and_expensive_selection(self):
        tests = [
            {"id": "direct", "number": 0, "seconds": 0.3},
            {"id": "unrelated", "number": 1, "seconds": 9.0},
        ]
        impact = {
            "tests": ["direct", "unrelated"],
            "statements": {"sqlglot/example.py": [[2, 2, [0]]]},
            "functions": {},
            "fixtures": {},
            "source_reads": {},
            "metadata_tests": [],
        }
        self.assertEqual(
            live_probe_candidate(
                {"direct", "unrelated"},
                tests,
                {"sqlglot/example.py": {2}},
                impact,
                [[0], [1]],
            ),
            tests[0],
        )
        self.assertEqual(
            live_probe_candidate(
                {"direct", "slow"},
                tests,
                {"sqlglot/example.py": {2}},
                impact,
                [[0]],
            ),
            tests[0],
        )

    def test_candidate_must_reduce_the_current_worker_bottleneck(self):
        tests = [
            {"id": "direct", "number": 0, "seconds": 0.3},
            {"id": "slow", "number": 1, "seconds": 20.0},
        ]
        impact = {
            "tests": ["direct", "slow"],
            "statements": {"sqlglot/example.py": [[2, 2, [0]]]},
            "functions": {},
            "fixtures": {},
            "source_reads": {},
            "metadata_tests": [],
        }
        self.assertIsNone(
            live_probe_candidate(
                {"direct"},
                tests,
                {"sqlglot/example.py": {2}},
                impact,
                [[0, 1]],
            )
        )

    def test_confirmed_failure_keeps_patched_source_out_of_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root, patch_text, changes, test_id, modified = self._example_patch(directory)

            def report_failure(command, **kwargs):
                worktree = Path(kwargs["cwd"])
                self.assertEqual(
                    (worktree / "sqlglot/example.py").read_text(), modified
                )
                self.assertEqual(
                    (root / "sqlglot/example.py").read_text(), "value = 1\n"
                )
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout=(
                        PROBE_RESULT_PREFIX
                        + json.dumps({
                            "results": {test_id: "failed"},
                            "collection_failed": False,
                            "exit_code": 1,
                        })
                    ),
                )

            with patch("live_probe.subprocess.run", side_effect=report_failure):
                self.assertEqual(
                    probe_test_failure(
                        root,
                        changes,
                        patch_text,
                        test_id,
                        time.monotonic() + 10,
                        2,
                    ),
                    "failed",
                )

    def test_passing_probe_is_distinguishable_from_an_inconclusive_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root, patch_text, changes, test_id, _ = self._example_patch(directory)
            with patch(
                "live_probe.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [],
                    0,
                    stdout=(
                        PROBE_RESULT_PREFIX
                        + json.dumps({
                            "results": {test_id: "passed"},
                            "collection_failed": False,
                            "exit_code": 0,
                        })
                    ),
                ),
            ):
                self.assertEqual(
                    probe_test_failure(
                        root,
                        changes,
                        patch_text,
                        test_id,
                        time.monotonic() + 10,
                        2,
                    ),
                    "passed",
                )

    def test_timed_out_probe_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root, patch_text, changes, test_id, _ = self._example_patch(directory)
            with patch(
                "live_probe.subprocess.run",
                side_effect=subprocess.TimeoutExpired("pytest", 1),
            ):
                self.assertEqual(
                    probe_test_failure(
                        root,
                        changes,
                        patch_text,
                        test_id,
                        time.monotonic() + 10,
                        1,
                    ),
                    "unknown",
                )

    def test_batch_reports_each_conclusive_test_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root, patch_text, changes, test_id, _ = self._example_patch(directory)
            other = "tests/test_example.py::test_other"
            with patch(
                "live_probe.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [],
                    1,
                    stdout=(
                        PROBE_RESULT_PREFIX
                        + json.dumps({
                            "results": {
                                test_id: "failed",
                                other: "passed",
                            },
                            "collection_failed": False,
                            "exit_code": 1,
                        })
                    ),
                ),
            ):
                self.assertEqual(
                    probe_test_failures(
                        root,
                        changes,
                        patch_text,
                        [test_id, other],
                        time.monotonic() + 10,
                        2,
                    ),
                    {test_id: "failed", other: "passed"},
                )

    def _example_patch(self, directory):
        root = Path(directory)
        source = root / "sqlglot/example.py"
        source.parent.mkdir()
        source.write_text("value = 1\n")
        test_file = root / "tests/test_example.py"
        test_file.parent.mkdir()
        test_file.write_text("def test_value():\n    pass\n")
        modified = "value = 2\n"
        diff = difflib.unified_diff(
            source.read_text().splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="a/sqlglot/example.py",
            tofile="b/sqlglot/example.py",
        )
        patch_text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
        return (
            root,
            patch_text,
            file_changes(patch_text, root),
            "tests/test_example.py::test_value",
            modified,
        )
