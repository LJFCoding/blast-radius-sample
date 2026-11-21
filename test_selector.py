import difflib
import math
import tempfile
import unittest
from pathlib import Path

from selector import (
    build_interval_indexes, categorize_patch, choose_workers, inspect_patch,
    patch_changes, prioritize_workers, select_tests, validate_test_inventory,
    canary_priority_scores,
)


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.tests = {"a", "b", "unmapped", "process", "unrelated"}
        self.impact = {
            "tests": ["a", "b", "process", "unrelated"],
            "process_tests": [2],
            "fixtures": {"tests/fixtures/example.sql": [1]},
            "functions": {"sqlglot/example.py": [[10, 20, [0]]]},
            "statements": {},
            "source_reads": {},
            "uncertain_source_reads": [],
            "metadata_tests": [],
        }

    def test_function_and_fixture_changes_are_combined(self):
        patch = """\
diff --git a/sqlglot/example.py b/sqlglot/example.py
--- a/sqlglot/example.py
+++ b/sqlglot/example.py
@@ -14,1 +14,1 @@
-old
+new
diff --git a/tests/fixtures/example.sql b/tests/fixtures/example.sql
--- a/tests/fixtures/example.sql
+++ b/tests/fixtures/example.sql
@@ -1 +1 @@
-old
+new
"""
        self.assertEqual(
            select_tests(patch_changes(patch), self.impact, self.tests),
            {"a", "b", "unmapped"},
        )

    def test_unknown_or_module_level_change_runs_every_test(self):
        for changes in (None, {"missing.py": {1}}, {"sqlglot/example.py": {1}}):
            with self.subTest(changes=changes):
                self.assertEqual(select_tests(changes, self.impact, self.tests), self.tests)

    def test_function_change_retains_unmapped_tests(self):
        selected = select_tests({"sqlglot/example.py": {15}}, self.impact, self.tests)
        self.assertEqual(selected, {"a", "unmapped"})

    def test_process_pool_test_is_retained_when_its_coverage_matches(self):
        self.impact["functions"]["sqlglot/example.py"] = [[10, 20, [2]]]
        selected = select_tests({"sqlglot/example.py": {15}}, self.impact, self.tests)
        self.assertEqual(selected, {"process", "unmapped"})

    def test_function_header_change_uses_function_coverage(self):
        selected = select_tests({"sqlglot/example.py": {10}}, self.impact, self.tests)
        self.assertEqual(selected, {"a", "unmapped"})

    def test_statement_coverage_narrows_function_coverage(self):
        self.impact["functions"]["sqlglot/example.py"] = [[10, 20, [0, 1]]]
        self.impact["statements"]["sqlglot/example.py"] = [[14, 15, [0]]]
        selected = select_tests({"sqlglot/example.py": {15}}, self.impact, self.tests)
        self.assertEqual(selected, {"a", "unmapped"})

    def test_unknown_statement_uses_function_coverage(self):
        self.impact["functions"]["sqlglot/example.py"] = [[10, 20, [0, 1]]]
        self.impact["statements"]["sqlglot/example.py"] = [[14, 15, [0]]]
        selected = select_tests({"sqlglot/example.py": {16}}, self.impact, self.tests)
        self.assertEqual(selected, {"a", "b", "unmapped"})

    def test_comment_change_includes_source_and_metadata_readers(self):
        self.impact["source_reads"] = {"sqlglot/example.py": [1]}
        self.impact["metadata_tests"] = [0]
        selected = select_tests(
            {"sqlglot/example.py": {15}}, self.impact, self.tests, frozenset({"sqlglot/example.py"})
        )
        self.assertEqual(selected, {"a", "b", "unmapped"})

    def test_standalone_comment_is_not_assumed_harmless(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = "def f():\n    value = 1\n    return value\n"
            modified = "def f():\n    # note\n    value = 1\n    return value\n"
            path.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py",
                tofile="b/sqlglot/example.py",
            )
            text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
            self.impact["functions"]["sqlglot/example.py"] = [[1, 4, [0]]]
            self.assertEqual(inspect_patch(text, root), frozenset())
            self.assertIn("a", select_tests(
                patch_changes(text), self.impact, self.tests, inspect_patch(text, root)
            ))

    def test_code_change_includes_source_readers(self):
        self.impact["source_reads"] = {"sqlglot/example.py": [1]}
        selected = select_tests({"sqlglot/example.py": {15}}, self.impact, self.tests)
        self.assertEqual(selected, {"a", "b", "unmapped"})

    def test_position_preserving_format_change_skips_covered_slow_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = "def f():\n    x= 1\n    return x\n"
            modified = "def f():\n    x =1\n    return x\n"
            path.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True), modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py", tofile="b/sqlglot/example.py",
            )
            text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
            self.impact["source_reads"] = {"sqlglot/example.py": [1]}
            selected = select_tests(patch_changes(text), self.impact, self.tests, inspect_patch(text, root))
            self.assertEqual(selected, {"b", "unmapped"})

    def test_inline_comment_skips_coverage_but_keeps_source_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = "def f():\n    return 1  # old\n"
            modified = "def f():\n    return 1  # explanation\n"
            path.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True), modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py", tofile="b/sqlglot/example.py",
            )
            text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
            self.impact["source_reads"] = {"sqlglot/example.py": [1]}
            self.impact["functions"]["sqlglot/example.py"] = [[1, 2, [0]]]
            selected = select_tests(
                patch_changes(text), self.impact, self.tests, inspect_patch(text, root),
            )
            self.assertEqual(selected, {"b", "unmapped"})
            self.impact["uncertain_source_reads"] = ["sqlglot/example.py"]
            self.assertEqual(
                select_tests(patch_changes(text), self.impact, self.tests, inspect_patch(text, root)),
                self.tests,
            )

    def test_class_literal_spelling_change_keeps_readers_but_value_change_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = "class C:\n    x = 0xA\n"
            path.write_text(original)
            self.impact["source_reads"] = {"sqlglot/example.py": [1]}
            for modified, expected in (
                ("class C:\n    x = 1_0\n", {"b", "unmapped"}),
                ("class C:\n    x = 0xB\n", self.tests),
            ):
                with self.subTest(modified=modified):
                    diff = difflib.unified_diff(
                        original.splitlines(keepends=True), modified.splitlines(keepends=True),
                        fromfile="a/sqlglot/example.py", tofile="b/sqlglot/example.py",
                    )
                    text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
                    result = select_tests(patch_changes(text), self.impact, self.tests, inspect_patch(text, root))
                    self.assertEqual(result, expected)

    def test_enum_identity_and_nullable_equality_are_recognized_as_equivalent(self):
        for original, modified, support in (
            (
                "from enum import Enum\n\n"
                "class Strategy(Enum):\n"
                "    UPPER = 1\n\n"
                "class C:\n"
                "    def check(self):\n"
                "        return self.strategy is Strategy.UPPER\n",
                "from enum import Enum\n\n"
                "class Strategy(Enum):\n"
                "    UPPER = 1\n\n"
                "class C:\n"
                "    def check(self):\n"
                "        return self.strategy == Strategy.UPPER\n",
                "",
            ),
            (
                "class C:\n"
                "    def __init__(self):\n"
                "        self.value: Table | None = None\n"
                "    def check(self):\n"
                "        return self.value is None\n",
                "class C:\n"
                "    def __init__(self):\n"
                "        self.value: Table | None = None\n"
                "    def check(self):\n"
                "        return self.value == None\n",
                "class Table:\n    pass\n",
            ),
        ):
            with self.subTest(original=original):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path = root / "sqlglot/example.py"
                    path.parent.mkdir()
                    path.write_text(original)
                    (root / "sqlglot/types.py").write_text(support)
                    diff = difflib.unified_diff(
                        original.splitlines(keepends=True),
                        modified.splitlines(keepends=True),
                        fromfile="a/sqlglot/example.py",
                        tofile="b/sqlglot/example.py",
                    )
                    text = (
                        "diff --git a/sqlglot/example.py b/sqlglot/example.py\n"
                        + "".join(diff)
                    )
                    self.assertEqual(
                        inspect_patch(text, root),
                        frozenset({"sqlglot/example.py"}),
                    )

    def test_nullable_equality_with_custom_eq_is_not_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = (
                "class C:\n"
                "    def __init__(self):\n"
                "        self.value: Table | None = None\n"
                "    def check(self):\n"
                "        return self.value is None\n"
            )
            modified = original.replace("is None", "== None")
            path.write_text(original)
            (root / "sqlglot/table.py").write_text(
                "class Table:\n"
                "    def __eq__(self, other):\n"
                "        return False\n"
            )
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py",
                tofile="b/sqlglot/example.py",
            )
            text = (
                "diff --git a/sqlglot/example.py b/sqlglot/example.py\n"
                + "".join(diff)
            )
            self.assertEqual(inspect_patch(text, root), frozenset())

    def test_enum_equality_with_custom_eq_is_not_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = (
                "from enum import Enum\n\n"
                "class Strategy(Enum):\n"
                "    UPPER = 1\n"
                "    def __eq__(self, other):\n"
                "        return False\n\n"
                "class C:\n"
                "    def check(self):\n"
                "        return self.strategy is Strategy.UPPER\n"
            )
            modified = original.replace("is Strategy", "== Strategy")
            path.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py",
                tofile="b/sqlglot/example.py",
            )
            text = (
                "diff --git a/sqlglot/example.py b/sqlglot/example.py\n"
                + "".join(diff)
            )
            self.assertEqual(inspect_patch(text, root), frozenset())

    def test_unattributed_source_read_requires_full_suite(self):
        self.impact["uncertain_source_reads"] = ["sqlglot/example.py"]
        selected = select_tests(
            {"sqlglot/example.py": {15}}, self.impact, self.tests, frozenset({"sqlglot/example.py"})
        )
        self.assertEqual(selected, self.tests)

    def test_uncertain_setting_keeps_class_and_dynamic_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = "class C:\n    OPTION = 1\n"
            modified = "class C:\n    OPTION = 2\n"
            path.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py",
                tofile="b/sqlglot/example.py",
            )
            text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
            impact = dict(self.impact)
            impact["classes"] = {"sqlglot/example.py": [[1, 2, [0]]]}
            dependency = {
                "settings": {"OPTION": [0]},
                "uncertain_settings": ["OPTION"],
                "dynamic_tests": [1],
                "dynamic_by_file": {"sqlglot/example.py": [1]},
                "inputs": {},
            }
            selected = select_tests(
                patch_changes(text),
                impact,
                self.tests,
                inspect_patch(text, root),
                dependency,
                text,
                root,
            )
            self.assertEqual(selected, {"a", "b", "unmapped"})

    def test_uncertain_setting_uses_owner_scoped_dynamic_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = "class C:\n    OPTION = 1\n"
            modified = "class C:\n    OPTION = 2\n"
            path.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py",
                tofile="b/sqlglot/example.py",
            )
            text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
            impact = dict(self.impact)
            impact["classes"] = {"sqlglot/example.py": [[1, 2, [0]]]}
            dependency = {
                "settings": {"OPTION": [0, 1]},
                "class_setting_definitions": {
                    "OPTION": [{"file": "sqlglot/example.py", "class": "C"}]
                },
                "owner_tests": {"sqlglot/example.py::C": [0]},
                "uncertain_settings": ["OPTION"],
                "dynamic_tests": [0, 1],
                "inputs": {},
                "scoped_inputs": {},
            }
            selected = select_tests(
                patch_changes(text),
                impact,
                self.tests,
                inspect_patch(text, root),
                dependency,
                text,
                root,
            )
            self.assertEqual(selected, {"a", "unmapped"})

    def test_static_patch_ignores_short_dictionary_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = 'class C:\n    OPTIONS = {"x": 1}\n'
            modified = 'class C:\n    OPTIONS = {"x": 2}\n'
            path.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py",
                tofile="b/sqlglot/example.py",
            )
            text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
            impact = dict(self.impact)
            impact["classes"] = {"sqlglot/example.py": [[1, 2, [0]]]}
            dependency = {
                "settings": {"OPTIONS": [0]},
                "class_setting_definitions": {
                    "OPTIONS": [{"file": "sqlglot/example.py", "class": "C"}]
                },
                "owner_tests": {"sqlglot/example.py::C": [0]},
                "uncertain_settings": [],
                "dynamic_tests": [],
                "inputs": {"x": [1]},
                "scoped_inputs": {"x": [1]},
            }
            selected = select_tests(
                patch_changes(text),
                impact,
                self.tests,
                inspect_patch(text, root),
                dependency,
                text,
                root,
            )
            self.assertEqual(selected, {"a", "unmapped"})

    def test_mixed_setting_code_and_fixture_patch_unions_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sqlglot/example.py"
            source.parent.mkdir()
            original = "class C:\n    OPTION = 1\n\ndef f():\n    return 1\n"
            modified = "class C:\n    OPTION = 2\n\ndef f():\n    return 2\n"
            source.write_text(original)
            fixture = root / "tests/fixtures/example.sql"
            fixture.parent.mkdir(parents=True)
            fixture.write_text("SELECT 1;\n")
            source_diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py",
                tofile="b/sqlglot/example.py",
            )
            fixture_diff = difflib.unified_diff(
                ["SELECT 1;\n"],
                ["SELECT 2;\n"],
                fromfile="a/tests/fixtures/example.sql",
                tofile="b/tests/fixtures/example.sql",
            )
            text = (
                "diff --git a/sqlglot/example.py b/sqlglot/example.py\n"
                + "".join(source_diff)
                + "diff --git a/tests/fixtures/example.sql b/tests/fixtures/example.sql\n"
                + "".join(fixture_diff)
            )
            impact = dict(self.impact)
            impact["functions"] = {"sqlglot/example.py": [[4, 5, [1]]]}
            impact["classes"] = {"sqlglot/example.py": [[1, 2, [0]]]}
            dependency = {
                "settings": {"OPTION": [0]},
                "uncertain_settings": [],
                "dynamic_tests": [1],
                "inputs": {},
            }
            selected = select_tests(
                patch_changes(text),
                impact,
                self.tests,
                inspect_patch(text, root),
                dependency,
                text,
                root,
            )
            self.assertEqual(selected, {"a", "b", "unmapped"})

    def test_module_assignment_and_dictionary_key_use_baseline_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sqlglot/example.py"
            source.parent.mkdir()
            original = "DEFAULT = 1\n"
            modified = "DEFAULT = 2\n"
            source.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py",
                tofile="b/sqlglot/example.py",
            )
            text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
            impact = dict(self.impact)
            impact["modules"] = {"sqlglot/example.py": [[1, 1, [1]]]}
            dependency = {
                "settings": {},
                "module_constants": {"DEFAULT": [1]},
                "uncertain_settings": [],
                "dynamic_tests": [],
                "inputs": {},
            }
            selected = select_tests(
                patch_changes(text), impact, self.tests, frozenset(), dependency, text, root
            )
            self.assertEqual(selected, {"b", "unmapped"})

            original = 'class C:\n    OPTIONS = {"constraint": False}\n'
            modified = 'class C:\n    OPTIONS = {"constraint": True}\n'
            source.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py",
                tofile="b/sqlglot/example.py",
            )
            text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
            dependency = {
                "settings": {},
                "uncertain_settings": [],
                "dynamic_tests": [],
                "inputs": {"constraint": [1]},
            }
            selected = select_tests(
                patch_changes(text), impact, self.tests, frozenset(), dependency, text, root
            )
            self.assertIn("b", selected)

    def test_static_setting_patch_eliminates_unrelated_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = "class C:\n    OPTION = 1\n"
            modified = "class C:\n    OPTION = 2\n"
            path.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True), modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py", tofile="b/sqlglot/example.py",
            )
            text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
            dependency = {
                "settings": {"OPTION": [0]},
                "uncertain_settings": [],
                "dynamic_tests": [],
                "inputs": {},
            }
            selected = select_tests(
                patch_changes(text),
                self.impact,
                self.tests,
                inspect_patch(text, root),
                dependency,
                text,
                root,
            )
            self.assertEqual(selected, {"a", "unmapped"})

    def test_attributed_dynamic_setting_reader_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = "class C:\n    OPTION = 1\n"
            modified = "class C:\n    OPTION = 2\n"
            path.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py",
                tofile="b/sqlglot/example.py",
            )
            text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
            dependency = {
                "settings": {"OPTION": [0]},
                "uncertain_settings": [],
                "dynamic_tests": [1],
                "dynamic_settings": {"OPTION": [1]},
                "inputs": {},
            }
            selected = select_tests(
                patch_changes(text),
                self.impact,
                self.tests,
                inspect_patch(text, root),
                dependency,
                text,
                root,
            )
            self.assertEqual(selected, {"a", "b", "unmapped"})

    def test_uncertain_static_setting_requires_full_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = "class C:\n    OPTION = 1\n"
            modified = "class C:\n    OPTION = 2\n"
            path.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True), modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py", tofile="b/sqlglot/example.py",
            )
            text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
            dependency = {
                "settings": {"OPTION": [0]},
                "uncertain_settings": ["OPTION"],
                "dynamic_tests": [],
                "inputs": {},
            }
            selected = select_tests(
                patch_changes(text),
                self.impact,
                self.tests,
                inspect_patch(text, root),
                dependency,
                text,
                root,
            )
            self.assertEqual(selected, self.tests)

    def test_categoriser_reports_eliminated_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sqlglot/example.py"
            path.parent.mkdir()
            original = "class C:\n    OPTION = 1\n"
            modified = "class C:\n    OPTION = 2\n"
            path.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True), modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py", tofile="b/sqlglot/example.py",
            )
            text = "diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff)
            dependency = {
                "settings": {"OPTION": [0]},
                "uncertain_settings": [],
                "dynamic_tests": [],
                "inputs": {},
            }
            result = categorize_patch(text, self.impact, self.tests, dependency, root)
            self.assertEqual(result["eliminated_tests"], 3)
            self.assertIn("b", result["eliminated"])
            self.assertIn("unrelated", result["eliminated"])
            self.assertIn("process", result["eliminated"])
            self.assertEqual(result["failure_implication"], "unproven")

    def test_canary_scores_prioritize_direct_statement_coverage(self):
        self.impact["statements"]["sqlglot/example.py"] = [[14, 15, [1]]]
        scores = canary_priority_scores({"sqlglot/example.py": {15}}, self.impact)
        self.assertEqual(scores["b"], 4)
        self.assertNotIn("a", scores)

    def test_canary_order_changes_only_order_within_workers(self):
        tests_by_number = {
            0: {"id": "a", "seconds": 0.5},
            1: {"id": "b", "seconds": 0.5},
            2: {"id": "unmapped", "seconds": 0.0},
        }
        workers = [[0, 1, 2]]
        ordered = prioritize_workers(workers, tests_by_number, {"b": 4})
        self.assertEqual(ordered[0][0], 1)
        self.assertCountEqual(ordered[0], workers[0])

    def test_worker_count_accounts_for_worker_cost(self):
        short = [{"number": 0, "seconds": 1.0}, {"number": 1, "seconds": 1.0}]
        long = [{"number": 0, "seconds": 10.0}, {"number": 1, "seconds": 10.0}]
        self.assertEqual(len(choose_workers(short)), 1)
        self.assertEqual(len(choose_workers(long)), 2)
        self.assertEqual(choose_workers([]), [])
        zero = [{"number": 0, "seconds": 0.0}, {"number": 1, "seconds": 0.0}]
        self.assertEqual(len(choose_workers(zero)), 1)

    def test_two_worker_split_can_improve_on_greedy_balancing(self):
        durations = [3, 3, 2, 2, 2]
        tests = [{"number": number, "seconds": seconds} for number, seconds in enumerate(durations)]
        workers = choose_workers(tests)
        cost = max(sum(durations[number] for number in worker) for worker in workers) + 2 * len(workers)
        self.assertEqual(cost, 10)
        self.assertCountEqual([number for worker in workers for number in worker], range(len(tests)))

    def test_multi_worker_search_improves_balancing_and_keeps_zero_duration_tests(self):
        durations = [9, 8, 6, 5, 4, 3, 3, 3, 3, 0, 0]
        tests = [{"number": number, "seconds": seconds} for number, seconds in enumerate(durations)]
        workers = choose_workers(tests)
        cost = max(sum(durations[number] for number in worker) for worker in workers) + 2 * len(workers)
        self.assertEqual(cost, 19)
        self.assertCountEqual([number for worker in workers for number in worker], range(len(tests)))

    def test_large_worker_split_is_balanced_and_preserves_every_test(self):
        tests = [
            {"number": number, "seconds": ((number * 37) % 7000 + 1000) / 1000}
            for number in range(127)
        ]
        workers = choose_workers(tests)
        durations = {test["number"]: round(test["seconds"] * 1000) for test in tests}
        loads = [sum(durations[number] for number in worker) for worker in workers]
        total = sum(durations.values())
        self.assertLessEqual(len(workers), 8)
        self.assertTrue(all(workers))
        self.assertCountEqual(
            [number for worker in workers for number in worker], range(len(tests))
        )
        self.assertLessEqual(max(loads), math.ceil(total / len(workers)) + max(durations.values()))

    def test_duplicate_numbers_and_invalid_durations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            choose_workers([
                {"number": 0, "seconds": 0.1},
                {"number": 0, "seconds": 0.2},
            ])
        for seconds in (-0.1, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(seconds=seconds):
                with self.assertRaisesRegex(ValueError, "duration"):
                    choose_workers([{"number": 0, "seconds": seconds}])
        with self.assertRaisesRegex(ValueError, "duplicate test ID"):
            validate_test_inventory([
                {"number": 0, "id": "same", "seconds": 0.1},
                {"number": 1, "id": "same", "seconds": 0.2},
            ])

    def test_stale_map_indexes_and_unmapped_runtime_tests_are_safe(self):
        impact = dict(self.impact)
        impact["tests"] = [*self.impact["tests"], "stale-map-test"]
        impact["process_tests"] = [2, 20]
        impact["functions"] = {"sqlglot/example.py": [[10, 20, [0, 20]]]}
        all_ids = {*self.tests, "new-runtime-test"}
        indexes = build_interval_indexes(impact)
        selected = select_tests(
            {"sqlglot/example.py": {15}},
            impact,
            all_ids,
            interval_indexes=indexes,
        )
        self.assertEqual(selected, {"a", "unmapped", "new-runtime-test"})

    def test_interval_index_uses_narrowest_nonempty_range(self):
        impact = {
            "tests": ["a", "b"],
            "functions": {"sqlglot/example.py": [[1, 20, [0, 1]]]},
            "classes": {"sqlglot/example.py": [[1, 20, [0, 1]]]},
            "modules": {"sqlglot/example.py": [[1, 30, [0, 1]]]},
            "statements": {"sqlglot/example.py": [[12, 12, [1]]]},
            "process_tests": [],
            "fixtures": {},
            "source_reads": {},
            "uncertain_source_reads": [],
            "metadata_tests": [],
        }
        selected = select_tests(
            {"sqlglot/example.py": {12}},
            impact,
            {"a", "b", "unmapped"},
            interval_indexes=build_interval_indexes(impact),
        )
        self.assertEqual(selected, {"b", "unmapped"})


if __name__ == "__main__":
    unittest.main()
