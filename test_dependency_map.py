import difflib
import tempfile
import unittest
from pathlib import Path

from dependency_map import (
    build_dependency_map, coverage_for_test, evidence_for_patch, evidence_for_test,
    matching_inputs, patch_reasons,
)
from selector import select_tests


class CoverageData:
    def __init__(self, path, contexts):
        self.path = str(path)
        self.contexts = contexts

    def measured_files(self):
        return {self.path}

    def contexts_by_lineno(self, filename):
        return self.contexts if filename == self.path else {}


class DependencyMapTests(unittest.TestCase):
    def test_records_setting_references_and_static_input_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sqlglot/example.py"
            source.parent.mkdir()
            source.write_text(
                "class C:\n    OPTION = 1\n\ndef use(value):\n    return value.OPTION\n"
                "\ndef dynamic(value, key):\n    return getattr(value, key)\n"
            )
            test = root / "tests/test_example.py"
            test.parent.mkdir()
            test.write_text('def test_a():\n    return "SELECT GROUPS FOR USER"\n')
            fixture = root / "tests/fixtures/example.sql"
            fixture.parent.mkdir()
            fixture.write_text("SHOW COLUMNS")
            impact = {
                "tests": ["tests/test_example.py::test_a", "tests/other.py::test_b"],
                "functions": {"sqlglot/example.py": [[4, 5, [0]], [7, 8, [1]]]},
                "fixtures": {"tests/fixtures/example.sql": [1]},
            }
            data = CoverageData(source, {5: ["test:a"], 8: [""]})
            result = build_dependency_map(root, impact, data)
            self.assertEqual(result["settings"]["OPTION"], [0])
            self.assertTrue(result["uncertain_dynamic"])
            self.assertEqual(result["dynamic_tests"], [1])
            self.assertEqual(matching_inputs(result["inputs"], "groups for user"), {0})
            self.assertEqual(matching_inputs(result["inputs"], "show columns"), {1})
            self.assertEqual(matching_inputs(result["inputs"], "not present"), set())
            self.assertEqual(evidence_for_test(result, 0)["setting_names"], ["OPTION"])
            self.assertTrue(evidence_for_test(result, 1)["dynamic_reference"])

    def test_unattributed_setting_reference_is_marked_uncertain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sqlglot/example.py"
            source.parent.mkdir()
            source.write_text("class C:\n    OPTION = 1\n\nvalue = C.OPTION\n")
            impact = {"tests": [], "functions": {}, "fixtures": {}}
            result = build_dependency_map(root, impact, CoverageData(source, {4: [""]}))
            self.assertIn("OPTION", result["uncertain_settings"])

    def test_builds_scoped_input_and_owner_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sqlglot/dialects/snowflake.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "class Snowflake:\n    class Tokenizer:\n        NUMERIC_ESCAPES = {}\n"
            )
            test = root / "tests/dialects/test_snowflake.py"
            test.parent.mkdir(parents=True)
            test.write_text(
                "class TestSnowflake:\n"
                "    dialect = \"snowflake\"\n\n"
                "    def test_escape(self):\n"
                "        version = (23, 12)\n"
                "        self.validate_identity(\"SELECT needle\")\n\n"
                "        self.validate_identity(r\"SELECT \\\\x41\")\n\n"
                "    def test_other(self):\n"
                "        self.validate_identity(\"SELECT 1\")\n"
            )
            test_ids = [
                "tests/dialects/test_snowflake.py::TestSnowflake::test_escape",
                "tests/dialects/test_snowflake.py::TestSnowflake::test_other",
            ]
            result = build_dependency_map(
                root,
                {"tests": test_ids, "functions": {}, "fixtures": {}},
                CoverageData(source, {}),
            )
            self.assertEqual(
                result["owner_tests"][
                    "sqlglot/dialects/snowflake.py::Snowflake.Tokenizer"
                ],
                [0, 1],
            )
            self.assertEqual(
                matching_inputs(result["scoped_inputs"], "select needle"),
                {0},
            )
            self.assertEqual(result["scoped_inputs"]["escape:x"], [0])
            self.assertEqual(result["scoped_inputs"]["literal:23"], [0])
            self.assertEqual(result["test_symbols"]["literal:12"], [0])

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            matching_inputs({}, "123")

    def test_test_profile_reports_coverage_and_source_readers(self):
        impact = {
            "functions": {"sqlglot/example.py": [[1, 5, [0]]]},
            "statements": {"sqlglot/example.py": [[3, 3, [0]]]},
            "fixtures": {"tests/fixtures/example.sql": [0]},
            "source_reads": {"sqlglot/example.py": [0]},
            "metadata_tests": [],
            "process_tests": [1],
        }
        profile = coverage_for_test(impact, 0)
        self.assertEqual(profile["covered_files"], [
            {"file": "sqlglot/example.py", "functions": 1, "statements": 1}
        ])
        self.assertEqual(profile["fixtures"], ["tests/fixtures/example.sql"])
        self.assertEqual(profile["source_reads"], ["sqlglot/example.py"])
        self.assertFalse(profile["untraced_process_fallback"])

    def test_patch_reasons_match_selector_for_coverage_and_fallbacks(self):
        impact = {
            "tests": ["a", "b", "c"],
            "functions": {"sqlglot/example.py": [[1, 5, [0, 1]]]},
            "statements": {"sqlglot/example.py": [[3, 3, [0]]]},
            "fixtures": {"tests/fixtures/example.sql": [1]},
            "source_reads": {"sqlglot/example.py": [1]},
            "uncertain_source_reads": [],
            "metadata_tests": [],
            "process_tests": [2],
        }
        for changes, unchanged in (
            ({"sqlglot/example.py": {3}}, frozenset()),
            ({"sqlglot/example.py": {3}}, frozenset({"sqlglot/example.py"})),
            ({"tests/fixtures/example.sql": {1}}, frozenset()),
            ({"sqlglot/unknown.py": {1}}, frozenset()),
            ({"sqlglot/example.py": {1}}, frozenset()),
            (None, frozenset()),
        ):
            expected = select_tests(changes, impact, set(impact["tests"]), unchanged)
            for number, name in enumerate(impact["tests"]):
                with self.subTest(changes=changes, unchanged=unchanged, test=name):
                    self.assertEqual(bool(patch_reasons(impact, number, changes, unchanged)), name in expected)

    def test_patch_profile_reflects_static_heuristic_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sqlglot/example.py"
            source.parent.mkdir()
            original = "class C:\n    OPTION = 1\n"
            modified = "class C:\n    OPTION = 2\n"
            source.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True), modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py", tofile="b/sqlglot/example.py",
            )
            patch = root / "change.patch"
            patch.write_text("diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff))
            impact = {
                "tests": ["candidate", "heavy"], "functions": {}, "statements": {},
                "fixtures": {}, "source_reads": {}, "uncertain_source_reads": [],
                "metadata_tests": [], "process_tests": [],
            }
            dependency = {
                "settings": {"OPTION": [0]}, "uncertain_settings": [],
                "dynamic_tests": [], "inputs": {},
            }
            self.assertTrue(evidence_for_patch(impact, 0, patch, dependency, root)["selected"])
            self.assertEqual(
                evidence_for_patch(impact, 1, patch, dependency, root),
                {"selected": False, "reasons": []},
            )


if __name__ == "__main__":
    unittest.main()
