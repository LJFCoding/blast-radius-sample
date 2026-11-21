import ast
import unittest
from collections import defaultdict
from unittest.mock import patch

from impact_collector import metadata_references, record_event


class ImpactCollectorTests(unittest.TestCase):
    def test_unattributed_source_open_is_not_silently_discarded(self):
        reads = defaultdict(set)
        uncertain = set()
        with patch("impact_collector.importing_source", return_value=False):
            record_event(
                "open", ("/src/sqlglot/example.py",), "", defaultdict(set), set(),
                reads, uncertain,
            )
        self.assertEqual(uncertain, {"sqlglot/example.py"})

    def test_import_reads_are_not_treated_as_source_inspection(self):
        uncertain = set()
        with patch("impact_collector.importing_source", return_value=True):
            record_event(
                "open", ("/src/sqlglot/example.py",), "", defaultdict(set), set(),
                defaultdict(set), uncertain,
            )
        self.assertEqual(uncertain, set())

    def test_metadata_in_one_test_does_not_select_unrelated_tests(self):
        source = """
class TestExample:
    def test_metadata(self):
        return function.__code__

    def test_other(self):
        return 1
"""
        tests = {
            "tests/example.py::TestExample::test_metadata[one]",
            "tests/example.py::TestExample::test_metadata[two::three]",
            "tests/example.py::TestExample::test_other",
        }
        self.assertEqual(
            metadata_references(ast.parse(source), tests),
            {test for test in tests if "test_metadata" in test},
        )

    def test_shared_or_indirect_metadata_access_keeps_the_whole_file(self):
        tests = {"tests/example.py::TestExample::test_a", "tests/example.py::TestExample::test_b"}
        for source in (
            "def helper(f):\n    return f.__code__\n",
            "class TestExample:\n"
            "    def test_a(self): return getattr(f, '__code__')\n"
            "    def test_b(self): return self.test_a()\n",
        ):
            with self.subTest(source=source):
                self.assertEqual(metadata_references(ast.parse(source), tests), tests)

    def test_nested_function_uses_its_containing_test(self):
        tests = {"tests/example.py::test_a", "tests/example.py::test_b"}
        source = "def test_a():\n    def test_b(): return f.__code__\n    return 1\n"
        self.assertEqual(metadata_references(ast.parse(source), tests), {"tests/example.py::test_a"})

    def test_indirect_test_call_without_metadata_does_not_select_tests(self):
        tests = {"tests/example.py::TestExample::test_a", "tests/example.py::TestExample::test_b"}
        source = "class TestExample:\n    def test_b(self): return self.test_a()\n"
        self.assertEqual(metadata_references(ast.parse(source), tests), set())


if __name__ == "__main__":
    unittest.main()
