import ast
import unittest

from statement_map import compact_test_ids, statement_ranges


class StatementMapTests(unittest.TestCase):
    def test_function_header_is_mapped_without_a_docstring_statement(self):
        source = 'def f(\n    arg=1,\n):\n    """Documentation."""\n    return arg\n'
        self.assertEqual(statement_ranges(ast.parse(source)), {(1, 3), (5, 5)})

    def test_condition_and_body_are_mapped_separately(self):
        source = "def f(x):\n    if (\n        x\n    ):\n        return 1\n    return 2\n"
        self.assertEqual(statement_ranges(ast.parse(source)), {(1, 1), (2, 3), (5, 5), (6, 6)})

    def test_class_level_lambda_header_is_not_a_runtime_statement(self):
        source = "class C:\n    fn = lambda x: (\n        x + 1\n    )\n"
        self.assertEqual(statement_ranges(ast.parse(source)), {(1, 1), (2, 3), (2, 4)})

    def test_module_and_nested_class_assignments_are_mapped(self):
        source = (
            "VALUE = 1\n"
            "@decorate(\n    int,\n)\n"
            "class Outer:\n"
            "    OPTIONS: dict[str, int] = {\n"
            "        'key': 1,\n"
            "    }\n"
            "    class Inner:\n"
            "        VALUE = 2\n"
        )
        ranges = statement_ranges(ast.parse(source))
        self.assertIn((1, 1), ranges)
        self.assertIn((2, 5), ranges)
        self.assertIn((6, 8), ranges)
        self.assertIn((9, 9), ranges)
        self.assertIn((10, 10), ranges)

    def test_compacted_map_preserves_test_references(self):
        impact = {
            "tests": ["test_a", "test_b"],
            "process_tests": ["test_b"],
            "metadata_tests": ["test_a"],
            "fixtures": {"fixture.sql": ["test_b"]},
            "source_reads": {"source.py": ["test_a"]},
            "member_reads": {"source.py": [["function", 2, 3, ["test_b"]]]},
            "functions": {"source.py": [[1, 4, ["test_a", "test_b"]]]},
            "statements": {"source.py": [[2, 2, ["test_b"]]]},
            "classes": {"source.py": [[1, 4, ["test_a"]]]},
            "modules": {"source.py": [[1, 4, ["test_b"]]]},
        }
        compact_test_ids(impact)
        self.assertEqual(impact["functions"]["source.py"][0][2], [0, 1])
        self.assertEqual(impact["statements"]["source.py"][0][2], [1])
        self.assertEqual(impact["classes"]["source.py"][0][2], [0])
        self.assertEqual(impact["modules"]["source.py"][0][2], [1])
        self.assertEqual(impact["fixtures"]["fixture.sql"], [1])
        self.assertEqual(impact["process_tests"], [1])
        self.assertEqual(impact["member_reads"]["source.py"][0][3], [1])


if __name__ == "__main__":
    unittest.main()
