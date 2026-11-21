import difflib
import tempfile
import unittest
from pathlib import Path

from canary import (
    _patch_context, apply_test_dominance, build_dominance_map,
    candidate_order_scores, patch_dominance_edges, structural_dominance_groups,
)


class CanaryPatchTests(unittest.TestCase):
    def test_patch_context_respects_inline_comment_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sqlglot/example.py"
            source.parent.mkdir()
            original = "def f():\n    return 1  # old\n"
            modified = "def f():\n    return 1  # new\n"
            source.write_text(original)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True), modified.splitlines(keepends=True),
                fromfile="a/sqlglot/example.py", tofile="b/sqlglot/example.py",
            )
            patch = root / "change.patch"
            patch.write_text("diff --git a/sqlglot/example.py b/sqlglot/example.py\n" + "".join(diff))
            baseline = {
                "seconds": {"heavy": 20.0, "source_reader": 1.0},
                "source_root": root,
                "dependency": None,
                "impact": {
                    "tests": ["heavy", "source_reader"],
                    "functions": {"sqlglot/example.py": [[1, 2, [0]]]},
                    "statements": {"sqlglot/example.py": [[2, 2, [0]]]},
                    "fixtures": {},
                    "source_reads": {"sqlglot/example.py": [1]},
                    "uncertain_source_reads": [],
                    "metadata_tests": [],
                    "process_tests": [],
                },
            }
            _, selected, fallback = _patch_context(patch, baseline)
            self.assertEqual(selected, {"source_reader"})
            self.assertFalse(fallback)

    def test_structural_equivalence_creates_safe_patch_dominance_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tests/test_duplicate.py"
            source.parent.mkdir()
            source.write_text(
                "class TestDuplicate:\n"
                "    def test_fast(self):\n"
                "        assert transform(1) == 2\n"
                "\n"
                "    def test_slow(self):\n"
                "        assert transform(1) == 2\n"
            )
            fast = "tests/test_duplicate.py::TestDuplicate::test_fast"
            slow = "tests/test_duplicate.py::TestDuplicate::test_slow"
            impact = {
                "tests": [fast, slow],
                "functions": {
                    "sqlglot/example.py": [[1, 10, [0, 1]]],
                },
                "statements": {
                    "sqlglot/example.py": [[4, 4, [0, 1]]],
                },
                "classes": {},
                "modules": {},
                "fixtures": {},
                "source_reads": {},
                "process_tests": [],
                "metadata_tests": [],
            }

            groups = structural_dominance_groups(impact, root)
            self.assertEqual(len(groups), 1)
            self.assertEqual(set(groups[0]["members"]), {fast, slow})
            self.assertTrue(groups[0]["safe_for_selection"])
            dominance = build_dominance_map(impact, root)
            selected, edges = apply_test_dominance(
                {fast, slow},
                {fast: 0.01, slow: 20.0},
                dominance,
            )
            self.assertEqual(selected, {fast})
            self.assertEqual(len(edges), 1)
            self.assertEqual(edges[0]["heavy"], slow)
            self.assertEqual(
                edges[0]["implication"],
                "heavy_fails_implies_canary_fails",
            )
            patch_selected, patch_edges = patch_dominance_edges(
                {"seconds": {fast: 0.01, slow: 20.0}},
                {fast, slow},
                dominance,
            )
            self.assertEqual(patch_selected, {fast})
            self.assertTrue(patch_edges[0]["patch_aware"])
            self.assertEqual(patch_edges[0]["canary_seconds"], 0.01)

    def test_unproven_overlap_cannot_change_selected_tests(self):
        dominance = {
            "schema": "baseline-canary-candidates-v3",
            "groups": [],
            "proof_version": 1,
        }
        with self.assertRaisesRegex(ValueError, "schema"):
            apply_test_dominance(
                {"canary", "heavy"},
                {"canary": 0.01, "heavy": 20.0},
                dominance,
            )

    def test_dominance_map_rejects_unsupported_proof(self):
        dominance = {
            "schema": "structural-test-dominance-v1",
            "proof_version": 1,
            "groups": [{
                "members": ["canary", "heavy"],
                "proof_kind": "observed_mutation_similarity",
                "scope": "isolated-fixture-free-test-methods",
                "safe_for_selection": True,
                "implication": "heavy_fails_implies_canary_fails",
                "proof": {
                    "file": "tests/test_example.py",
                    "classes": ["Example"],
                    "same_fixture_free_function_shape": True,
                    "same_baseline_coverage": True,
                },
            }],
            "group_count": 1,
            "dominated_test_count": 1,
        }
        with self.assertRaisesRegex(ValueError, "proof"):
            apply_test_dominance(
                {"canary", "heavy"},
                {"canary": 0.01, "heavy": 20.0},
                dominance,
            )

    def test_structural_dominance_requires_matching_class_trace_and_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tests/test_duplicate.py"
            source.parent.mkdir()
            source.write_text(
                "class TestDuplicate:\n"
                "    def test_one(self):\n"
                "        assert transform(1) == 2\n"
                "\n"
                "    def test_two(self, request):\n"
                "        assert transform(1) == 2\n"
            )
            impact = {
                "tests": [
                    "tests/test_duplicate.py::TestDuplicate::test_one",
                    "tests/test_duplicate.py::TestDuplicate::test_two",
                ],
                "functions": {"sqlglot/example.py": [[1, 10, [0, 1]]]},
                "statements": {},
                "classes": {},
                "modules": {},
                "fixtures": {},
                "source_reads": {},
                "process_tests": [],
                "metadata_tests": [],
            }
            self.assertEqual(structural_dominance_groups(impact, root), [])

    def test_dominance_map_does_not_remove_an_unpaired_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tests/test_duplicate.py"
            source.parent.mkdir()
            source.write_text(
                "class TestDuplicate:\n"
                "    def test_fast(self):\n"
                "        assert transform(1) == 2\n"
                "\n"
                "    def test_slow(self):\n"
                "        assert transform(1) == 2\n"
            )
            fast = "tests/test_duplicate.py::TestDuplicate::test_fast"
            slow = "tests/test_duplicate.py::TestDuplicate::test_slow"
            impact = {
                "tests": [fast, slow],
                "functions": {"sqlglot/example.py": [[1, 10, [0, 1]]]},
                "statements": {},
                "classes": {},
                "modules": {},
                "fixtures": {},
                "source_reads": {},
                "process_tests": [],
                "metadata_tests": [],
            }
            selected, edges = apply_test_dominance(
                {slow},
                {fast: 0.01, slow: 20.0},
                build_dominance_map(impact, root),
            )
            self.assertEqual(selected, {slow})
            self.assertEqual(edges, [])

    def test_candidate_order_scores_never_promote_candidates_to_proofs(self):
        unit = ("sqlglot/example.py", 10, 10)
        canary = {
            "statements": {unit},
            "functions": set(),
            "fixtures": set(),
            "source_reads": set(),
            "assertions": set(),
            "parameters": set(),
            "helpers": set(),
            "bodies": set(),
        }
        heavy = {
            key: set(value)
            for key, value in canary.items()
        }
        scores = candidate_order_scores(
            {
                "seconds": {"canary": 0.01, "heavy": 20.0},
                "profiles": {"canary": canary, "heavy": heavy},
            },
            {"canary", "heavy"},
            {"sqlglot/example.py": {10}},
        )
        self.assertGreater(scores["canary"], 0)


if __name__ == "__main__":
    unittest.main()
