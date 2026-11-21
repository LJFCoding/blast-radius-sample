import difflib
import tempfile
import unittest
from pathlib import Path

from commentry import (
    cache_statistics, commentry_check, file_changes, has_docstring_change,
    is_formatting_only, is_inline_comment_change, is_plain_comment_change,
    parse_python_source,
)


def make_patch(path: str, original: str, modified: str) -> str:
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return f"diff --git a/{path} b/{path}\n" + "".join(diff)


class CommentryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def check(self, path: str, original: str, modified: str):
        file = self.root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(original)
        return commentry_check(make_patch(path, original, modified), self.root)

    def test_python_comment_and_docstring_changes(self):
        original = '"""Old description."""\nvalue = "# in a string"\n# Old comment\n'
        modified = '"""New description."""\nvalue = "# in a string"\n# New comment\n'
        changes = self.check("sample.py", original, modified)
        self.assertEqual(
            {(change.action, change.kind, change.line) for change in changes},
            {
                ("removed", "docstring", 1),
                ("added", "docstring", 1),
                ("removed", "python_comment", 3),
                ("added", "python_comment", 3),
            },
        )

    def test_multiline_docstring_and_non_docstring(self):
        content = "".join(f"    line {number}\n" for number in range(20))
        original = 'def f():\n    """Start\n' + content + '    """\n    return 1\n'
        modified = original.replace("    line 10\n", "    changed line\n")
        changes = self.check("sample.py", original, modified)
        self.assertEqual([change.kind for change in changes], ["docstring", "docstring"])

        original = 'value = """Start\nnot a docstring\n"""\n'
        modified = original.replace("not a docstring", "still not a docstring")
        self.assertEqual(self.check("other.py", original, modified), [])

    def test_comment_like_strings_are_not_comments(self):
        original = "value = '# old'\nnumber = 1  # retained\n"
        modified = "value = '# new'\nnumber = 2  # retained\n"
        changes = self.check("sample.py", original, modified)
        self.assertEqual(
            [(change.action, change.kind, change.line) for change in changes],
            [("removed", "python_comment", 2), ("added", "python_comment", 2)],
        )

    def test_python_directives_are_distinguished_from_plain_comments(self):
        original = "value = 1\n"
        modified = "#!/usr/bin/env python\n# coding: utf-8\nvalue = 1  # type: ignore\n"
        changes = self.check("sample.py", original, modified)
        self.assertEqual(
            [change.kind for change in changes],
            ["python_shebang", "python_encoding_directive", "python_type_directive"],
        )

    def test_sql_line_block_and_ambiguous_hash(self):
        original = "SELECT '-- not a comment';\n-- old\n/* old\n  block */\n# dialect: presto\n"
        modified = "SELECT '-- not a comment';\n-- new\n/* new\n  block */\n# dialect: trino\n"
        changes = self.check("fixture.sql", original, modified)
        self.assertEqual(
            [(change.action, change.kind, change.line) for change in changes],
            [
                ("removed", "sql_line_comment", 2),
                ("removed", "sql_block_comment", 3),
                ("removed", "sql_hash_or_directive", 5),
                ("added", "sql_line_comment", 2),
                ("added", "sql_block_comment", 3),
                ("added", "sql_hash_or_directive", 5),
            ],
        )

    def test_sql_hints_and_executable_comments_are_distinct(self):
        original = "SELECT 1;\n"
        modified = "SELECT /*+ INDEX(t) */ 1; /*! special */\n"
        changes = self.check("fixture.sql", original, modified)
        self.assertEqual(
            {change.kind for change in changes},
            {"sql_optimizer_hint", "sql_executable_comment"},
        )

    def test_insertions_and_deletions_at_file_boundaries(self):
        changes = self.check("sample.py", "value = 1\n", "# beginning\nvalue = 1\n# end\n")
        self.assertEqual(
            [(change.action, change.line) for change in changes],
            [("added", 1), ("added", 3)],
        )
        changes = self.check("sample.py", "# only comment\n", "")
        self.assertEqual([(change.action, change.line) for change in changes], [("removed", 1)])

    def test_original_file_without_final_newline(self):
        (self.root / "sample.py").write_text("value = 1")
        patch = (
            "diff --git a/sample.py b/sample.py\n"
            "--- a/sample.py\n"
            "+++ b/sample.py\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "\\ No newline at end of file\n"
            "+# comment\n"
        )
        changes = commentry_check(patch, self.root)
        self.assertEqual([(change.action, change.kind) for change in changes], [
            ("added", "python_comment")
        ])

    def test_only_standalone_non_directive_comments_are_eligible(self):
        file = self.root / "sample.py"
        for original, modified, expected in (
            ("value = 1\n", "# helpful note\nvalue = 1\n", True),
            ("value = 1\n", "value = 1  # helpful note\n", False),
            ("value = 1\n", "# type: ignore\nvalue = 1\n", False),
            ('"""old"""\n', '"""new"""\n', False),
            ("# header\n# coding: latin-1\nvalue = 1\n",
             "# extra\n# header\n# coding: latin-1\nvalue = 1\n", False),
        ):
            with self.subTest(modified=modified):
                file.write_text(original)
                change = file_changes(make_patch("sample.py", original, modified), self.root)[0]
                self.assertEqual(is_plain_comment_change(change), expected)

    def test_docstring_change_is_detected_separately_from_string_literals(self):
        file = self.root / "sample.py"
        for original, modified, expected in (
            ('"""old"""\n', '"""new"""\n', True),
            ('value = "old"\n', 'value = "new"\n', False),
        ):
            with self.subTest(modified=modified):
                file.write_text(original)
                change = file_changes(make_patch("sample.py", original, modified), self.root)[0]
                self.assertEqual(has_docstring_change(change), expected)

    def test_inline_comments_require_unchanged_code_and_no_directives(self):
        file = self.root / "sample.py"
        for original, modified, expected in (
            ("x = 1  # old\n", "x = 1  # new\n", True),
            ("x = 1\n", "x = 1  # explanation\n", True),
            ("x = 1  # explanation\n", "x = 1\n", True),
            ("x = 1  # old\n", "x = 2  # new\n", False),
            ("x = 1\n", "x = 1  # type: ignore\n", False),
            ("x = 1  # noqa\n", "x = 1  # explanation\n", False),
            ("x = '# old'\n", "x = '# new'\n", False),
        ):
            with self.subTest(original=original, modified=modified):
                file.write_text(original)
                change = file_changes(make_patch("sample.py", original, modified), self.root)[0]
                self.assertEqual(is_inline_comment_change(change), expected)

    def test_formatting_requires_unchanged_syntax_positions_and_comments(self):
        file = self.root / "sample.py"
        for original, modified, expected in (
            ("x= 1\n", "x =1\n", True),
            ("class C:\n    x = 0xA\n", "class C:\n    x = 1_0\n", True),
            ("x = 1\n", "x = 2\n", False),
            ("x = 1\n", "x  = 1\n", False),
            ("x = 1  # old\n", "x = 1  # new\n", False),
            ("x= 1 # type: ignore\n", "x =1 # type: ignore\n", True),
        ):
            with self.subTest(modified=modified):
                file.write_text(original)
                change = file_changes(make_patch("sample.py", original, modified), self.root)[0]
                self.assertEqual(is_formatting_only(change), expected)

    def test_source_mismatch_and_unsupported_files_raise(self):
        file = self.root / "sample.py"
        file.write_text("different = 1\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            commentry_check(make_patch("sample.py", "value = 1\n", "# comment\n"), self.root)
        file = self.root / "sample.txt"
        file.write_text("old\n")
        with self.assertRaisesRegex(ValueError, "Unsupported file type"):
            commentry_check(make_patch("sample.txt", "old\n", "new\n"), self.root)

    def test_patch_paths_cannot_escape_or_rename_source_files(self):
        with self.assertRaisesRegex(ValueError, "escapes source root"):
            file_changes(make_patch("../outside.py", "old = 1\n", "old = 2\n"), self.root)
        patch = (
            "diff --git a/old.py b/new.py\n"
            "--- a/old.py\n"
            "+++ b/new.py\n"
            "@@ -1 +1 @@\n"
            "-old = 1\n"
            "+old = 2\n"
        )
        with self.assertRaisesRegex(ValueError, "Renamed files"):
            file_changes(patch, self.root)

    def test_ast_source_and_comment_classifications_are_cached(self):
        source = "value = 1  # note\n"
        parse_python_source(source)
        parse_python_source(source)
        path = self.root / "sample.py"
        path.write_text(source)
        patch = make_patch("sample.py", source, "value = 2  # note\n")
        before = cache_statistics()
        commentry_check(patch, self.root)
        commentry_check(patch, self.root)
        after = cache_statistics()
        self.assertGreater(after["ast_hits"], before["ast_hits"])
        self.assertGreater(after["source_hits"], before["source_hits"])
        self.assertGreater(after["comment_hits"], before["comment_hits"])


if __name__ == "__main__":
    unittest.main()
