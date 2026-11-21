"""Classify comment-containing lines in a patch against the original source."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Literal

SOURCE_ROOT = Path("/source")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DOLLAR_QUOTE = re.compile(r"\$(?:[a-zA-Z_]\w*)?\$")
ENCODING = re.compile(r"coding[:=][ \t]*[-\w.]+")
TOOL_DIRECTIVE = re.compile(r"#\s*(?:noqa|pylint:|ruff:|fmt:|pragma:)", re.IGNORECASE)
POSSIBLE_DIRECTIVE = re.compile(
    r"\b(?:coding|encoding|coverage|doctest|flake8|fmt|ignore|isort|lint|"
    r"mypy|noqa|nosec|noinspection|pragma|pylint|pyright|pytest|ruff|type)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CommentChange:
    path: str
    line: int
    action: Literal["added", "removed"]
    kind: str
    text: str


@dataclass
class FileChange:
    path: str
    original: str
    modified: str
    removed: dict[int, str]
    added: dict[int, str]
    impact_lines: set[int] = field(default_factory=set)
    mapping_supported: bool = True
    added_impact_lines: dict[int, int] = field(default_factory=dict)


@dataclass
class _PatchState:
    original: list[str]
    modified: list[str] = field(default_factory=list)
    removed: dict[int, str] = field(default_factory=dict)
    added: dict[int, str] = field(default_factory=dict)
    impact_lines: set[int] = field(default_factory=set)
    added_impact_lines: dict[int, int] = field(default_factory=dict)
    cursor: int = 0


_AST_PARSE_SECONDS = 0.0


@lru_cache(maxsize=128)
def _parse_python(source: str) -> ast.Module:
    global _AST_PARSE_SECONDS
    started = perf_counter()
    tree = ast.parse(source)
    _AST_PARSE_SECONDS += perf_counter() - started
    return tree


@lru_cache(maxsize=256)
def _read_source_cached(path: str, mtime_ns: int, size: int, inode: int) -> str:
    return Path(path).read_text(encoding="utf-8")


def _read_source(root: Path, name: str) -> str:
    path = _source_path(root, name)
    stat = path.stat()
    return _read_source_cached(str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino)


@lru_cache(maxsize=128)
def _ast_dump(source: str, include_attributes: bool = False) -> str:
    return ast.dump(_parse_python(source), include_attributes=include_attributes)


@lru_cache(maxsize=128)
def _encoding(source: str) -> str:
    return tokenize.detect_encoding(io.BytesIO(source.encode()).readline)[0]


def cache_statistics() -> dict[str, int | float]:
    return {
        "ast_parse_seconds": _AST_PARSE_SECONDS,
        "ast_hits": _parse_python.cache_info().hits,
        "ast_misses": _parse_python.cache_info().misses,
        "source_hits": _read_source_cached.cache_info().hits,
        "source_misses": _read_source_cached.cache_info().misses,
        "comment_hits": _python_comments.cache_info().hits + _sql_comments.cache_info().hits,
        "comment_misses": _python_comments.cache_info().misses + _sql_comments.cache_info().misses,
    }


def parse_python_source(source: str) -> ast.Module:
    return _parse_python(source)


def _header_path(line: str, prefix: str) -> str | None:
    name = line[4:].rstrip("\n").split("\t", 1)[0]
    if name == "/dev/null":
        return None
    if not name.startswith(prefix):
        raise ValueError(f"Unsupported patch path: {name!r}")
    return name[len(prefix):]


def _source_path(root: Path, name: str) -> Path:
    root = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Patch path escapes source root: {name!r}")
    return path


def _apply_hunk(lines: list[str], index: int, path: str, state: _PatchState) -> int:
    match = HUNK.match(lines[index])
    if match is None:
        raise ValueError(f"Invalid hunk header for {path}")
    old_start = int(match.group(1))
    old_count = int(match.group(2) or 1)
    new_start = int(match.group(3))
    new_count = int(match.group(4) or 1)
    start = old_start - 1 if old_count else old_start
    if start < state.cursor or start > len(state.original):
        raise ValueError(f"Hunk position is invalid for {path}")
    state.modified.extend(state.original[state.cursor:start])
    state.cursor = start
    expected_new = new_start - 1 if new_count else new_start
    if len(state.modified) != expected_new:
        raise ValueError(f"New hunk position is invalid for {path}")

    index += 1
    old_seen = new_seen = 0
    old_line = old_start
    deleted_lines: list[int] = []
    added_count = 0
    while old_seen < old_count or new_seen < new_count:
        if index >= len(lines) or not lines[index].startswith((" ", "-", "+")):
            raise ValueError(f"Incomplete hunk for {path}")
        marker, content = lines[index][0], lines[index][1:]
        index += 1
        if index < len(lines) and lines[index].startswith("\\ No newline at end of file"):
            content = content.removesuffix("\n")
            index += 1
        if marker in {" ", "-"}:
            if state.cursor >= len(state.original) or state.original[state.cursor] != content:
                raise ValueError(f"Patch does not match original file: {path}:{state.cursor + 1}")
            if marker == "-":
                state.removed[state.cursor + 1] = content.rstrip("\n")
                state.impact_lines.add(old_line)
                deleted_lines.append(old_line)
            state.cursor += 1
            old_seen += 1
            old_line += 1
        if marker in {" ", "+"}:
            state.modified.append(content)
            if marker == "+":
                impact_line = (
                    deleted_lines[added_count]
                    if added_count < len(deleted_lines)
                    else deleted_lines[-1]
                    if deleted_lines
                    else old_line
                )
                state.added[len(state.modified)] = content.rstrip("\n")
                state.impact_lines.add(impact_line)
                state.added_impact_lines[len(state.modified)] = impact_line
                added_count += 1
            new_seen += 1
        if marker == " ":
            deleted_lines = []
            added_count = 0
        if old_seen > old_count or new_seen > new_count:
            raise ValueError(f"Incorrect hunk length for {path}")
    return index


def _apply_file(lines: list[str], index: int, root: Path) -> tuple[FileChange, int]:
    old_path = _header_path(lines[index], "a/")
    index += 1
    if index >= len(lines) or not lines[index].startswith("+++ "):
        raise ValueError("Missing new-file header")
    new_path = _header_path(lines[index], "b/")
    index += 1
    path = new_path or old_path
    if path is None:
        raise ValueError("Patch has no file path")
    if old_path and new_path and old_path != new_path:
        raise ValueError("Renamed files are not supported")

    original = _read_source(root, old_path) if old_path else ""
    state = _PatchState(original.splitlines(keepends=True))
    hunks = 0
    while index < len(lines) and not lines[index].startswith("diff --git "):
        index = _apply_hunk(lines, index, path, state)
        hunks += 1
    if not hunks:
        raise ValueError(f"No text hunks for {path}")
    state.modified.extend(state.original[state.cursor:])
    if new_path is None and state.modified:
        raise ValueError(f"Deleted file has remaining content: {path}")
    return FileChange(
        path,
        original,
        "".join(state.modified),
        state.removed,
        state.added,
        state.impact_lines,
        old_path is not None,
        state.added_impact_lines,
    ), index


def file_changes(patch: str, root: Path = SOURCE_ROOT) -> list[FileChange]:
    """Reconstruct changed files from a unified diff and a pinned source tree."""
    lines = patch.splitlines(keepends=True)
    changes = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("diff --git "):
            raise ValueError("Expected a git diff header")
        index += 1
        while index < len(lines) and not lines[index].startswith(("--- ", "diff --git ")):
            if lines[index].startswith(("Binary files ", "GIT binary patch")):
                raise ValueError("Binary patches are not supported")
            index += 1
        if index >= len(lines) or not lines[index].startswith("--- "):
            raise ValueError("Patch has no text-file header")
        change, index = _apply_file(lines, index, root)
        changes.append(change)
    if not changes:
        raise ValueError("Patch is empty")
    return changes


def _docstring_lines(tree: ast.AST) -> set[int]:
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    if value.end_lineno is None:
                        raise ValueError("Docstring has no ending line")
                    lines.update(range(value.lineno, value.end_lineno + 1))
    return lines


def has_docstring_change(
    change: FileChange, original: ast.AST | None = None, modified: ast.AST | None = None,
) -> bool:
    return any(
        _docstring_lines(tree) & changed.keys()
        for tree, changed in (
            (original if original is not None else _parse_python(change.original), change.removed),
            (modified if modified is not None else _parse_python(change.modified), change.added),
        )
    )


@lru_cache(maxsize=128)
def _python_comments(source: str) -> dict[int, set[str]]:
    comments: dict[int, set[str]] = {
        line: {"docstring"} for line in _docstring_lines(_parse_python(source))
    }
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            text = token.string
            if token.start[0] == 1 and text.startswith("#!"):
                kind = "python_shebang"
            elif token.start[0] <= 2 and ENCODING.search(text):
                kind = "python_encoding_directive"
            elif re.match(r"#\s*type\s*:", text):
                kind = "python_type_directive"
            elif TOOL_DIRECTIVE.match(text):
                kind = "python_tool_directive"
            else:
                kind = "python_comment"
            comments.setdefault(token.start[0], set()).add(kind)
    return comments


def is_plain_comment_change(
    change: FileChange, original: ast.AST | None = None, modified: ast.AST | None = None,
) -> bool:
    """Identify standalone ordinary comments without treating them as necessarily safe."""
    versions = (
        (change.original, change.removed),
        (change.modified, change.added),
    )
    if not change.removed and not change.added:
        return False
    for source, lines in versions:
        comments = _python_comments(source)
        for line, text in lines.items():
            if (
                not text.lstrip().startswith("#")
                or comments.get(line) != {"python_comment"}
                or POSSIBLE_DIRECTIVE.search(text)
            ):
                return False
    new_encoding = _encoding(change.modified)
    return (
        _encoding(change.original) == new_encoding
        and _ast_dump(change.original) == _ast_dump(change.modified)
    )


def is_inline_comment_change(
    change: FileChange, original: ast.AST | None = None, modified: ast.AST | None = None,
) -> bool:
    """Identify same-line edits limited to ordinary trailing comments."""
    if not change.removed or change.removed.keys() != change.added.keys():
        return False
    if not any("#" in text for text in (*change.removed.values(), *change.added.values())):
        return False

    def comments(source: str) -> dict[int, tokenize.TokenInfo]:
        return {
            token.start[0]: token
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        }

    old_comments = comments(change.original)
    new_comments = comments(change.modified)
    old_kinds = _python_comments(change.original)
    new_kinds = _python_comments(change.modified)
    for line, old_text in change.removed.items():
        new_text = change.added[line]
        old_comment, new_comment = old_comments.get(line), new_comments.get(line)
        if old_comment is None and new_comment is None:
            return False
        for token, kinds in ((old_comment, old_kinds), (new_comment, new_kinds)):
            if token and (
                kinds.get(line) != {"python_comment"} or POSSIBLE_DIRECTIVE.search(token.string)
            ):
                return False
        old_code = old_text[:old_comment.start[1]] if old_comment else old_text
        new_code = new_text[:new_comment.start[1]] if new_comment else new_text
        if old_code.rstrip() != new_code.rstrip():
            return False

    return (
        _encoding(change.original) == _encoding(change.modified)
        and _ast_dump(change.original, include_attributes=True)
        == _ast_dump(change.modified, include_attributes=True)
    )


def is_formatting_only(
    change: FileChange, original: ast.AST | None = None, modified: ast.AST | None = None,
) -> bool:
    """Check that executable syntax, encoding and comment positions are unchanged."""
    def comparable_lines(before: str, after: str) -> bool:
        if "".join(before.split()) == "".join(after.split()):
            return True
        try:
            return _ast_dump(before.lstrip(), include_attributes=True) == _ast_dump(
                after.lstrip(), include_attributes=True
            )
        except SyntaxError:
            return False

    if change.removed.keys() != change.added.keys() or any(
        len(change.removed[line]) != len(change.added[line])
        or not comparable_lines(change.removed[line], change.added[line])
        for line in change.removed
    ):
        return False

    def comments(source: str) -> list[tuple[str, tuple[int, int]]]:
        return [
            (token.string, token.start)
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.COMMENT
        ]

    old_encoding = _encoding(change.original)
    new_encoding = _encoding(change.modified)
    return (
        old_encoding == new_encoding
        and _ast_dump(change.original, include_attributes=True)
        == _ast_dump(change.modified, include_attributes=True)
        and comments(change.original) == comments(change.modified)
    )


def _skip_sql_block(source: str, index: int) -> int:
    index += 2
    depth = 1
    while index < len(source) and depth:
        if source.startswith("/*", index):
            depth += 1
            index += 2
        elif source.startswith("*/", index):
            depth -= 1
            index += 2
        else:
            index += 1
    if depth:
        raise ValueError("Unterminated SQL block comment")
    return index


def _skip_sql_quoted(source: str, index: int) -> int:
    closing = "]" if source[index] == "[" else source[index]
    index += 1
    while index < len(source):
        if source[index] == "\\" and index + 1 < len(source):
            index += 2
        elif source[index] == closing:
            index += 1
            if index < len(source) and source[index] == closing:
                index += 1
            else:
                return index
        else:
            index += 1
    raise ValueError("Unterminated SQL quoted value")


@lru_cache(maxsize=128)
def _sql_comments(source: str) -> dict[int, set[str]]:
    comments: dict[int, set[str]] = {}
    index = 0
    line = 1
    while index < len(source):
        start = index
        char = source[index]
        if source.startswith("--", index):
            end = source.find("\n", index)
            index = len(source) if end < 0 else end
            kind = (
                "sql_line_comment"
                if start + 2 == len(source) or source[start + 2].isspace()
                else "sql_dash_or_operator"
            )
            comments.setdefault(line, set()).add(kind)
        elif source.startswith("/*", index):
            if source.startswith("/*+", index):
                kind = "sql_optimizer_hint"
            elif source.startswith(("/*!", "/*M!"), index):
                kind = "sql_executable_comment"
            else:
                kind = "sql_block_comment"
            index = _skip_sql_block(source, index)
            for comment_line in range(line, line + source.count("\n", start, index) + 1):
                comments.setdefault(comment_line, set()).add(kind)
        elif char == "#":
            comments.setdefault(line, set()).add("sql_hash_or_directive")
            if not source[source.rfind("\n", 0, index) + 1:index].strip():
                end = source.find("\n", index)
                index = len(source) if end < 0 else end
            else:
                index += 1
        elif char in "'\"`[":
            index = _skip_sql_quoted(source, index)
        elif char == "$" and (match := DOLLAR_QUOTE.match(source, index)):
            end = source.find(match.group(), match.end())
            if end < 0:
                raise ValueError("Unterminated SQL dollar-quoted value")
            index = end + len(match.group())
        else:
            index += 1
        line += source.count("\n", start, index)
    return comments


def commentry_check(patch: str, source_root: Path = SOURCE_ROOT) -> list[CommentChange]:
    """Return added/removed lines containing comments or docstrings.

    A reported line may also contain executable code. SQL ``#`` lines are
    ambiguous: they can be comments or fixture metadata.
    """
    results = []
    for change in file_changes(patch, source_root):
        suffix = Path(change.path).suffix
        if suffix == ".py":
            classify = _python_comments
        elif suffix == ".sql":
            classify = _sql_comments
        else:
            raise ValueError(f"Unsupported file type: {change.path}")
        versions: tuple[tuple[Literal["removed", "added"], str, dict[int, str]], ...] = (
            ("removed", change.original, change.removed),
            ("added", change.modified, change.added),
        )
        for action, source, changed in versions:
            comments = classify(source)
            for line, text in changed.items():
                for kind in sorted(comments.get(line, ())):
                    results.append(CommentChange(change.path, line, action, kind, text))
    return results
