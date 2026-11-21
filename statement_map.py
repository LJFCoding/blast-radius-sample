"""Build per-statement coverage from the recorded baseline test contexts."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

SIMPLE_STATEMENTS = (
    ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Raise,
    ast.Assert, ast.Expr, ast.Delete, ast.Break, ast.Continue, ast.Pass,
)


def statement_ranges(tree: ast.AST) -> set[tuple[int, int]]:
    ranges = set()

    def add(start: int, end: int | None) -> None:
        if end is None:
            raise ValueError("Statement has no ending line")
        if start <= end:
            ranges.add((start, end))

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorators = getattr(node, "decorator_list", ())
            start = min((decorator.lineno for decorator in decorators), default=node.lineno)
            body = node.body
            if body:
                add(start, body[0].lineno - 1)
            if body and isinstance(body[0], ast.Expr):
                value = body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    body = body[1:]
            for child in body:
                visit(child)
            return
        if isinstance(node, ast.Lambda):
            add(node.lineno, node.body.end_lineno)
        if isinstance(node, SIMPLE_STATEMENTS):
            add(node.lineno, node.end_lineno)
        elif isinstance(node, (ast.If, ast.While)):
            add(node.lineno, node.test.end_lineno)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            add(node.lineno, node.iter.end_lineno)
        elif isinstance(node, (ast.With, ast.AsyncWith)) and node.items:
            add(node.lineno, node.items[-1].context_expr.end_lineno)
        elif isinstance(node, ast.Match):
            add(node.lineno, node.subject.end_lineno)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return ranges


def build_statement_map() -> None:
    import coverage
    from impact_collector import OUTPUT, ROOT, SOURCE, expanded_contexts

    impact_path = OUTPUT / "impact-map.json"
    impact = json.loads(impact_path.read_text())
    groups: dict[str, set[str]] = defaultdict(set)
    for test in impact["tests"]:
        groups[test.rsplit("::", 1)[0]].add(test)

    data = coverage.CoverageData(basename=str(ROOT / ".coverage"))
    data.read()
    if not data.measured_files():
        raise RuntimeError("No merged coverage data for statement map")

    statements = {}
    for filename in sorted(data.measured_files()):
        path = Path(filename)
        if not path.is_relative_to(SOURCE):
            continue
        contexts_by_line = data.contexts_by_lineno(filename)
        entries = []
        for start, end in sorted(statement_ranges(ast.parse(path.read_bytes(), filename=filename))):
            contexts = set()
            for line in range(start, end + 1):
                contexts.update(contexts_by_line.get(line, ()))
            entries.append([start, end, sorted(expanded_contexts(contexts, groups))])
        statements[str(path.relative_to(ROOT))] = entries

    impact["statements"] = statements
    compact_test_ids(impact)
    impact_path.write_text(json.dumps(impact))
    print(f"Mapped statements in {len(statements)} source files")


def compact_test_ids(impact: dict) -> None:
    numbers = {test: number for number, test in enumerate(impact["tests"])}

    def compact(tests: list[str]) -> list[int]:
        return [numbers[test] for test in tests]

    for key in ("process_tests", "metadata_tests"):
        impact[key] = compact(impact[key])
    for key in ("fixtures", "source_reads"):
        for path, tests in impact[key].items():
            impact[key][path] = compact(tests)
    for entries in impact.get("member_reads", {}).values():
        for entry in entries:
            entry[3] = compact(entry[3])
    for key in ("functions", "statements", "classes", "modules"):
        for entries in impact.get(key, {}).values():
            for entry in entries:
                entry[2] = compact(entry[2])


if __name__ == "__main__":
    build_statement_map()
