"""Select tests using the coverage map built from the pinned SQLGlot revision."""

from __future__ import annotations

import ast
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import json
import math
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

from commentry import (
    SOURCE_ROOT, FileChange, cache_statistics, file_changes, has_docstring_change,
    is_formatting_only, is_inline_comment_change, is_plain_comment_change,
    parse_python_source,
)
from dependency_map import (
    identifier_variants, input_words, matching_input_words, matching_inputs,
)

DATA = Path("/data")
TESTS_PATH = DATA / "tests.json"
IMPACT_MAP = Path("/app/impact-map.json")
DEPENDENCY_MAP = Path("/app/dependency-map.json")
DOMINANCE_MAP = Path("/app/dominance-map.json")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
MAX_WORKERS = 8
WORKER_COST_MS = 2000
MAX_TWO_WORKER_TESTS = 64
MAX_TWO_WORKER_TOTAL_MS = 200_000
MAX_SEARCH_TESTS = 12
MAX_SEARCH_NODES = 2000
LIVE_PROBE_BUDGET_SECONDS = 35.0
LIVE_PROBE_MIN_TEST_SECONDS = 0.25
LIVE_PROBE_MAX_TEST_SECONDS = 2.0
LIVE_PROBE_TEST_TIMEOUT_SECONDS = 6.0
LIVE_PROBE_MIN_COMMIT_COST_MS = 2_000
LIVE_PROBE_MIN_SAVINGS_MS = 500
LIVE_PROBE_BATCH_TOTAL_MS = 4_000
SEARCH_NODES = 0


@dataclass(frozen=True)
class _IntervalIndex:
    entries: tuple[tuple[int, int, list[int] | tuple[int, ...]], ...]
    starts: tuple[int, ...]
    prefix_max_ends: tuple[int, ...]


@dataclass(frozen=True)
class _ClassFacts:
    bases: tuple[str, ...]
    methods: frozenset[str]
    decorators: frozenset[str]


def patch_changes(text: str) -> dict[str, set[int]] | None:
    changes: dict[str, set[int]] = {}
    path = None
    old_line = None
    deleted_lines: list[int] = []
    added_count = 0
    for line in text.splitlines():
        if line.startswith("diff --git "):
            path = None
            old_line = None
            deleted_lines = []
            added_count = 0
        elif line.startswith("--- "):
            name = line[4:]
            if not name.startswith("a/"):
                return None
            path = name[2:]
            changes.setdefault(path, set())
        elif line.startswith("@@ "):
            match = HUNK.match(line)
            if path is None or match is None:
                return None
            old_line = int(match.group(1))
            deleted_lines = []
            added_count = 0
        elif line.startswith(("+", "-", " ")) and old_line is not None:
            if line[0] == "-":
                changes[path].add(old_line)
                deleted_lines.append(old_line)
                old_line += 1
            elif line[0] == "+":
                impact_line = (
                    deleted_lines[added_count]
                    if added_count < len(deleted_lines)
                    else deleted_lines[-1]
                    if deleted_lines
                    else old_line
                )
                changes[path].add(impact_line)
                added_count += 1
            else:
                old_line += 1
                deleted_lines = []
                added_count = 0
        elif line.startswith(("Binary files ", "GIT binary patch")):
            return None
    return changes if changes and all(changes.values()) else None


def patch_changes_from_files(changes: list[FileChange]) -> dict[str, set[int]] | None:
    if not changes or any(not change.mapping_supported or not change.impact_lines for change in changes):
        return None
    result: dict[str, set[int]] = {}
    for change in changes:
        if change.path in result:
            return None
        result[change.path] = set(change.impact_lines)
    return result or None


def build_interval_indexes(impact: dict) -> dict[tuple[str, str], _IntervalIndex]:
    indexes = {}
    for kind in ("statements", "functions", "classes", "modules"):
        for path, raw_entries in impact.get(kind, {}).items():
            entries = []
            for entry in raw_entries:
                if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                    continue
                start, end, numbers = entry
                if (
                    isinstance(start, bool)
                    or isinstance(end, bool)
                    or not isinstance(start, int)
                    or not isinstance(end, int)
                    or start > end
                    or not isinstance(numbers, (list, tuple))
                ):
                    continue
                entries.append((start, end, numbers))
            entries.sort(key=lambda entry: (entry[0], entry[1]))
            prefix_max_ends = []
            maximum = -1
            for _, end, _ in entries:
                maximum = max(maximum, end)
                prefix_max_ends.append(maximum)
            indexes[(kind, path)] = _IntervalIndex(
                tuple(entries),
                tuple(entry[0] for entry in entries),
                tuple(prefix_max_ends),
            )
    return indexes


def _matching_ranges(
    index: _IntervalIndex | None, line: int,
) -> list[tuple[int, int, list[int] | tuple[int, ...]]]:
    if index is None:
        return []
    position = bisect_right(index.starts, line) - 1
    matches = []
    while position >= 0 and index.prefix_max_ends[position] >= line:
        entry = index.entries[position]
        if entry[0] <= line <= entry[1]:
            matches.append(entry)
        position -= 1
    return sorted(matches, key=lambda entry: (entry[1] - entry[0], entry[0], entry[1]))


def _ids_for_indexes(indexes: list[int] | tuple[int, ...], mapped_tests: list[str], all_ids: set[str]) -> set[str]:
    return {
        mapped_tests[index]
        for index in indexes
        if isinstance(index, int)
        and not isinstance(index, bool)
        and 0 <= index < len(mapped_tests)
        and mapped_tests[index] in all_ids
    }


def _source_reader_indexes(impact: dict, path: str, lines: set[int]) -> set[int]:
    """Keep member-attributed readers and conservatively retain file-wide reads."""
    file_readers = set(impact.get("source_reads", {}).get(path, ()))
    member_entries = impact.get("member_reads", {}).get(path, ())
    attributed: set[int] = set()
    matching: set[int] = set()
    for entry in member_entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            continue
        _, start, end, indexes = entry
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start > end
            or not isinstance(indexes, (list, tuple))
        ):
            continue
        readers = {
            index
            for index in indexes
            if isinstance(index, int) and not isinstance(index, bool)
        }
        attributed.update(readers)
        if any(start <= line <= end for line in lines):
            matching.update(readers)
    return matching | (file_readers - attributed)


def selection_evidence(
    changes: dict[str, set[int]] | None,
    impact: dict,
    unchanged_code: frozenset[str],
    interval_indexes: dict[tuple[str, str], _IntervalIndex] | None = None,
) -> tuple[str | None, list[tuple[str, list[int]]]]:
    if changes is None:
        return "unsupported patch: full-suite fallback", []

    evidence = []
    for path, lines in changes.items():
        if path.endswith(".py"):
            if path in impact["uncertain_source_reads"]:
                return f"{path}: unattributed source read requires full suite", []
            evidence.append((
                f"{path}: source read",
                sorted(_source_reader_indexes(impact, path, lines)),
            ))
        if path in unchanged_code:
            continue
        if path.endswith(".py"):
            evidence.append((f"{path}: metadata-sensitive test", impact["metadata_tests"]))
        if path in impact["fixtures"]:
            evidence.append((f"{path}: fixture read", impact["fixtures"][path]))
            continue
        if not any(path in impact.get(kind, {}) for kind in ("functions", "classes", "modules")):
            return f"{path}: unmapped file requires full suite", []
        for line in lines:
            matches = []
            for kind in ("statements", "functions", "classes", "modules"):
                matching = _matching_ranges(
                    interval_indexes.get((kind, path)) if interval_indexes else None,
                    line,
                )
                mapped = [entry for entry in matching if entry[2]]
                if mapped:
                    matches = mapped
                    break
                if interval_indexes is None:
                    entries = impact.get(kind, {}).get(path, ())
                    matching = [
                        (start, end, tuple(numbers))
                        for start, end, numbers in entries
                        if start <= line <= end
                    ]
                    mapped = [entry for entry in matching if entry[2]]
                    if mapped:
                        matches = mapped
                        break
            if not matches:
                return f"{path}:{line}: unmapped change requires full suite", []
            evidence.extend((f"{path}:{line}: covered range", list(numbers)) for _, _, numbers in matches)
    return None, evidence


@dataclass
class _StaticPatchEvidence:
    setting_names: set[str]
    module_names: set[str]
    changed_strings: set[str]
    static_lines: dict[str, set[int]]
    owners: dict[str, set[str]]
    settings_by_path: dict[str, set[str]]
    module_names_by_path: dict[str, set[str]]
    strings_by_path: dict[str, set[str]]
    container_keys_by_path: dict[str, set[str]]
    container_lines_by_path: dict[str, set[int]]
    container_safe_lines_by_path: dict[str, set[int]]
    container_names_by_path: dict[str, set[str]]
    condition_lines_by_path: dict[str, set[int]]
    symbols_by_path: dict[str, set[str]]


def _assignment_target_names(statement: ast.stmt) -> set[str]:
    targets = []
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        targets = [statement.target]
    names = set()
    for target in targets:
        for node in ast.walk(target):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                names.add(node.attr)
        if isinstance(target, ast.Subscript):
            if isinstance(target.value, ast.Name):
                names.add(target.value.id)
            elif isinstance(target.value, ast.Attribute):
                names.add(target.value.attr)
    return names


def _static_container_value(node: ast.AST) -> bool:
    if isinstance(node, (ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom)):
        return False
    if isinstance(node, ast.Call):
        if not isinstance(node.func, (ast.Name, ast.Attribute)):
            return False
        return all(
            _static_container_value(child)
            for child in (*node.args, *(keyword.value for keyword in node.keywords))
        )
    if isinstance(node, (ast.Constant, ast.Name, ast.Attribute)):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return all(
            _static_container_value(child)
            for child in ast.iter_child_nodes(node)
        )
    if isinstance(node, ast.UnaryOp):
        return _static_container_value(node.operand)
    return not any(
        isinstance(child, (ast.Lambda, ast.Call))
        for child in ast.iter_child_nodes(node)
    )


def _container_key_values(
    tree: ast.AST,
) -> dict[tuple[str, ...], tuple[str, int, bool]]:
    result: dict[tuple[str, ...], tuple[str, int, bool]] = {}

    def collect_value(
        value: ast.AST,
        identity: tuple[str, ...],
    ) -> None:
        if not isinstance(value, ast.Dict):
            return
        for key, child in zip(value.keys, value.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            key_id = (*identity, key.value)
            static_value = _static_container_value(child)
            result[key_id] = (
                ast.dump(child, include_attributes=False),
                key.lineno,
                static_value,
            )
            if isinstance(child, ast.Dict):
                collect_value(child, key_id)

    def visit(node: ast.AST, owners: tuple[str, ...] = ()) -> None:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                visit(child, (*owners, node.name))
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = getattr(node, "value", None)
            if value is not None:
                for name in _assignment_target_names(node):
                    collect_value(value, (*owners, name))
            return
        for child in ast.iter_child_nodes(node):
            visit(child, owners)

    visit(tree)
    return result


def _changed_expression_symbols(node: ast.AST) -> set[str]:
    result = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            result.add(child.id)
        elif isinstance(child, ast.Attribute):
            result.add(child.attr)
        elif isinstance(child, ast.Constant):
            if isinstance(child.value, str):
                result.update(input_words(child.value))
            elif isinstance(child.value, bool):
                result.add(f"literal:{str(child.value).lower()}")
            elif isinstance(child.value, (int, float)):
                result.add(f"literal:{child.value!r}")
            elif child.value is None:
                result.add("literal:none")
    return {value for value in result if value}


def _condition_values(tree: ast.AST) -> dict[int, tuple[str, set[str]]]:
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            continue
        test = node.test
        result[test.lineno] = (
            ast.dump(test, include_attributes=False),
            _changed_expression_symbols(test),
        )
    return result


def _static_patch_evidence(changes: list[FileChange]) -> _StaticPatchEvidence:
    setting_names: set[str] = set()
    module_names: set[str] = set()
    changed_strings: set[str] = set()
    static_lines: dict[str, set[int]] = {}
    owners: dict[str, set[str]] = {}
    settings_by_path: dict[str, set[str]] = {}
    module_names_by_path: dict[str, set[str]] = {}
    strings_by_path: dict[str, set[str]] = {}
    container_keys_by_path: dict[str, set[str]] = defaultdict(set)
    container_lines_by_path: dict[str, set[int]] = defaultdict(set)
    container_safe_lines_by_path: dict[str, set[int]] = defaultdict(set)
    container_names_by_path: dict[str, set[str]] = defaultdict(set)
    condition_lines_by_path: dict[str, set[int]] = defaultdict(set)
    symbols_by_path: dict[str, set[str]] = defaultdict(set)

    for change in changes:
        if Path(change.path).suffix != ".py":
            continue
        original_tree = parse_python_source(change.original)
        modified_tree = parse_python_source(change.modified)
        original_keys = _container_key_values(original_tree)
        modified_keys = _container_key_values(modified_tree)
        changed_keys = {
            key
            for key in original_keys.keys() | modified_keys.keys()
            if (
                original_keys.get(key, ("", 0))[0]
                != modified_keys.get(key, ("", 0))[0]
            )
        }
        for key in changed_keys:
            container_keys_by_path[change.path].add(key[-1])
            container_names_by_path[change.path].add(key[-2])
            old_record = original_keys.get(key)
            new_record = modified_keys.get(key)
            line = (
                old_record[1]
                if old_record
                else change.added_impact_lines.get(new_record[1], new_record[1])
                if new_record
                else 0
            )
            if line:
                container_lines_by_path[change.path].add(line)
                if (
                    (old_record and new_record and old_record[2] and new_record[2])
                    or (old_record and not new_record and old_record[2])
                    or (new_record and not old_record and new_record[2])
                ):
                    container_safe_lines_by_path[change.path].add(line)
        original_conditions = _condition_values(original_tree)
        modified_conditions = _condition_values(modified_tree)
        changed_conditions = {
            line
            for line in original_conditions.keys() | modified_conditions.keys()
            if original_conditions.get(line) != modified_conditions.get(line)
        }
        condition_lines_by_path[change.path].update(changed_conditions)
        for tree, changed_lines in (
            (original_tree, set(change.removed)),
            (modified_tree, set(change.added)),
        ):
            for node in ast.walk(tree):
                if (
                    hasattr(node, "lineno")
                    and node.lineno in changed_lines
                ):
                    symbols_by_path[change.path].update(
                        _changed_expression_symbols(node)
                    )
        versions = (
            (original_tree, set(change.removed), {}),
            (
                modified_tree,
                set(change.added),
                change.added_impact_lines,
            ),
        )
        for tree, changed_lines, line_mapping in versions:
            if not changed_lines:
                continue

            def visit(node: ast.AST, classes: tuple[str, ...] = ()) -> None:
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        visit(child, (*classes, node.name))
                    return
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    affected = {
                        line for line in changed_lines
                        if node.lineno <= line <= node.end_lineno
                    }
                    if affected:
                        old_lines = {
                            line_mapping.get(line, line) for line in affected
                        }
                        static_lines.setdefault(change.path, set()).update(old_lines)
                        names = _assignment_target_names(node)
                        owner = ".".join(classes)
                        for name in names:
                            if classes:
                                setting_names.add(name)
                                owners.setdefault(name, set()).add(owner)
                                settings_by_path.setdefault(change.path, set()).add(name)
                            else:
                                module_names.add(name)
                                module_names_by_path.setdefault(change.path, set()).add(name)
                        for child in ast.walk(node):
                            if (
                                isinstance(child, ast.Constant)
                                and isinstance(child.value, str)
                                and child.lineno in affected
                            ):
                                changed_strings.add(child.value)
                                strings_by_path.setdefault(change.path, set()).add(child.value)
                for child in ast.iter_child_nodes(node):
                    visit(child, classes)

            visit(tree)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.lineno in changed_lines
                ):
                    changed_strings.add(node.value)
                    strings_by_path.setdefault(change.path, set()).add(node.value)

    return _StaticPatchEvidence(
        setting_names,
        module_names,
        changed_strings,
        static_lines,
        owners,
        settings_by_path,
        module_names_by_path,
        strings_by_path,
        dict(container_keys_by_path),
        dict(container_lines_by_path),
        dict(container_safe_lines_by_path),
        dict(container_names_by_path),
        dict(condition_lines_by_path),
        dict(symbols_by_path),
    )


def changed_static_evidence(
    text: str, root: Path = SOURCE_ROOT, changes: list[FileChange] | None = None,
) -> tuple[set[str], set[str]]:
    evidence = _static_patch_evidence(changes or file_changes(text, root))
    return evidence.setting_names | evidence.module_names, evidence.changed_strings


def _precise_input_indexes(dependency: dict, text: str) -> set[int]:
    words = input_words(text)
    if text.strip() == "0":
        words.add("0")
    result = matching_input_words(
        dependency.get("scoped_inputs", {}),
        {word for word in words if len(word) >= 2},
    )
    for key in words & {"0", "x", "u"}:
        result.update(
            dependency.get("scoped_inputs", {}).get(f"escape:{key}", ())
        )
    return result


def _precise_symbol_indexes(dependency: dict, symbols: set[str]) -> set[int]:
    result = set()
    index = dependency.get("test_symbols", {})
    for symbol in symbols:
        result.update(index.get(symbol, ()))
        if not symbol.startswith("literal:"):
            for variant in identifier_variants(symbol):
                result.update(index.get(variant, ()))
    return result


def select_static_patch(
    patch_text: str,
    changes: dict[str, set[int]] | None,
    impact: dict,
    all_ids: set[str],
    dependency: dict,
    unchanged_code: frozenset[str],
    root: Path = SOURCE_ROOT,
    file_records: list[FileChange] | None = None,
    interval_indexes: dict[tuple[str, str], _IntervalIndex] | None = None,
) -> set[str] | None:
    if changes is None:
        return None

    records = file_records if file_records is not None else file_changes(patch_text, root)
    if not records or any(not change.mapping_supported for change in records):
        return None
    static = _static_patch_evidence(records)
    if not static.static_lines:
        return None

    uncertain_settings = set(dependency.get("uncertain_settings", ()))
    uncertain_dynamic_files = set(dependency.get("uncertain_dynamic_files", ()))
    mapped_tests = impact["tests"]
    selected = all_ids - (set(mapped_tests) & all_ids)

    def indexes_for_line(
        path: str,
        line: int,
        kinds: tuple[str, ...],
        narrowest: bool = False,
    ) -> set[int]:
        for kind in kinds:
            ranges = (
                _matching_ranges(interval_indexes.get((kind, path)), line)
                if interval_indexes is not None
                else [
                    (start, end, tuple(numbers))
                    for start, end, numbers in impact.get(kind, {}).get(path, ())
                    if start <= line <= end
                ]
            )
            if narrowest and ranges:
                width = min(end - start for start, end, _ in ranges)
                ranges = [
                    entry for entry in ranges if entry[1] - entry[0] == width
                ]
            candidates = {number for _, _, numbers in ranges for number in numbers}
            if candidates:
                return candidates
        return set()

    for path, lines in changes.items():
        if path in impact.get("fixtures", {}):
            selected.update(
                _ids_for_indexes(impact["fixtures"][path], mapped_tests, all_ids)
            )
            continue
        if not path.endswith(".py") or path in impact.get("uncertain_source_reads", ()):
            return None
        if path in uncertain_dynamic_files:
            return set(all_ids)

        selected.update(
            _ids_for_indexes(
                _source_reader_indexes(impact, path, lines),
                mapped_tests,
                all_ids,
            )
        )
        if path in unchanged_code:
            continue
        selected.update(
            _ids_for_indexes(
                impact.get("metadata_tests", ()),
                mapped_tests,
                all_ids,
            )
        )

        names = static.settings_by_path.get(path, set())
        module_names = static.module_names_by_path.get(path, set())
        changed_strings = static.strings_by_path.get(path, set())
        container_lines = static.container_lines_by_path.get(path, set())
        container_safe_lines = static.container_safe_lines_by_path.get(path, set())
        container_keys = static.container_keys_by_path.get(path, set())
        container_names = static.container_names_by_path.get(path, set())
        key_indexes = set()
        for key in container_keys:
            key_indexes.update(_precise_input_indexes(dependency, key))
            if len(key) >= 2:
                key_indexes.update(
                    dependency.get("scoped_inputs", {}).get(key.lower(), ())
                )
        symbol_indexes = _precise_symbol_indexes(
            dependency,
            static.symbols_by_path.get(path, set()),
        )
        precise_input_indexes = set()
        for text in changed_strings:
            precise_input_indexes.update(
                _precise_input_indexes(dependency, text)
            )
        for line in lines:
            if line not in static.static_lines.get(path, set()):
                candidates = indexes_for_line(
                    path, line, ("statements", "functions", "classes", "modules")
                )
                if not candidates:
                    return None
                selected.update(_ids_for_indexes(candidates, mapped_tests, all_ids))
                continue
            if (
                line in container_lines
                and line in container_safe_lines
                and key_indexes
                and container_names
                and all(name.isupper() for name in container_names)
            ):
                key_scope = key_indexes & symbol_indexes
                if not key_scope:
                    key_scope = key_indexes
                selected.update(
                    _ids_for_indexes(
                        key_scope,
                        mapped_tests,
                        all_ids,
                    )
                )
                continue

            candidate_indexes: set[int] = set()
            owner_indexes = indexes_for_line(
                path,
                line,
                ("statements", "functions", "classes", "modules"),
                narrowest=True,
            )
            for name in names:
                definitions = dependency.get("class_setting_definitions", {}).get(name, ())
                matching_owners = [
                    definition
                    for definition in definitions
                    if isinstance(definition, dict)
                    and definition.get("file") == path
                    and definition.get("class") in static.owners.get(name, set())
                ]
                owner_tests = {
                    number
                    for definition in matching_owners
                    for number in dependency.get("owner_tests", {}).get(
                        f"{path}::{definition.get('class')}", ()
                    )
                }
                setting_indexes = set(dependency.get("settings", {}).get(name, ()))
                dynamic_file_indexes = set(
                    dependency.get("dynamic_by_file", {}).get(path, ())
                )
                owner_setting_indexes = setting_indexes & owner_tests
                owner_scope = owner_tests or owner_indexes
                if name in uncertain_settings:
                    if not owner_scope or (
                        not owner_setting_indexes
                        and not owner_tests
                        and not dynamic_file_indexes
                    ):
                        return set(all_ids)
                if matching_owners and owner_tests and owner_setting_indexes:
                    attributed = (
                        owner_setting_indexes & symbol_indexes
                        if not container_lines
                        else set()
                    )
                    candidate_indexes.update(attributed or owner_setting_indexes)
                else:
                    candidate_indexes.update(setting_indexes)
                dynamic_indexes = set(
                    dependency.get("dynamic_settings", {}).get(name, ())
                )
                if name in uncertain_settings:
                    candidate_indexes.update(
                        dynamic_indexes & (owner_tests or precise_input_indexes)
                    )
                    if not owner_tests and not precise_input_indexes:
                        candidate_indexes.update(dynamic_file_indexes)
                else:
                    candidate_indexes.update(
                        dynamic_indexes & owner_tests
                        if owner_tests
                        else dynamic_indexes
                    )

            for name in module_names:
                if name in uncertain_settings:
                    if not owner_indexes:
                        return set(all_ids)
                    candidate_indexes.update(
                        dependency.get("dynamic_settings", {}).get(name, ())
                    )
                    candidate_indexes.update(
                        dependency.get("dynamic_by_file", {}).get(path, ())
                    )
                    candidate_indexes.update(owner_indexes)
                    continue
                candidate_indexes.update(
                    dependency.get("module_constants", {}).get(name, ())
                )
                candidate_indexes.update(dependency.get("settings", {}).get(name, ()))
                candidate_indexes.update(
                    dependency.get("dynamic_settings", {}).get(name, ())
                )

            static_scope = {
                number
                for name in names
                for definition in (
                    dependency.get("class_setting_definitions", {}).get(name, ())
                )
                if isinstance(definition, dict)
                and definition.get("file") == path
                and definition.get("class") in static.owners.get(name, set())
                for number in dependency.get("owner_tests", {}).get(
                    f"{path}::{definition.get('class')}", ()
                )
            }
            dynamic_scope = (
                precise_input_indexes
                if precise_input_indexes
                else static_scope
            )
            candidate_indexes.update(
                set(dependency.get("dynamic_by_file", {}).get(path, ()))
                & dynamic_scope
            )
            if any(name in uncertain_settings for name in names | module_names):
                candidate_indexes.update(
                    set(dependency.get("dynamic_tests", ())) & dynamic_scope
                )

            for text in changed_strings:
                words = {word for word in input_words(text) if len(word) >= 2}
                scoped_matches = _precise_input_indexes(dependency, text)
                if scoped_matches:
                    candidate_indexes.update(scoped_matches)
                    continue
                if not words:
                    continue
                try:
                    broad_matches = matching_inputs(
                        dependency.get("inputs", {}), text
                    )
                except ValueError:
                    continue
                if static_scope:
                    candidate_indexes.update(broad_matches & static_scope)
                else:
                    candidate_indexes.update(broad_matches)

            candidate_indexes = {
                number for number in candidate_indexes
                if isinstance(number, int)
                and not isinstance(number, bool)
                and 0 <= number < len(mapped_tests)
            }
            if not candidate_indexes:
                candidate_indexes.update(
                    indexes_for_line(
                        path, line, ("statements", "functions", "classes", "modules")
                    )
                )
            candidate_indexes = {
                number for number in candidate_indexes
                if isinstance(number, int)
                and not isinstance(number, bool)
                and 0 <= number < len(mapped_tests)
            }
            if not candidate_indexes:
                return None
            selected.update(_ids_for_indexes(candidate_indexes, mapped_tests, all_ids))

    return selected & all_ids


def select_condition_patch(
    patch_text: str,
    changes: dict[str, set[int]] | None,
    impact: dict,
    all_ids: set[str],
    dependency: dict,
    unchanged_code: frozenset[str],
    root: Path = SOURCE_ROOT,
    file_records: list[FileChange] | None = None,
    interval_indexes: dict[tuple[str, str], _IntervalIndex] | None = None,
) -> set[str] | None:
    if changes is None:
        return None
    records = file_records if file_records is not None else file_changes(patch_text, root)
    if not records:
        return None
    if any(not change.mapping_supported for change in records):
        return None
    static = _static_patch_evidence(records)
    if not static.condition_lines_by_path:
        return None

    mapped_tests = impact["tests"]
    selected = all_ids - (set(mapped_tests) & all_ids)
    narrowed = False

    for path, lines in changes.items():
        if path in impact.get("fixtures", {}):
            selected.update(
                _ids_for_indexes(impact["fixtures"][path], mapped_tests, all_ids)
            )
            continue
        if not path.endswith(".py") or path in impact.get("uncertain_source_reads", ()):
            return None
        selected.update(
            _ids_for_indexes(
                _source_reader_indexes(impact, path, lines),
                mapped_tests,
                all_ids,
            )
        )
        if path in unchanged_code:
            continue
        selected.update(
            _ids_for_indexes(
                impact.get("metadata_tests", ()),
                mapped_tests,
                all_ids,
            )
        )
        condition_lines = static.condition_lines_by_path.get(path, set())
        symbol_indexes = _precise_symbol_indexes(
            dependency, static.symbols_by_path.get(path, set())
        )
        for line in lines:
            direct = set()
            for kind in ("statements", "functions", "classes", "modules"):
                ranges = (
                    _matching_ranges(interval_indexes.get((kind, path)), line)
                    if interval_indexes is not None
                    else [
                        (start, end, tuple(numbers))
                        for start, end, numbers in impact.get(kind, {}).get(path, ())
                        if start <= line <= end
                    ]
                )
                direct = {
                    number for _, _, numbers in ranges for number in numbers
                }
                if direct:
                    break
            if not direct:
                return None
            precise = direct & symbol_indexes
            if line in condition_lines and precise:
                direct = precise
                narrowed = True
            selected.update(_ids_for_indexes(tuple(direct), mapped_tests, all_ids))

    return selected & all_ids if narrowed else None


def categorize_patch(
    patch_text: str,
    impact: dict,
    all_ids: set[str],
    dependency: dict | None = None,
    root: Path = SOURCE_ROOT,
) -> dict:
    mapped_tests = impact["tests"]
    unmapped = all_ids - (set(mapped_tests) & all_ids)
    process = _ids_for_indexes(impact.get("process_tests", ()), mapped_tests, all_ids)
    metadata = _ids_for_indexes(impact.get("metadata_tests", ()), mapped_tests, all_ids)
    dynamic = set()
    if dependency is not None:
        dynamic = _ids_for_indexes(
            dependency.get("dynamic_tests", ()), mapped_tests, all_ids
        )

    try:
        records = file_changes(patch_text, root)
        unchanged_code = inspect_changes(
            records,
            root,
            dependency.get("equivalence") if dependency is not None else None,
        )
        changes = patch_changes_from_files(records)
        interval_indexes = build_interval_indexes(impact)
        selected = select_tests(
            changes,
            impact,
            all_ids,
            unchanged_code,
            dependency,
            patch_text,
            root,
            interval_indexes,
            records,
        )
        fallback, evidence = selection_evidence(
            changes, impact, unchanged_code, interval_indexes
        )
        reasons = [reason for reason, _ in evidence]
        if fallback is not None:
            reasons = [fallback]

        static = set()
        if dependency is not None:
            static_selected = select_static_patch(
                patch_text,
                changes,
                impact,
                all_ids,
                dependency,
                unchanged_code,
                root,
                records,
                interval_indexes,
            )
            if static_selected is not None:
                static = static_selected & selected
    except (OSError, UnicodeError, ValueError, SyntaxError) as error:
        records = []
        changes = None
        unchanged_code = frozenset()
        interval_indexes = {}
        selected = all_ids
        reasons = [f"cannot safely analyze patch: {error}; full-suite fallback"]

    eliminated = all_ids - selected
    fixture = set()
    source_read = set()
    direct_coverage = set()
    if changes is not None:
        for path, lines in changes.items():
            fixture.update(
                _ids_for_indexes(
                    impact.get("fixtures", {}).get(path, ()), mapped_tests, all_ids
                )
            )
            source_read.update(
                _ids_for_indexes(
                    impact.get("source_reads", {}).get(path, ()), mapped_tests, all_ids
                )
            )
            if path not in unchanged_code:
                for line in lines:
                    for kind in ("statements", "functions", "classes", "modules"):
                        matches = _matching_ranges(
                            interval_indexes.get((kind, path)), line
                        )
                        indexes = {
                            number
                            for _, _, numbers in matches
                            for number in numbers
                        }
                        if indexes:
                            direct_coverage.update(
                                _ids_for_indexes(indexes, mapped_tests, all_ids)
                            )
                            break

    return {
        "schema": "blast-radius-test-categories-v1",
        "selected_tests": len(selected),
        "eliminated_tests": len(eliminated),
        "selected": sorted(selected),
        "eliminated": sorted(eliminated),
        "categories": {
            "direct_coverage": sorted(direct_coverage & selected),
            "fixture": sorted(fixture & selected),
            "source_read": sorted(source_read & selected),
            "static_setting_or_input": sorted(static & selected),
            "metadata": sorted(metadata & selected),
            "dynamic": sorted(dynamic & selected),
            "process": sorted(process & selected),
            "unmapped": sorted(unmapped & selected),
        },
        "reasons": reasons,
        "full_suite_fallback": selected == all_ids,
        "failure_implication": "unproven",
    }


def select_tests(
    changes: dict[str, set[int]] | None,
    impact: dict,
    all_ids: set[str],
    unchanged_code: frozenset[str] = frozenset(),
    dependency: dict | None = None,
    patch_text: str | None = None,
    root: Path = SOURCE_ROOT,
    interval_indexes: dict[tuple[str, str], _IntervalIndex] | None = None,
    file_records: list[FileChange] | None = None,
) -> set[str]:
    fallback, evidence = selection_evidence(
        changes, impact, unchanged_code, interval_indexes
    )
    if dependency is not None and patch_text is not None:
        static_selected = select_static_patch(
            patch_text,
            changes,
            impact,
            all_ids,
            dependency,
            unchanged_code,
            root,
            file_records,
            interval_indexes,
        )
        if static_selected is not None:
            return static_selected
    if fallback is not None:
        return all_ids
    mapped_tests = impact["tests"]
    selected = all_ids - (set(mapped_tests) & all_ids)
    for _, numbers in evidence:
        selected.update(_ids_for_indexes(numbers, mapped_tests, all_ids))
    return selected & all_ids


def canary_priority_scores(
    changes: dict[str, set[int]] | None,
    impact: dict,
    unchanged_code: frozenset[str] = frozenset(),
    interval_indexes: dict[tuple[str, str], _IntervalIndex] | None = None,
) -> dict[str, int]:
    if changes is None:
        return {}

    scores: dict[str, int] = {}
    mapped_tests = impact["tests"]

    def add(test_numbers: set[int] | list[int], weight: int) -> None:
        for index in test_numbers:
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(mapped_tests)
            ):
                continue
            test_id = mapped_tests[index]
            scores[test_id] = scores.get(test_id, 0) + weight

    for path, lines in changes.items():
        if path in impact["fixtures"]:
            add(impact["fixtures"][path], 4)
        elif path not in unchanged_code:
            for line in lines:
                for kind, weight in (
                    ("statements", 4),
                    ("functions", 2),
                    ("classes", 1),
                    ("modules", 1),
                ):
                    ranges = (
                        _matching_ranges(interval_indexes.get((kind, path)), line)
                        if interval_indexes is not None
                        else [
                            (start, end, tuple(numbers))
                            for start, end, numbers in impact.get(kind, {}).get(path, ())
                            if start <= line <= end
                        ]
                    )
                    matching = [entry for entry in ranges if entry[2]]
                    if matching:
                        for _, _, numbers in matching:
                            add(numbers, weight)
                        break

        if path.endswith(".py"):
            add(_source_reader_indexes(impact, path, lines), 1)
            add(impact.get("metadata_tests", ()), 1)

    return scores


def prioritize_workers(
    workers: list[list[int]],
    tests_by_number: dict[int, dict],
    scores: dict[str, int],
) -> list[list[int]]:
    def priority_key(number: int) -> tuple[bool, float, int, int]:
        test = tests_by_number[number]
        score = scores.get(test["id"], 0)
        duration_ms = _duration_ms(test)
        return (
            score > 0,
            score / max(duration_ms, 1),
            -duration_ms,
            -number,
        )

    return [
        sorted(worker, key=priority_key, reverse=True)
        for worker in workers
    ]


def _docstring_start_lines(tree: ast.AST) -> set[int]:
    result = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if not node.body or not isinstance(node.body[0], ast.Expr):
            continue
        value = node.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            result.add(value.lineno)
    return result


def _is_nonexecution_comment_change(
    change: FileChange,
    original: ast.Module,
    modified: ast.Module,
) -> bool:
    return is_plain_comment_change(change, original, modified)


def live_probe_candidates(
    selected: set[str],
    tests: list[dict],
    changes: dict[str, set[int]] | None,
    impact: dict,
    workers: list[list[int]],
    unchanged_code: frozenset[str] = frozenset(),
    preferred: set[str] | None = None,
) -> list[dict]:
    if changes is None or not workers:
        return []

    scores = canary_priority_scores(changes, impact, unchanged_code)
    test_by_number = {test["number"]: test for test in tests}
    loads = _worker_loads(workers, test_by_number)
    current_cost = max(loads, default=0) + WORKER_COST_MS * len(workers)
    if current_cost < LIVE_PROBE_MIN_COMMIT_COST_MS:
        return []

    worker_by_number = {
        number: worker_index
        for worker_index, worker in enumerate(workers)
        for number in worker
    }
    process_tests = _ids_for_indexes(
        impact.get("process_tests", ()), impact["tests"], selected
    )
    candidates = []
    for test in tests:
        if (
            test["id"] not in selected
            or test["id"] in process_tests
            or not LIVE_PROBE_MIN_TEST_SECONDS
            <= test["seconds"]
            <= LIVE_PROBE_MAX_TEST_SECONDS
            or test["number"] not in worker_by_number
        ):
            continue

        worker_index = worker_by_number[test["number"]]
        remaining_loads = loads.copy()
        remaining_loads[worker_index] -= _duration_ms(test)
        remaining_worker_count = len(workers) - (
            1 if len(workers[worker_index]) == 1 else 0
        )
        estimated_cost = (
            max(remaining_loads, default=0)
            + WORKER_COST_MS * remaining_worker_count
        )
        savings = current_cost - estimated_cost
        if savings >= LIVE_PROBE_MIN_SAVINGS_MS:
            candidates.append((savings, test))

    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda item: (
            -int(item[1]["id"] in (preferred or set())),
            -item[0],
            -scores.get(item[1]["id"], 0),
            -_duration_ms(item[1]),
            item[1]["number"],
        ),
    )
    total = 0
    result = []
    for _, test in ordered:
        duration = _duration_ms(test)
        if result and total + duration > LIVE_PROBE_BATCH_TOTAL_MS:
            continue
        if not result and duration > LIVE_PROBE_BATCH_TOTAL_MS:
            continue
        result.append(test)
        total += duration
    return result


def live_probe_candidate(
    selected: set[str],
    tests: list[dict],
    changes: dict[str, set[int]] | None,
    impact: dict,
    workers: list[list[int]],
    unchanged_code: frozenset[str] = frozenset(),
    preferred: set[str] | None = None,
) -> dict | None:
    candidates = live_probe_candidates(
        selected, tests, changes, impact, workers, unchanged_code, preferred
    )
    return candidates[0] if candidates else None


def _reference_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


@lru_cache(maxsize=8)
def _source_class_facts(root: str) -> dict[str, tuple[_ClassFacts, ...]]:
    result: dict[str, list[_ClassFacts]] = {}
    source_root = Path(root)
    for path in source_root.rglob("*.py"):
        tree = parse_python_source(path.read_text(encoding="utf-8"))
        relative = path.relative_to(source_root).with_suffix("")
        module_parts = list(relative.parts)
        if module_parts[-1] == "__init__":
            module_parts.pop()
        module = ".".join(module_parts)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            definition = _ClassFacts(
                tuple(
                    name
                    for base in node.bases
                    if (name := _reference_name(base))
                ),
                frozenset(
                    child.name
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                ),
                frozenset(
                    name
                    for decorator in node.decorator_list
                    if (name := _reference_name(decorator))
                ),
            )
            result.setdefault(node.name, []).append(definition)
            if module:
                result.setdefault(f"{module}.{node.name}", []).append(definition)
    return {name: tuple(facts) for name, facts in result.items()}


def _enum_class(
    name: str,
    facts: dict[str, tuple[_ClassFacts, ...]],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if name == "Enum" or name in seen:
        return name == "Enum"
    definitions = facts.get(name, ())
    if len(definitions) != 1:
        return False
    if "__eq__" in definitions[0].methods:
        return False
    return any(
        _enum_class(base, facts, seen | {name})
        for base in definitions[0].bases
    )


def _safe_equality_type(
    name: str,
    facts: dict[str, tuple[_ClassFacts, ...]],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if name in {
        "bool", "bytes", "float", "frozenset", "int", "object",
        "str", "tuple", "type", "AnyStr", "NoneType",
    }:
        return True
    definitions = facts.get(name, ())
    if len(definitions) != 1 or name in seen:
        return False
    definition = definitions[0]
    if "__eq__" in definition.methods:
        return False
    if "dataclass" in definition.decorators:
        return False
    return all(
        _safe_equality_type(base, facts, seen | {name})
        for base in definition.bases
    )


def _nullable_attributes(
    tree: ast.AST,
    facts: dict[str, tuple[_ClassFacts, ...]] | None,
    safe_types: frozenset[str] = frozenset(),
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    classes: list[str] = []
    imports = {
        alias.asname or alias.name: f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            classes.append(node.name)
            for child in node.body:
                self.visit(child)
            classes.pop()

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if not (
                classes
                and isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id in {"self", "cls"}
            ):
                return
            names = {
                imports.get(name, name)
                for child in ast.walk(node.annotation)
                if (
                    name := _reference_name(child)
                )
                and name not in {"None", "Optional", "Union", "list", "set", "dict"}
            }
            has_none = any(
                isinstance(child, ast.Constant) and child.value is None
                for child in ast.walk(node.annotation)
            )
            if has_none and names and all(
                name in safe_types
                or (
                    facts is not None
                    and _safe_equality_type(name, facts)
                )
                for name in names
            ):
                result.setdefault(classes[-1], set()).add(node.target.attr)

    Visitor().visit(tree)
    return result


def _has_identity_equality_swap(change: FileChange) -> bool:
    def operators(lines: dict[int, str]) -> set[str]:
        text = "\n".join(lines.values())
        result = set()
        if re.search(r"\bis\s+not\b", text):
            result.add("not_identity")
        elif re.search(r"\bis\b", text):
            result.add("identity")
        if "!=" in text:
            result.add("not_equal")
        elif "==" in text:
            result.add("equal")
        return result

    before = operators(change.removed)
    after = operators(change.added)
    return (
        "identity" in before
        and "equal" in after
        or "identity" in after
        and "equal" in before
        or "not_identity" in before
        and "not_equal" in after
        or "not_identity" in after
        and "not_equal" in before
    )


def is_identity_equality_equivalent(
    original: ast.Module,
    modified: ast.Module,
    root: Path,
    equivalence: dict | None = None,
) -> bool:
    facts = (
        None
        if equivalence is not None
        else _source_class_facts(str(root))
    )
    enum_types = frozenset(
        equivalence.get("enum_types", ()) if equivalence is not None else ()
    )
    safe_types = frozenset(
        equivalence.get("safe_equality_types", ())
        if equivalence is not None
        else ()
    )
    nullable = _nullable_attributes(original, facts, safe_types)

    class Normalize(ast.NodeVisitor):
        def __init__(self) -> None:
            self.classes: list[str] = []
            self.restorations: list[tuple[ast.Compare, list[ast.cmpop]]] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.classes.append(node.name)
            self.generic_visit(node)
            self.classes.pop()

        def visit_Compare(self, node: ast.Compare) -> None:
            self.generic_visit(node)
            left = node.left
            operators = []
            for operator, right in zip(node.ops, node.comparators):
                safe = (
                    isinstance(right, ast.Attribute)
                    and isinstance(right.value, ast.Name)
                    and (
                        right.value.id in enum_types
                        or (
                            facts is not None
                            and _enum_class(right.value.id, facts)
                        )
                    )
                ) or (
                    isinstance(right, ast.Constant)
                    and right.value is None
                    and self.classes
                    and isinstance(left, ast.Attribute)
                    and isinstance(left.value, ast.Name)
                    and left.value.id in {"self", "cls"}
                    and left.attr in nullable.get(self.classes[-1], set())
                )
                if safe and isinstance(operator, ast.Is):
                    operators.append(ast.Eq())
                elif safe and isinstance(operator, ast.IsNot):
                    operators.append(ast.NotEq())
                else:
                    operators.append(operator)
                left = right
            if operators != node.ops:
                self.restorations.append((node, node.ops))
                node.ops = operators

    before = Normalize()
    after = Normalize()
    try:
        before.visit(original)
        after.visit(modified)
        return ast.dump(original) == ast.dump(modified)
    finally:
        for node, operators in (*before.restorations, *after.restorations):
            node.ops = operators


def inspect_changes(
    changes: list[FileChange],
    root: Path = SOURCE_ROOT,
    equivalence: dict | None = None,
) -> frozenset[str]:
    unchanged_code = set()
    for change in changes:
        if Path(change.path).suffix != ".py":
            continue
        original = parse_python_source(change.original)
        modified = parse_python_source(change.modified)
        if has_docstring_change(change, original, modified):
            raise ValueError(f"{change.path}: docstring change requires the full suite")
        changed_lines = list(change.removed.values()) + list(change.added.values())
        if (
            (root / change.path).is_file()
            and changed_lines
            and (
                is_formatting_only(change, original, modified)
                or is_inline_comment_change(change, original, modified)
                or (
                    _has_identity_equality_swap(change)
                    and is_identity_equality_equivalent(
                        original, modified, root, equivalence
                    )
                )
                or (
                    all(
                        line.lstrip().startswith("#")
                        for line in changed_lines
                    )
                    and _is_nonexecution_comment_change(
                        change, original, modified
                    )
                )
            )
        ):
            unchanged_code.add(change.path)
    return frozenset(unchanged_code)


def inspect_patch(
    text: str,
    root: Path = SOURCE_ROOT,
    equivalence: dict | None = None,
) -> frozenset[str]:
    return inspect_changes(file_changes(text, root), root, equivalence)


def split(tests: list[dict], worker_count: int) -> list[list[int]]:
    ordered = sorted(tests, key=lambda test: (-_duration_ms(test), test["number"]))
    return _split_ordered(ordered, worker_count)[0]


def _duration_ms(test: dict) -> int:
    seconds = test.get("seconds")
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or seconds < 0
    ):
        raise ValueError(f"Invalid test duration: {seconds!r}")
    try:
        milliseconds = float(seconds) * 1000
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid test duration: {seconds!r}") from error
    if not math.isfinite(milliseconds):
        raise ValueError(f"Invalid test duration: {seconds!r}")
    return math.floor(milliseconds + 0.5)


def _split_ordered(tests: list[dict], worker_count: int) -> tuple[list[list[int]], int]:
    workers: list[list[int]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for test in tests:
        lightest = loads.index(min(loads))
        workers[lightest].append(test["number"])
        loads[lightest] += _duration_ms(test)
    return [worker for worker in workers if worker], max(loads, default=0)


def _split_two_optimally(tests: list[dict], durations: list[int]) -> tuple[list[list[int]], int] | None:
    target = sum(durations) // 2
    mask = (1 << (target + 1)) - 1
    history = [1]
    for duration in durations:
        history.append((history[-1] | (history[-1] << duration)) & mask)
    smaller_load = history[-1].bit_length() - 1
    remaining = smaller_load
    workers: list[list[int]] = [[], []]
    for index in range(len(tests) - 1, -1, -1):
        if (history[index] >> remaining) & 1:
            workers[1].append(tests[index]["number"])
        else:
            workers[0].append(tests[index]["number"])
            remaining -= durations[index]
    if not all(workers):
        return None
    return workers, sum(durations) - smaller_load


def _search_split(
    tests: list[dict], durations: list[int], count: int, limit: int, budget: list[int],
) -> tuple[list[list[int]], int] | None:
    workers: list[list[int]] = [[] for _ in range(count)]
    loads = [0] * count
    lower_bound = max(max(durations), (sum(durations) + count - 1) // count)
    suffix_totals = [0] * (len(durations) + 1)
    for index in range(len(durations) - 1, -1, -1):
        suffix_totals[index] = suffix_totals[index + 1] + durations[index]
    best_load = limit
    best_workers = None

    def search(index: int) -> None:
        global SEARCH_NODES
        nonlocal best_load, best_workers
        if budget[0] == 0 or best_load == lower_bound:
            return
        budget[0] -= 1
        SEARCH_NODES += 1
        if len(tests) - index < sum(not worker for worker in workers):
            return
        remaining_bound = max(
            max(loads),
            durations[index] if index < len(durations) else 0,
            (sum(loads) + suffix_totals[index] + count - 1) // count,
        )
        if remaining_bound >= best_load:
            return
        if index == len(tests):
            largest = max(loads)
            if largest < best_load:
                best_load = largest
                best_workers = [worker.copy() for worker in workers]
            return
        seen = set()
        for worker in sorted(range(count), key=lambda item: loads[item]):
            state = (loads[worker], bool(workers[worker]))
            if state in seen or loads[worker] + durations[index] >= best_load:
                continue
            seen.add(state)
            loads[worker] += durations[index]
            workers[worker].append(tests[index]["number"])
            search(index + 1)
            workers[worker].pop()
            loads[worker] -= durations[index]
            if budget[0] == 0 or best_load == lower_bound:
                return

    search(0)
    return (best_workers, best_load) if best_workers is not None else None


def choose_workers(tests: list[dict]) -> list[list[int]]:
    if not tests:
        return []
    numbers = [test.get("number") for test in tests]
    if any(
        isinstance(number, bool) or not isinstance(number, int) or number < 0
        for number in numbers
    ) or len(numbers) != len(set(numbers)):
        raise ValueError("Test numbers must be unique non-negative integers")
    durations = [_duration_ms(test) for test in tests]
    duration_by_number = {test["number"]: duration for test, duration in zip(tests, durations)}
    descending = sorted(tests, key=lambda test: (-duration_by_number[test["number"]], test["number"]))
    ascending = list(reversed(descending))
    interleaved = []
    low, high = 0, len(descending) - 1
    while low <= high:
        interleaved.append(descending[low])
        low += 1
        if low <= high:
            interleaved.append(descending[high])
            high -= 1
    orders = []
    seen_orders = set()
    for order in (descending, ascending, interleaved):
        key = tuple(test["number"] for test in order)
        if key not in seen_orders:
            orders.append(order)
            seen_orders.add(key)

    best: list[list[int]] = []
    best_cost = math.inf
    for count in range(1, min(MAX_WORKERS, len(tests)) + 1):
        for order in orders:
            workers, largest_load = _split_ordered(order, count)
            cost = largest_load + WORKER_COST_MS * len(workers)
            if cost < best_cost:
                best, best_cost = workers, cost
    ordered = descending
    durations = [duration_by_number[test["number"]] for test in ordered]
    total = sum(durations)
    if (
        1 < len(tests) <= MAX_TWO_WORKER_TESTS
        and total <= MAX_TWO_WORKER_TOTAL_MS
        and max(durations[0], (total + 1) // 2) + 2 * WORKER_COST_MS < best_cost
    ):
        result = _split_two_optimally(ordered, durations)
        if result is not None:
            workers, largest_load = result
            cost = largest_load + 2 * WORKER_COST_MS
            if cost < best_cost:
                best, best_cost = workers, cost
    if len(tests) <= MAX_SEARCH_TESTS:
        budget = [MAX_SEARCH_NODES]
        for count in range(3, min(MAX_WORKERS, len(tests)) + 1):
            lower_bound = max(durations[0], (total + count - 1) // count)
            if lower_bound + count * WORKER_COST_MS >= best_cost:
                continue
            result = _search_split(
                ordered, durations, count, best_cost - count * WORKER_COST_MS, budget
            )
            if result is not None:
                workers, largest_load = result
                best, best_cost = workers, largest_load + count * WORKER_COST_MS
            if budget[0] == 0:
                break
    return best


def validate_test_inventory(tests: object) -> list[dict]:
    if not isinstance(tests, list):
        raise ValueError("Test inventory must contain a list")
    seen_ids = set()
    seen_numbers = set()
    for test in tests:
        if not isinstance(test, dict):
            raise ValueError("Test inventory entries must be objects")
        test_id = test.get("id")
        number = test.get("number")
        if not isinstance(test_id, str) or not test_id or test_id in seen_ids:
            raise ValueError(f"Invalid or duplicate test ID: {test_id!r}")
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number < 0
            or number in seen_numbers
        ):
            raise ValueError(f"Invalid or duplicate test number: {number!r}")
        _duration_ms(test)
        seen_ids.add(test_id)
        seen_numbers.add(number)
    return tests


def _worker_loads(workers: list[list[int]], tests_by_number: dict[int, dict]) -> list[int]:
    return [
        sum(_duration_ms(tests_by_number[number]) for number in worker)
        for worker in workers
    ]


def main() -> None:
    global SEARCH_NODES
    try:
        tests_data = json.loads(TESTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(tests_data, dict):
            raise ValueError("Test inventory must be an object")
        tests = validate_test_inventory(tests_data["tests"])
        impact = json.loads(IMPACT_MAP.read_text(encoding="utf-8"))
        if not isinstance(impact, dict):
            raise ValueError("Impact map must be an object")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as error:
        print(f"cannot load selector inputs: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    dependency = None
    all_ids = {test["id"] for test in tests}
    seconds_by_id = {test["id"]: float(test["seconds"]) for test in tests}
    try:
        dependency = json.loads(DEPENDENCY_MAP.read_text(encoding="utf-8"))
        if not isinstance(dependency, dict):
            raise ValueError("Dependency map must be an object")
    except FileNotFoundError:
        print("dependency map unavailable; using coverage-only selection", file=sys.stderr)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        print(f"dependency map unavailable: {error}; using coverage-only selection", file=sys.stderr)

    interval_indexes = build_interval_indexes(impact)
    dominance = None
    apply_dominance = None
    try:
        from canary import apply_test_dominance, validate_dominance_map

        dominance = validate_dominance_map(
            json.loads(DOMINANCE_MAP.read_text(encoding="utf-8"))
        )
        apply_dominance = apply_test_dominance
    except FileNotFoundError:
        print("dominance map unavailable; using candidate-only selection", file=sys.stderr)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        print(
            f"dominance map unavailable: {error}; using candidate-only selection",
            file=sys.stderr,
        )

    canary_order = os.environ.get("BLAST_RADIUS_CANARY_ORDER") == "1"
    live_probe = os.environ.get("BLAST_RADIUS_LIVE_PROBE") == "1"
    live_probe_deadline = (
        time.monotonic() + LIVE_PROBE_BUDGET_SECONDS if live_probe else 0.0
    )
    diagnostics = os.environ.get("BLAST_RADIUS_DIAGNOSTICS") == "1"
    tests_by_number = {test["number"]: test for test in tests} if canary_order else {}
    all_tests_by_number = {test["number"]: test for test in tests}
    canary_baseline = None
    canary_order_scores = None
    if canary_order:
        try:
            from canary import _load_baseline, candidate_order_scores

            canary_baseline = _load_baseline(TESTS_PATH, IMPACT_MAP, SOURCE_ROOT)
            canary_order_scores = candidate_order_scores
        except (OSError, UnicodeError, ValueError, SyntaxError) as error:
            print(f"canary ranking unavailable: {error}", file=sys.stderr)
    total_cpu_start = time.process_time()
    for patch in sorted((DATA / "commits").glob("*.patch")):
        changes = None
        unchanged_code = frozenset()
        records: list[FileChange] = []
        SEARCH_NODES = 0
        patch_cpu_start = time.process_time()
        cache_before = cache_statistics()
        fallback_reason = "normal dependency match"
        dominance_edges = []
        try:
            text = patch.read_text(encoding="utf-8")
            records = file_changes(text)
            unchanged_code = inspect_changes(
                records,
                equivalence=(
                    dependency.get("equivalence")
                    if dependency is not None
                    else None
                ),
            )
            changes = patch_changes_from_files(records)
            fallback, _ = selection_evidence(
                changes, impact, unchanged_code, interval_indexes
            )
            selected = select_tests(
                changes,
                impact,
                all_ids,
                unchanged_code,
                dependency,
                text,
                interval_indexes=interval_indexes,
                file_records=records,
            )
            if fallback is not None and selected != all_ids:
                fallback_reason = "static setting/input dependency"
            elif fallback is not None:
                fallback_reason = fallback
            if dominance is not None and apply_dominance is not None:
                selected, dominance_edges = apply_dominance(
                    selected,
                    seconds_by_id,
                    dominance,
                )
        except (OSError, UnicodeError, ValueError, SyntaxError) as error:
            print(f"{patch.name}: cannot safely analyze patch: {error}", file=sys.stderr)
            selected = all_ids
            fallback_reason = "patch cannot be reconstructed"
        workers = choose_workers([test for test in tests if test["id"] in selected])
        probe_test_id = None
        probe_outcome = None
        probe_results = {}
        if live_probe and time.monotonic() < live_probe_deadline:
            preferred_probe_tests = set()
            if dependency is not None and changes is not None:
                condition_probe_tests = select_condition_patch(
                    text,
                    changes,
                    impact,
                    all_ids,
                    dependency,
                    unchanged_code,
                    SOURCE_ROOT,
                    records,
                    interval_indexes,
                )
                preferred_probe_tests = set(condition_probe_tests or set()) & selected
            candidates = live_probe_candidates(
                selected,
                tests,
                changes,
                impact,
                workers,
                unchanged_code,
                preferred_probe_tests,
            )
            if candidates:
                try:
                    from live_probe import probe_test_failures

                    probe_ids = [candidate["id"] for candidate in candidates]
                    probe_results = probe_test_failures(
                        SOURCE_ROOT,
                        records,
                        text,
                        probe_ids,
                        live_probe_deadline,
                        LIVE_PROBE_TEST_TIMEOUT_SECONDS,
                    )
                    passed = {
                        test_id
                        for test_id, outcome in probe_results.items()
                        if outcome == "passed"
                    }
                    if probe_ids:
                        probe_test_id = probe_ids[0]
                        probe_outcome = probe_results.get(probe_test_id)
                    if passed:
                        selected.difference_update(passed)
                        workers = choose_workers(
                            [test for test in tests if test["id"] in selected]
                        )
                    unknown = [
                        test_id
                        for test_id, outcome in probe_results.items()
                        if outcome == "unknown"
                    ]
                    if unknown:
                        print(
                            f"{patch.name}: probe inconclusive for "
                            f"{', '.join(unknown)}; keeping selection",
                            file=sys.stderr,
                        )
                except (OSError, UnicodeError, ValueError) as error:
                    print(
                        f"{patch.name}: live test probe unavailable: {error}",
                        file=sys.stderr,
                    )
        if canary_order:
            scores = canary_priority_scores(changes, impact, unchanged_code)
            if (
                canary_baseline is not None
                and canary_order_scores is not None
                and changes is not None
            ):
                graph_scores = canary_order_scores(
                    canary_baseline, selected, changes
                )
                for test_id, score in graph_scores.items():
                    scores[test_id] = scores.get(test_id, 0) + score
            workers = prioritize_workers(workers, tests_by_number, scores)
        if diagnostics:
            loads = _worker_loads(workers, all_tests_by_number)
            cache_after = cache_statistics()
            diagnostic = {
                "commit": patch.stem,
                "fallback_reason": fallback_reason,
                "selected_tests": len(selected),
                "selected_duration_ms": sum(
                    _duration_ms(test) for test in tests if test["id"] in selected
                ),
                "worker_count": len(workers),
                "worker_makespan_ms": max(loads, default=0),
                "simulated_commit_cost_ms": (
                    max(loads, default=0) + WORKER_COST_MS * len(workers)
                ),
                "misses": None,
                "dominance_edges": dominance_edges,
                "live_probe_test": probe_test_id,
                "live_probe_outcome": probe_outcome,
                "live_probe_results": probe_results,
                "cpu_ms": round((time.process_time() - patch_cpu_start) * 1000, 3),
                "ast_parse_ms": round(
                    (cache_after["ast_parse_seconds"] - cache_before["ast_parse_seconds"]) * 1000,
                    3,
                ),
                "search_nodes": SEARCH_NODES,
                "cache_hits": {
                    key: cache_after[key] - cache_before[key]
                    for key in ("ast_hits", "source_hits", "comment_hits")
                },
                "cache_misses": {
                    key: cache_after[key] - cache_before[key]
                    for key in ("ast_misses", "source_misses", "comment_misses")
                },
            }
            print(json.dumps(diagnostic, separators=(",", ":")), file=sys.stderr)
        print(json.dumps({"commit": patch.stem, "workers": workers}))
    if diagnostics:
        print(
            f"selector CPU time: {time.process_time() - total_cpu_start:.3f}s",
            file=sys.stderr,
        )
    print(f"processed {len(all_ids)} available tests", file=sys.stderr)


if __name__ == "__main__":
    main()
