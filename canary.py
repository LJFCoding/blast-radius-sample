"""Build and evaluate patch-aware candidate canary links from baseline tests.

Coverage and test-source similarities produce candidates, not proven failure
implications. The evaluator accepts observed patch or mutation outcomes.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from selector import inspect_patch, patch_changes, select_tests

DEFAULT_TESTS = Path("/data/tests.json")
DEFAULT_IMPACT_MAP = Path("/app/impact-map.json")
DEFAULT_SOURCE_ROOT = Path("/source")
MAX_CANARIES_PER_TEST = 10
MAX_COMMON_TESTS = 32
COMMON_TESTS_DIVISOR = 20
RANGE_KEYS = ("statements", "functions")
PATH_KEYS = ("fixtures", "source_reads")
SEMANTIC_KEYS = ("assertions", "parameters", "helpers", "bodies")
PROFILE_KEYS = RANGE_KEYS + PATH_KEYS + SEMANTIC_KEYS
DOMINANCE_SCHEMA = "structural-test-dominance-v1"
DOMINANCE_PROOF = "fixture_free_structural_test_equivalence"
DOMINANCE_IMPLICATION = "heavy_fails_implies_canary_fails"
DOMINANCE_SCOPE = "isolated-fixture-free-test-methods"
PROOF_SAFE_KEYS = (
    "statements", "functions", "classes", "modules", "fixtures", "source_reads",
)
DYNAMIC_TEST_ATTRIBUTES = {
    "__class__", "__name__", "__qualname__", "co_name", "f_code",
    "getframe", "getsource", "request", "_testMethodName",
}


def _read_tests(path: Path) -> tuple[dict[str, float], dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    seconds_by_id: dict[str, float] = {}
    number_to_id: dict[int, str] = {}
    for test in data["tests"]:
        test_id = test["id"]
        seconds = float(test["seconds"])
        number = test["number"]
        if (
            test_id in seconds_by_id
            or isinstance(number, bool)
            or not isinstance(number, int)
            or number < 0
            or number in number_to_id
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError(f"Invalid or duplicate test entry: {test_id}")
        seconds_by_id[test_id] = seconds
        number_to_id[number] = test_id
    return seconds_by_id, number_to_id


def _index_to_test(
    impact: dict[str, Any], test_ids: set[str]
) -> tuple[list[str | None], int]:
    mapped_tests = impact["tests"]
    if len(mapped_tests) != len(set(mapped_tests)):
        raise ValueError("Impact map contains duplicate test IDs")
    known_tests = [test if test in test_ids else None for test in mapped_tests]
    return known_tests, sum(test is None for test in known_tests)


def _test_for_index(index: Any, index_to_test: list[str | None]) -> str | None:
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"Invalid compact test index: {index!r}")
    if index < 0 or index >= len(index_to_test):
        raise ValueError(f"Compact test index is out of range: {index}")
    return index_to_test[index]


def _profiles_for_tests(test_ids: set[str]) -> dict[str, dict[str, set[Any]]]:
    return {
        test_id: {key: set() for key in PROFILE_KEYS}
        for test_id in test_ids
    }


def _add_range_profiles(
    profiles: dict[str, dict[str, set[Any]]],
    impact: dict[str, Any],
    key: str,
    index_to_test: list[str | None],
) -> None:
    for path, entries in impact.get(key, {}).items():
        for entry in entries:
            if len(entry) != 3:
                raise ValueError(f"Invalid {key} entry for {path}")
            start, end, indexes = entry
            unit = (path, int(start), int(end))
            for index in indexes:
                test_id = _test_for_index(index, index_to_test)
                if test_id is not None:
                    profiles[test_id][key].add(unit)


def _add_path_profiles(
    profiles: dict[str, dict[str, set[Any]]],
    impact: dict[str, Any],
    key: str,
    index_to_test: list[str | None],
) -> None:
    for path, indexes in impact.get(key, {}).items():
        for index in indexes:
            test_id = _test_for_index(index, index_to_test)
            if test_id is not None:
                profiles[test_id][key].add(path)


def _function_nodes(
    tree: ast.AST,
) -> dict[tuple[tuple[str, ...], str], list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    functions: dict[
        tuple[tuple[str, ...], str], list[ast.FunctionDef | ast.AsyncFunctionDef]
    ] = defaultdict(list)

    def visit(node: ast.AST, classes: tuple[str, ...] = ()) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, classes + (child.name,))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name.startswith("test_"):
                    functions[(classes, child.name)].append(child)
                visit(child, classes)
            else:
                visit(child, classes)

    visit(tree)
    return functions


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _test_semantics(
    source_root: Path, test_ids: set[str]
) -> tuple[dict[str, dict[str, set[str]]], int, list[str]]:
    semantics: dict[str, dict[str, set[str]]] = {
        test_id: {key: set() for key in SEMANTIC_KEYS}
        for test_id in test_ids
    }
    tests_by_file: dict[str, list[str]] = defaultdict(list)
    for test_id in test_ids:
        tests_by_file[test_id.split("::", 1)[0]].append(test_id)

    unavailable = 0
    errors = []
    for filename, file_tests in tests_by_file.items():
        path = source_root / filename
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except FileNotFoundError:
            unavailable += len(file_tests)
            continue
        except (OSError, SyntaxError) as error:
            errors.append(f"{filename}: {error}")
            unavailable += len(file_tests)
            continue

        functions = _function_nodes(tree)
        for test_id in file_tests:
            parts = test_id.split("::")
            method = parts[-1].split("[", 1)[0]
            class_path = tuple(parts[1:-1])
            matches = functions.get((class_path, method), [])
            if not matches:
                matches = [
                    function
                    for (classes, name), nodes in functions.items()
                    if name == method
                    for function in nodes
                ]
            if len(matches) != 1:
                unavailable += 1
                continue

            function = matches[0]
            body = ast.Module(body=function.body, type_ignores=[])
            semantics[test_id]["bodies"].add(
                ast.dump(body, include_attributes=False)
            )
            for node in ast.walk(function):
                if isinstance(node, ast.Assert):
                    semantics[test_id]["assertions"].add(
                        ast.dump(node, include_attributes=False)
                    )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call) and (
                            getattr(decorator.func, "attr", "") == "parametrize"
                        ):
                            semantics[test_id]["parameters"].add(
                                ast.dump(decorator, include_attributes=False)
                            )
                elif isinstance(node, ast.Call):
                    name = _call_name(node.func).lower()
                    if name.startswith(("assert", "check", "verify", "expect")):
                        semantics[test_id]["helpers"].add(
                            ast.dump(node, include_attributes=False)
                        )

    return semantics, unavailable, errors


def _coverage_memberships(impact: dict[str, Any]) -> list[tuple]:
    count = len(impact["tests"])
    memberships = [
        {
            "statements": set(),
            "functions": set(),
            "classes": set(),
            "modules": set(),
            "fixtures": set(),
            "source_reads": set(),
            "process_tests": False,
            "metadata_tests": False,
        }
        for _ in range(count)
    ]

    def add_indexes(indexes: object, unit: Any, key: str, label: str) -> None:
        if not isinstance(indexes, (list, tuple)):
            raise ValueError(f"{label} indexes must be a list")
        for index in indexes:
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= count
            ):
                raise ValueError(f"{label} has invalid test index: {index!r}")
            memberships[index][key].add(unit)

    for key in ("statements", "functions", "classes", "modules"):
        for path, entries in impact.get(key, {}).items():
            for entry in entries:
                if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                    raise ValueError(f"Invalid {key} entry for {path}")
                add_indexes(entry[2], (path, entry[0], entry[1]), key, path)
    for key in ("fixtures", "source_reads"):
        for path, indexes in impact.get(key, {}).items():
            add_indexes(indexes, path, key, path)
    for key in ("process_tests", "metadata_tests"):
        for index in impact.get(key, ()):
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= count
            ):
                raise ValueError(f"{key} has invalid test index: {index!r}")
            memberships[index][key] = True

    return [
        tuple(
            (
                key,
                tuple(sorted(membership[key]))
                if isinstance(membership[key], set)
                else membership[key],
            )
            for key in PROOF_SAFE_KEYS
        ) + (
            ("process_tests", membership["process_tests"]),
            ("metadata_tests", membership["metadata_tests"]),
        )
        for membership in memberships
    ]


def _fixture_free_test_signature(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, ...] | None:
    if isinstance(function, ast.AsyncFunctionDef) or function.decorator_list:
        return None
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    if (
        len(positional) != 1
        or positional[0].arg not in {"self", "cls"}
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or arguments.kwonlyargs
        or arguments.defaults
        or arguments.kw_defaults
    ):
        return None
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Name)
            and node.id in DYNAMIC_TEST_ATTRIBUTES
        ) or (
            isinstance(node, ast.Attribute)
            and node.attr in DYNAMIC_TEST_ATTRIBUTES
        ):
            return None
    body = ast.Module(body=function.body, type_ignores=[])
    return (
        ast.dump(arguments, include_attributes=False),
        ast.dump(body, include_attributes=False),
        ast.dump(function.returns, include_attributes=False)
        if function.returns is not None
        else "",
        function.type_comment or "",
    )


def structural_dominance_groups(
    impact: dict[str, Any], source_root: Path,
) -> list[dict[str, Any]]:
    tests = impact.get("tests")
    if not isinstance(tests, list) or not all(isinstance(test, str) for test in tests):
        raise ValueError("Impact map tests must be a list of strings")
    if len(tests) != len(set(tests)):
        raise ValueError("Impact map contains duplicate test IDs")

    functions_by_file: dict[str, dict] = {}
    coverage_memberships = _coverage_memberships(impact)
    groups: dict[tuple, list[str]] = defaultdict(list)
    proof_data: dict[tuple, dict[str, Any]] = {}
    for number, test_id in enumerate(tests):
        parts = test_id.split("::")
        if len(parts) < 3 or "[" in parts[-1]:
            continue
        filename = parts[0]
        classes = tuple(parts[1:-1])
        method = parts[-1]
        if filename not in functions_by_file:
            path = source_root / filename
            try:
                tree = ast.parse(path.read_bytes(), filename=str(path))
            except FileNotFoundError:
                functions_by_file[filename] = {}
                continue
            except (OSError, SyntaxError) as error:
                raise ValueError(f"Cannot parse test source {filename}: {error}") from error
            functions_by_file[filename] = _function_nodes(tree)
        matches = functions_by_file[filename].get((classes, method), ())
        if len(matches) != 1:
            continue
        test_signature = _fixture_free_test_signature(matches[0])
        if test_signature is None:
            continue
        key = (
            filename,
            classes,
            test_signature,
            coverage_memberships[number],
        )
        groups[key].append(test_id)
        proof_data[key] = {
            "file": filename,
            "classes": list(classes),
            "same_fixture_free_function_shape": True,
            "same_baseline_coverage": True,
        }

    result = []
    for key, members in sorted(groups.items(), key=lambda item: item[1]):
        if len(members) < 2:
            continue
        result.append({
            "members": sorted(members),
            "proof_kind": DOMINANCE_PROOF,
            "scope": DOMINANCE_SCOPE,
            "safe_for_selection": True,
            "implication": DOMINANCE_IMPLICATION,
            "proof": proof_data[key],
        })
    return result


def build_dominance_map(
    impact: dict[str, Any], source_root: Path,
) -> dict[str, Any]:
    groups = structural_dominance_groups(impact, source_root)
    return {
        "schema": DOMINANCE_SCHEMA,
        "proof_version": 1,
        "groups": groups,
        "group_count": len(groups),
        "dominated_test_count": sum(len(group["members"]) - 1 for group in groups),
    }


def validate_dominance_map(dominance: object) -> dict[str, Any]:
    if not isinstance(dominance, dict) or dominance.get("schema") != DOMINANCE_SCHEMA:
        raise ValueError("Invalid structural dominance map schema")
    if dominance.get("proof_version") != 1:
        raise ValueError("Unsupported structural dominance proof version")
    groups = dominance.get("groups")
    if not isinstance(groups, list):
        raise ValueError("Structural dominance groups must be a list")
    if (
        dominance.get("group_count") != len(groups)
        or dominance.get("dominated_test_count")
        != sum(len(group.get("members", ())) - 1 for group in groups if isinstance(group, dict))
    ):
        raise ValueError("Structural dominance totals do not match groups")
    seen = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Structural dominance group must be an object")
        members = group.get("members")
        proof = group.get("proof")
        if (
            not isinstance(members, list)
            or len(members) < 2
            or not all(isinstance(member, str) and member for member in members)
            or len(members) != len(set(members))
            or group.get("proof_kind") != DOMINANCE_PROOF
            or group.get("scope") != DOMINANCE_SCOPE
            or group.get("implication") != DOMINANCE_IMPLICATION
            or group.get("safe_for_selection") is not True
            or not isinstance(proof, dict)
            or proof.get("same_fixture_free_function_shape") is not True
            or proof.get("same_baseline_coverage") is not True
            or not isinstance(proof.get("file"), str)
            or not isinstance(proof.get("classes"), list)
        ):
            raise ValueError("Invalid structural dominance proof")
        if seen.intersection(members):
            raise ValueError("Structural dominance groups overlap")
        seen.update(members)
    return dominance


def apply_test_dominance(
    selected: set[str],
    seconds_by_id: dict[str, float],
    dominance: object,
) -> tuple[set[str], list[dict[str, Any]]]:
    validated = validate_dominance_map(dominance)
    retained = set(selected)
    replacements = []
    for group in validated["groups"]:
        members = [member for member in group["members"] if member in seconds_by_id]
        applicable = set(members) & selected
        if len(members) != len(group["members"]) or len(applicable) < 2:
            continue
        canary = min(applicable, key=lambda test_id: (seconds_by_id[test_id], test_id))
        dominated = sorted(
            test_id
            for test_id in applicable - {canary}
            if seconds_by_id[test_id] > seconds_by_id[canary]
        )
        if not dominated:
            continue
        retained.difference_update(dominated)
        replacements.extend({
            "canary": canary,
            "heavy": heavy,
            "proof_kind": DOMINANCE_PROOF,
            "scope": DOMINANCE_SCOPE,
            "implication": DOMINANCE_IMPLICATION,
            "failure_implication": DOMINANCE_IMPLICATION,
            "safe_for_selection": True,
        } for heavy in dominated)
    return retained, replacements


def _load_baseline(
    tests_path: Path, impact_path: Path, source_root: Path, dependency_path: Path | None = None,
) -> dict[str, Any]:
    seconds_by_id, number_to_id = _read_tests(tests_path)
    impact = json.loads(impact_path.read_text(encoding="utf-8"))
    index_to_test, unknown_map_tests = _index_to_test(impact, set(seconds_by_id))
    mapped_ids = {test_id for test_id in index_to_test if test_id is not None}
    profiles = _profiles_for_tests(mapped_ids)

    for key in RANGE_KEYS:
        _add_range_profiles(profiles, impact, key, index_to_test)
    for key in PATH_KEYS:
        _add_path_profiles(profiles, impact, key, index_to_test)

    semantics, unavailable_semantics, source_errors = _test_semantics(
        source_root, mapped_ids
    )
    for test_id, test_profile in semantics.items():
        for key in SEMANTIC_KEYS:
            profiles[test_id][key] = test_profile[key]

    return {
        "seconds": seconds_by_id,
        "number_to_id": number_to_id,
        "impact": impact,
        "profiles": profiles,
        "unknown_map_tests": unknown_map_tests,
        "unavailable_semantics": unavailable_semantics,
        "source_errors": source_errors,
        "source_root_available": source_root.is_dir(),
        "source_root": source_root,
        "dependency": (
            json.loads(dependency_path.read_text(encoding="utf-8"))
            if dependency_path is not None and dependency_path.is_file()
            else None
        ),
    }


def _unit_index(
    profiles: dict[str, dict[str, set[Any]]],
    key: str,
    max_common_tests: int,
) -> tuple[dict[Any, set[str]], int]:
    index: dict[Any, set[str]] = defaultdict(set)
    for test_id, profile in profiles.items():
        for unit in profile[key]:
            index[unit].add(test_id)
    common_count = sum(len(test_ids) > max_common_tests for test_ids in index.values())
    return (
        {
            unit: test_ids
            for unit, test_ids in index.items()
            if len(test_ids) <= max_common_tests
        },
        common_count,
    )


def _unit_changed(key: str, unit: Any, changes: dict[str, set[int]]) -> bool:
    if key in RANGE_KEYS:
        path, start, end = unit
        return any(start <= line <= end for line in changes.get(path, ()))
    return unit in changes


def _rarity(unit: Any, index: dict[Any, set[str]], test_count: int) -> float:
    frequency = len(index.get(unit, ()))
    if not frequency:
        return 0.0
    return math.log((test_count + 1) / (frequency + 1))


def _edge_evidence(
    canary_profile: dict[str, set[Any]],
    heavy_profile: dict[str, set[Any]],
) -> dict[str, set[Any]]:
    return {
        key: canary_profile[key] & heavy_profile[key]
        for key in PROFILE_KEYS
    }


def _ranking_score(
    canary_seconds: float,
    heavy_seconds: float,
    evidence: dict[str, set[Any]],
    indexes: dict[str, dict[Any, set[str]]],
    test_count: int,
) -> float:
    score = 30 * bool(evidence["bodies"])
    score += 8 * len(evidence["parameters"])
    score += 5 * len(evidence["assertions"])
    score += 2 * len(evidence["helpers"])
    score += 2 * sum(
        _rarity(unit, indexes[key], test_count)
        for key in ("statements", "functions")
        for unit in evidence[key]
    )
    score += sum(
        _rarity(unit, indexes[key], test_count)
        for key in ("fixtures", "source_reads")
        for unit in evidence[key]
    )
    score += math.log1p(heavy_seconds / max(canary_seconds, 0.001))
    return score


def _evidence_level(evidence: dict[str, set[Any]]) -> str:
    if evidence["bodies"]:
        return "identical_test_body_candidate"
    if evidence["parameters"] or evidence["assertions"] or evidence["helpers"]:
        return "shared_test_structure_candidate"
    if evidence["statements"] or evidence["functions"]:
        return "shared_code_candidate"
    return "shared_input_candidate"


def _serialize_edge(
    canary_id: str,
    heavy_id: str,
    seconds_by_id: dict[str, float],
    evidence: dict[str, set[Any]],
    score: float,
    patch_aware: bool,
) -> dict[str, Any]:
    statements = sorted(evidence["statements"])
    functions = sorted(evidence["functions"])
    return {
        "canary": canary_id,
        "canary_seconds": seconds_by_id[canary_id],
        "heavy": heavy_id,
        "heavy_seconds": seconds_by_id[heavy_id],
        "ranking_score": round(score, 4),
        "evidence_level": _evidence_level(evidence),
        "evidence": {
            "changed_statement_count" if patch_aware else "shared_statement_count": len(statements),
            "statement_examples": [
                {"file": path, "start": start, "end": end}
                for path, start, end in statements[:5]
            ],
            "shared_function_count": len(functions),
            "function_examples": [
                {"file": path, "start": start, "end": end}
                for path, start, end in functions[:5]
            ],
            "shared_fixtures": sorted(evidence["fixtures"]),
            "shared_source_reads": sorted(evidence["source_reads"]),
            "shared_assertion_shapes": len(evidence["assertions"]),
            "shared_parametrize_decorators": len(evidence["parameters"]),
            "shared_assertion_helpers": len(evidence["helpers"]),
            "identical_test_body": bool(evidence["bodies"]),
        },
        "failure_implication": "unproven",
    }


def _candidate_edges(
    baseline: dict[str, Any],
    affected_tests: set[str],
    changes: dict[str, set[int]] | None,
    max_canaries_per_test: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    seconds_by_id = baseline["seconds"]
    profiles = baseline["profiles"]
    max_common_tests = max(MAX_COMMON_TESTS, len(profiles) // COMMON_TESTS_DIVISOR)
    indexed = {
        key: _unit_index(profiles, key, max_common_tests)
        for key in PROFILE_KEYS
    }
    indexes = {key: value[0] for key, value in indexed.items()}
    ignored_common_units = {key: value[1] for key, value in indexed.items()}
    edges = []

    for heavy_id in sorted(affected_tests & profiles.keys()):
        heavy_profile = profiles[heavy_id]
        candidates: dict[str, dict[str, set[Any]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for key in PROFILE_KEYS:
            for unit in heavy_profile[key]:
                if changes is not None and key not in SEMANTIC_KEYS:
                    if not _unit_changed(key, unit, changes):
                        continue
                for canary_id in indexes[key].get(unit, ()):
                    if (
                        canary_id == heavy_id
                        or seconds_by_id[canary_id] >= seconds_by_id[heavy_id]
                        or (changes is not None and canary_id not in affected_tests)
                    ):
                        continue
                    candidates[canary_id][key].add(unit)

        ranked = []
        for canary_id, evidence in candidates.items():
            score = _ranking_score(
                seconds_by_id[canary_id],
                seconds_by_id[heavy_id],
                evidence,
                indexes,
                len(profiles),
            )
            ranked.append((score, canary_id, evidence))
        ranked.sort(key=lambda item: (-item[0], seconds_by_id[item[1]], item[1]))
        for score, canary_id, evidence in ranked[:max_canaries_per_test]:
            edges.append(_serialize_edge(
                canary_id,
                heavy_id,
                seconds_by_id,
                evidence,
                score,
                changes is not None,
            ))

    return (
        sorted(edges, key=lambda edge: (edge["heavy"], edge["canary"])),
        ignored_common_units,
    )


def candidate_order_scores(
    baseline: dict[str, Any],
    affected_tests: set[str],
    changes: dict[str, set[int]],
    max_canaries_per_test: int = MAX_CANARIES_PER_TEST,
) -> dict[str, int]:
    """Rank baseline-overlap canaries without removing any affected test."""
    if max_canaries_per_test < 1:
        raise ValueError("max_canaries_per_test must be positive")
    edges, _ = _candidate_edges(
        baseline, affected_tests, changes, max_canaries_per_test
    )
    scores: dict[str, int] = defaultdict(int)
    for edge in edges:
        scores[edge["canary"]] += math.floor(edge["ranking_score"] * 100 + 0.5)
    return dict(scores)


def _patch_context(
    patch_path: Path,
    baseline: dict[str, Any],
) -> tuple[dict[str, set[int]] | None, set[str], bool]:
    all_tests = set(baseline["seconds"])
    text = patch_path.read_text(encoding="utf-8")
    try:
        unchanged = inspect_patch(
            text,
            baseline["source_root"],
            baseline["dependency"].get("equivalence")
            if baseline["dependency"] is not None
            else None,
        )
        changes = patch_changes(text)
        affected = select_tests(
            changes, baseline["impact"], all_tests, unchanged, baseline["dependency"],
            text, baseline["source_root"],
        )
    except (OSError, UnicodeError, ValueError, SyntaxError) as error:
        print(f"{patch_path.name}: cannot safely analyze patch: {error}", file=sys.stderr)
        return None, all_tests, True
    return changes, affected, affected == all_tests


def patch_dominance_edges(
    baseline: dict[str, Any],
    affected_tests: set[str],
    dominance: object,
    patch_aware: bool = True,
) -> tuple[set[str], list[dict[str, Any]]]:
    selected, edges = apply_test_dominance(
        affected_tests, baseline["seconds"], dominance
    )
    for edge in edges:
        edge["patch_aware"] = patch_aware
        edge["canary_seconds"] = baseline["seconds"][edge["canary"]]
        edge["heavy_seconds"] = baseline["seconds"][edge["heavy"]]
    return selected, edges


def build_graph(
    tests_path: Path,
    impact_path: Path,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    patch_path: Path | None = None,
    max_canaries_per_test: int = MAX_CANARIES_PER_TEST,
    dependency_path: Path | None = None,
) -> dict[str, Any]:
    if max_canaries_per_test < 1:
        raise ValueError("max_canaries_per_test must be positive")
    baseline = _load_baseline(tests_path, impact_path, source_root, dependency_path)
    changes = None
    affected_tests = set(baseline["seconds"])
    full_suite_fallback = False
    if patch_path is not None:
        changes, affected_tests, full_suite_fallback = _patch_context(
            patch_path, baseline
        )

    edges, ignored_common_units = _candidate_edges(
        baseline,
        affected_tests,
        None if full_suite_fallback else changes,
        max_canaries_per_test,
    )
    dominance = build_dominance_map(baseline["impact"], baseline["source_root"])
    selected, dominance_edges = patch_dominance_edges(
        baseline,
        affected_tests,
        dominance,
        patch_path is not None and changes is not None,
    )
    return {
        "schema": "baseline-canary-candidates-v3",
        "patch": patch_path.name if patch_path else None,
        "patch_aware": patch_path is not None and changes is not None and not full_suite_fallback,
        "full_suite_fallback": full_suite_fallback,
        "affected_tests": len(affected_tests),
        "baseline_tests": len(baseline["profiles"]),
        "inventory_tests_without_coverage": len(
            set(baseline["seconds"]) - set(baseline["profiles"])
        ),
        "impact_tests_without_runtime": baseline["unknown_map_tests"],
        "test_source_root_available": baseline["source_root_available"],
        "test_source_files_unavailable": baseline["unavailable_semantics"],
        "test_source_errors": baseline["source_errors"],
        "ignored_high_fanout_units": ignored_common_units,
        "candidate_edges": len(edges),
        "edges": edges,
        "dominance_edges": dominance_edges,
        "dominance_group_count": len(dominance["groups"]),
        "selected_tests": len(selected),
        "warning": (
            "Candidate edges rank patch-relevant baseline overlap and test-source "
            "similarity. Only structural dominance edges are safe_for_selection."
        ),
    }


def _observed_failures(
    outcome: dict[str, Any],
    number_to_id: dict[int, str],
    known_ids: set[str],
) -> set[str]:
    result = set()
    for value in outcome["failing"]:
        if isinstance(value, bool):
            raise ValueError(f"Invalid failing test reference: {value!r}")
        if isinstance(value, int):
            if value not in number_to_id:
                raise ValueError(f"Unknown failing test number: {value}")
            result.add(number_to_id[value])
        elif isinstance(value, str) and value in known_ids:
            result.add(value)
        else:
            raise ValueError(f"Unknown failing test reference: {value!r}")
    return result


def evaluate_observations(
    tests_path: Path,
    impact_path: Path,
    patch_dir: Path,
    observations_path: Path,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    max_canaries_per_test: int = MAX_CANARIES_PER_TEST,
    dependency_path: Path | None = None,
) -> dict[str, Any]:
    if max_canaries_per_test < 1:
        raise ValueError("max_canaries_per_test must be positive")
    baseline = _load_baseline(tests_path, impact_path, source_root, dependency_path)
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    cases = observations.get("commits", observations.get("mutations"))
    if not isinstance(cases, dict):
        raise ValueError("Observations must contain a 'commits' or 'mutations' object")

    seconds_by_id = baseline["seconds"]
    number_to_id = baseline["number_to_id"]
    rows = []
    totals = defaultdict(float)
    edge_failure_observations = 0
    edge_failure_counterexamples = 0
    canary_pass_heavy_failure = 0
    implication_observations: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "patches": 0,
            "canary_failures": 0,
            "heavy_failures": 0,
            "canary_failure_heavy_passes": 0,
            "heavy_failure_canary_passes": 0,
        }
    )
    dominance = build_dominance_map(baseline["impact"], baseline["source_root"])

    for case_id, outcome in sorted(cases.items()):
        patch_path = patch_dir / f"{case_id}.patch"
        if not patch_path.is_file():
            raise ValueError(f"Missing patch for observation {case_id}: {patch_path}")
        changes, affected, full_suite_fallback = _patch_context(
            patch_path, baseline
        )
        candidate_edges = []
        if not full_suite_fallback:
            candidate_edges, _ = _candidate_edges(
                baseline, affected, changes, max_canaries_per_test
            )
        failures = _observed_failures(outcome, number_to_id, set(seconds_by_id))
        selected, dominance_edges = patch_dominance_edges(
            baseline, affected, dominance
        )
        canaries = {edge["canary"] for edge in dominance_edges}
        conditional_heavy = {edge["heavy"] for edge in dominance_edges}
        missed = failures - selected
        baseline_seconds = sum(seconds_by_id[test_id] for test_id in affected)
        selected_seconds = sum(seconds_by_id[test_id] for test_id in selected)

        for edge in candidate_edges:
            canary_failed = edge["canary"] in failures
            heavy_failed = edge["heavy"] in failures
            observation = implication_observations[(edge["canary"], edge["heavy"])]
            observation["patches"] += 1
            observation["canary_failures"] += int(canary_failed)
            observation["heavy_failures"] += int(heavy_failed)
            observation["canary_failure_heavy_passes"] += int(
                canary_failed and not heavy_failed
            )
            observation["heavy_failure_canary_passes"] += int(
                heavy_failed and not canary_failed
            )
            if canary_failed:
                edge_failure_observations += 1
                if not heavy_failed:
                    edge_failure_counterexamples += 1
            elif heavy_failed:
                canary_pass_heavy_failure += 1

        totals["cases"] += 1
        totals["cases_with_failures"] += bool(failures)
        totals["dominance_cases"] += bool(dominance_edges)
        totals["missed_cases"] += bool(missed)
        totals["missed_tests"] += len(missed)
        totals["affected_test_seconds"] += baseline_seconds
        totals["gated_test_seconds"] += selected_seconds
        rows.append({
            "id": case_id,
            "affected_tests": len(affected),
            "canaries": len(canaries),
            "conditional_heavy_tests": len(conditional_heavy),
            "dominance_edges": len(dominance_edges),
            "missed_tests": sorted(missed),
            "affected_test_seconds": round(baseline_seconds, 3),
            "gated_test_seconds": round(selected_seconds, 3),
        })

    totals["test_seconds_saved"] = (
        totals["affected_test_seconds"] - totals["gated_test_seconds"]
    )
    observed_implications = []
    for (canary, heavy), counts in sorted(implication_observations.items()):
        relations = []
        if (
            counts["heavy_failures"]
            and not counts["heavy_failure_canary_passes"]
        ):
            relations.append("heavy_fails_implies_canary_fails_observed")
        if (
            counts["canary_failures"]
            and not counts["canary_failure_heavy_passes"]
        ):
            relations.append("canary_fails_implies_heavy_fails_observed")
        if relations:
            observed_implications.append({
                "canary": canary,
                "heavy": heavy,
                **counts,
                "failure_implications": relations,
                "safe_for_selection": False,
                "proof": "observations_only",
            })
    return {
        "schema": "canary-observation-evaluation-v2",
        "max_canaries_per_test": max_canaries_per_test,
        "observation_kind": (
            "mutations" if "mutations" in observations else "commits"
        ),
        "cases": int(totals["cases"]),
        "cases_with_failures": int(totals["cases_with_failures"]),
        "dominance_cases": int(totals["dominance_cases"]),
        "missed_cases": int(totals["missed_cases"]),
        "missed_tests": int(totals["missed_tests"]),
        "affected_test_seconds": round(totals["affected_test_seconds"], 3),
        "gated_test_seconds": round(totals["gated_test_seconds"], 3),
        "test_seconds_saved": round(totals["test_seconds_saved"], 3),
        "time_metric_note": (
            "Sum of recorded test runtimes; excludes orchestration, worker "
            "parallelism, and canary test execution overhead."
        ),
        "edge_canary_failure_observations": edge_failure_observations,
        "edge_failure_implication_counterexamples": edge_failure_counterexamples,
        "canary_pass_but_heavy_failed_observations": canary_pass_heavy_failure,
        "observed_implications": observed_implications,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=Path, default=DEFAULT_TESTS)
    parser.add_argument("--impact-map", type=Path, default=DEFAULT_IMPACT_MAP)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dependency-map", type=Path, default=Path("/app/dependency-map.json"))
    parser.add_argument(
        "--max-canaries-per-test",
        type=int,
        default=MAX_CANARIES_PER_TEST,
        help="Maximum candidate canaries linked to each affected heavier test.",
    )
    parser.add_argument("--patch", type=Path)
    parser.add_argument(
        "--evaluate-observations",
        type=Path,
        help="Evaluate a commits/mutations failure-label file against matching patches.",
    )
    parser.add_argument("--patch-dir", type=Path)
    parser.add_argument(
        "--build-dominance",
        action="store_true",
        help="Build structurally proven test-equivalence groups.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.evaluate_observations) != bool(args.patch_dir):
        parser.error("--evaluate-observations and --patch-dir must be used together")
    if args.patch is not None and args.evaluate_observations is not None:
        parser.error("--patch cannot be combined with --evaluate-observations")
    if args.build_dominance and (
        args.patch is not None or args.evaluate_observations is not None
    ):
        parser.error("--build-dominance cannot be combined with patch analysis")
    if args.output is not None and not args.build_dominance:
        parser.error("--output requires --build-dominance")

    try:
        if args.build_dominance:
            impact = json.loads(args.impact_map.read_text(encoding="utf-8"))
            result = build_dominance_map(impact, args.source_root)
        elif args.evaluate_observations is not None:
            result = evaluate_observations(
                args.tests,
                args.impact_map,
                args.patch_dir,
                args.evaluate_observations,
                args.source_root,
                args.max_canaries_per_test,
                args.dependency_map,
            )
        else:
            result = build_graph(
                args.tests,
                args.impact_map,
                args.source_root,
                args.patch,
                args.max_canaries_per_test,
                args.dependency_map,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    encoded = json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
