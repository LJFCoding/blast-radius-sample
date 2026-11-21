"""Build and inspect baseline setting references and textual input evidence.

These observations are incomplete and must not be used to exclude tests.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

INPUT_WORD = re.compile(r"\\[A-Za-z][A-Za-z0-9]*|[A-Za-z_][A-Za-z_0-9]*")
DYNAMIC_CALLS = {"getattr", "hasattr", "setattr", "delattr", "vars"}
DYNAMIC_ATTRIBUTES = {"__dict__", "__getattribute__", "getmembers", "getattr_static", "attrgetter"}
IDENTIFIER_PART = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+"
)
IDENTIFIER_SUFFIXES = (
    "properties",
    "generator",
    "tokenizer",
    "dialect",
    "parser",
)
NUMERIC_ESCAPE = re.compile(
    r"\\(?:x[0-9A-Fa-f]{0,2}|u[0-9A-Fa-f]{0,8}|[0-7])"
)


def input_words(text: str) -> set[str]:
    return {match.group().lower() for match in INPUT_WORD.finditer(text) if len(match.group()) <= 64}


def matching_input_words(index: dict[str, list[int]], words: set[str]) -> set[int]:
    if not words:
        return set()
    return set.intersection(*(set(index.get(word, ())) for word in words))


def identifier_variants(identifier: str) -> set[str]:
    parts = IDENTIFIER_PART.findall(identifier.replace("_", " "))
    if not parts:
        return {identifier.lower()}
    lowered = [part.lower() for part in parts]
    variants = {identifier.lower(), "".join(lowered), *lowered}
    joined = "".join(lowered)
    for suffix in IDENTIFIER_SUFFIXES:
        if joined:
            if joined.endswith(suffix) and len(joined) > len(suffix):
                variants.add(joined[: -len(suffix)])
    return {variant for variant in variants if variant}


def numeric_escape_keys(source: str) -> set[str]:
    keys = set()
    for match in NUMERIC_ESCAPE.finditer(source):
        marker = match.group()[1].lower()
        keys.add(f"escape:{'u' if marker == 'u' else marker}")
    return keys


def owner_variants(owner: str) -> set[str]:
    first = owner.split(".", 1)[0]
    parts = IDENTIFIER_PART.findall(first)
    if not parts:
        return {first.lower()}
    lowered = [part.lower() for part in parts]
    joined = "".join(lowered)
    variants = {joined}
    for suffix in IDENTIFIER_SUFFIXES:
        if joined.endswith(suffix) and len(joined) > len(suffix):
            variants.add(joined[: -len(suffix)])
    return variants


def assignment_names(statement: ast.stmt) -> set[str]:
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


def class_setting_definitions(tree: ast.AST, path: str = "") -> dict[str, set[tuple[str, str]]]:
    definitions: dict[str, set[tuple[str, str]]] = defaultdict(set)

    def visit_class(node: ast.ClassDef, parents: tuple[str, ...] = ()) -> None:
        owner = ".".join((*parents, node.name))
        for statement in node.body:
            if isinstance(statement, ast.ClassDef):
                visit_class(statement, (*parents, node.name))
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            else:
                for name in assignment_names(statement):
                    definitions[name].add((path, owner))

    for node in getattr(tree, "body", ()):
        if isinstance(node, ast.ClassDef):
            visit_class(node)
    return definitions


def module_constant_definitions(tree: ast.AST, path: str = "") -> dict[str, set[str]]:
    definitions: dict[str, set[str]] = defaultdict(set)
    for statement in getattr(tree, "body", ()):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for name in assignment_names(statement):
            definitions[name].add(path)
    return definitions


def class_inheritance(tree: ast.AST, path: str) -> dict[str, list[str]]:
    result = {}

    def visit(node: ast.ClassDef, parents: tuple[str, ...] = ()) -> None:
        owner = ".".join((*parents, node.name))
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
        result[f"{path}::{owner}"] = bases
        for child in node.body:
            if isinstance(child, ast.ClassDef):
                visit(child, (*parents, node.name))

    for node in getattr(tree, "body", ()):
        if isinstance(node, ast.ClassDef):
            visit(node)
    return result


def class_equivalence_types(root: Path, trees: dict[Path, ast.AST]) -> dict:
    definitions: dict[str, list[tuple[tuple[str, ...], frozenset[str], frozenset[str]]]] = defaultdict(list)
    for path, tree in trees.items():
        relative = path.relative_to(root).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = tuple(
                name
                for base in node.bases
                if (name := (
                    base.id
                    if isinstance(base, ast.Name)
                    else base.attr
                    if isinstance(base, ast.Attribute)
                    else None
                ))
            )
            methods = frozenset(
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            decorators = frozenset(
                name
                for decorator in node.decorator_list
                if (name := (
                    decorator.id
                    if isinstance(decorator, ast.Name)
                    else decorator.func.id
                    if isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    else decorator.attr
                    if isinstance(decorator, ast.Attribute)
                    else decorator.func.attr
                    if isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    else None
                ))
            )
            value = bases, methods, decorators
            definitions[node.name].append(value)
            if module:
                definitions[f"{module}.{node.name}"].append(value)

    def unique(name: str) -> tuple[tuple[str, ...], frozenset[str], frozenset[str]] | None:
        values = definitions.get(name, ())
        return values[0] if len(values) == 1 else None

    def is_enum(name: str, seen: frozenset[str] = frozenset()) -> bool:
        if name == "Enum":
            return True
        if name in seen or (value := unique(name)) is None:
            return False
        if "__eq__" in value[1]:
            return False
        return any(is_enum(base, seen | {name}) for base in value[0])

    def safe_equality(name: str, seen: frozenset[str] = frozenset()) -> bool:
        if name in {
            "bool", "bytes", "float", "frozenset", "int", "object",
            "str", "tuple", "type", "AnyStr", "NoneType",
        }:
            return True
        if name in seen or (value := unique(name)) is None:
            return False
        if "__eq__" in value[1] or "dataclass" in value[2]:
            return False
        return all(safe_equality(base, seen | {name}) for base in value[0])

    return {
        "enum_types": sorted(name for name in definitions if is_enum(name)),
        "safe_equality_types": sorted(
            name for name in definitions if safe_equality(name)
        ),
    }


def class_setting_names(tree: ast.AST) -> set[str]:
    return set(class_setting_definitions(tree))


def setting_reference(
    node: ast.AST,
    names: set[str],
    aliases: dict[str, str] | None = None,
) -> tuple[set[str], bool]:
    found = set()
    dynamic = False
    aliases = aliases or {}
    if isinstance(node, ast.Attribute):
        if isinstance(node.ctx, ast.Load) and node.attr in names:
            found.add(node.attr)
        dynamic = node.attr in DYNAMIC_ATTRIBUTES
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names:
        found.add(node.id)
    elif (
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in aliases
    ):
        found.add(aliases[node.id])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DYNAMIC_CALLS:
        if node.func.id == "vars" or len(node.args) < 2:
            dynamic = True
        elif isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
            if node.args[1].value in names:
                found.add(node.args[1].value)
        else:
            dynamic = True
    return found, dynamic


def collect_source_references(root: Path, impact: dict, data) -> dict:
    source = root / "sqlglot"
    trees = {path: ast.parse(path.read_bytes(), filename=str(path)) for path in source.rglob("*.py")}
    class_definitions: dict[str, set[tuple[str, str]]] = defaultdict(set)
    module_definitions: dict[str, set[str]] = defaultdict(set)
    inheritance: dict[str, list[str]] = {}
    for path, tree in trees.items():
        relative = str(path.relative_to(root))
        for name, owners in class_setting_definitions(tree, relative).items():
            class_definitions[name].update(owners)
        for name, files in module_constant_definitions(tree, relative).items():
            module_definitions[name].update(files)
        inheritance.update(class_inheritance(tree, relative))
    class_names = set(class_definitions)
    module_names = set(module_definitions)
    names = class_names | module_names
    references: dict[str, set[int]] = defaultdict(set)
    module_references: dict[str, set[int]] = defaultdict(set)
    dynamic_settings: dict[str, set[int]] = defaultdict(set)
    dynamic_by_file: dict[str, set[int]] = defaultdict(set)
    uncertain: set[str] = set()
    dynamic_tests: set[int] = set()
    uncertain_dynamic_files: set[str] = set()
    measured = data.measured_files()

    for path, tree in trees.items():
        relative = str(path.relative_to(root))
        functions = impact["functions"].get(relative, ())
        classes = impact.get("classes", {}).get(relative, ())
        modules = impact.get("modules", {}).get(relative, ())
        contexts = data.contexts_by_lineno(str(path)) if str(path) in measured else {}
        aliases = {
            alias.asname or alias.name: alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name in names
        }
        for node in ast.walk(tree):
            found, dynamic = setting_reference(node, names, aliases)
            if not found and not dynamic:
                continue
            matches = [entry for entry in functions if entry[0] <= node.lineno <= entry[1]]
            if not matches:
                matches = [entry for entry in classes if entry[0] <= node.lineno <= entry[1]]
            if not matches:
                matches = [entry for entry in modules if entry[0] <= node.lineno <= entry[1]]
            tests = {test for entry in matches for test in entry[2]}
            if not tests or not contexts.get(node.lineno) or "" in contexts[node.lineno]:
                uncertain.update(found)
                if dynamic:
                    uncertain_dynamic_files.add(relative)
            for name in found:
                if name in class_names:
                    references[name].update(tests)
                if name in module_names:
                    module_references[name].update(tests)
                if dynamic:
                    dynamic_settings[name].update(tests)
            if dynamic:
                dynamic_tests.update(tests)
                dynamic_by_file[relative].update(tests)

    return {
        "names": names,
        "references": references,
        "module_references": module_references,
        "class_definitions": class_definitions,
        "module_definitions": module_definitions,
        "inheritance": inheritance,
        "equivalence": class_equivalence_types(source.parent, trees),
        "dynamic_settings": dynamic_settings,
        "dynamic_by_file": dynamic_by_file,
        "uncertain": uncertain,
        "dynamic_tests": dynamic_tests,
        "uncertain_dynamic_files": uncertain_dynamic_files,
        "uncertain_dynamic": bool(uncertain_dynamic_files),
    }


def collect_test_evidence(root: Path, impact: dict, settings: dict) -> dict[str, set[int]]:
    tests_by_file: dict[str, set[int]] = defaultdict(set)
    for number, test in enumerate(impact["tests"]):
        tests_by_file[test.split("::", 1)[0]].add(number)
    words: dict[str, set[int]] = defaultdict(set)
    scoped_inputs: dict[str, set[int]] = defaultdict(set)
    test_symbols: dict[str, set[int]] = defaultdict(set)
    for name, tests in tests_by_file.items():
        path = root / name
        if path.suffix != ".py" or not path.is_file():
            continue
        tree = ast.parse(path.read_bytes(), filename=str(path))
        aliases = {
            alias.asname or alias.name: alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name in settings["names"]
        }
        for node in ast.walk(tree):
            found, dynamic = setting_reference(node, settings["names"], aliases)
            for setting in found:
                if setting in settings["references"]:
                    settings["references"][setting].update(tests)
                if setting in settings["module_references"]:
                    settings["module_references"][setting].update(tests)
            if dynamic:
                settings["dynamic_tests"].update(tests)
                for setting in found:
                    settings["dynamic_settings"][setting].update(tests)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for word in input_words(node.value):
                    words[word].update(tests)
        index_test_contexts(
            tree,
            path.read_text(encoding="utf-8"),
            name,
            impact["tests"],
            tests,
            scoped_inputs,
            test_symbols,
        )
    for name, tests in impact["fixtures"].items():
        path = root / name
        if path.suffix == ".sql" and path.is_file():
            for word in input_words(path.read_text(encoding="utf-8")):
                words[word].update(tests)
                scoped_inputs[word].update(tests)
                test_symbols[word].update(tests)
            for key in numeric_escape_keys(path.read_text(encoding="utf-8")):
                scoped_inputs[key].update(tests)
    settings["scoped_inputs"] = scoped_inputs
    settings["test_symbols"] = test_symbols
    return words


def index_test_contexts(
    tree: ast.AST,
    source: str,
    filename: str,
    test_ids: list[str],
    indexes: set[int],
    scoped_inputs: dict[str, set[int]],
    test_symbols: dict[str, set[int]],
) -> None:
    functions: dict[tuple[tuple[str, ...], str], list[ast.FunctionDef | ast.AsyncFunctionDef]] = defaultdict(list)
    classes: dict[tuple[str, ...], ast.ClassDef] = {}

    def visit(node: ast.AST, parents: tuple[str, ...] = ()) -> None:
        if isinstance(node, ast.ClassDef):
            path = (*parents, node.name)
            classes[path] = node
            for child in node.body:
                visit(child, path)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[(parents, node.name)].append(node)
            for child in node.body:
                visit(child, parents)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, parents)

    visit(tree)
    for number in indexes:
        test_id = test_ids[number]
        parts = test_id.split("::")
        if len(parts) < 2 or parts[0] != filename:
            continue
        method = parts[-1].split("[", 1)[0]
        class_path = tuple(parts[1:-1])
        matches = functions.get((class_path, method), ())
        if not matches:
            matches = [
                function
                for (classes_, name), nodes in functions.items()
                if name == method
                and (not class_path or classes_ == class_path)
                for function in nodes
            ]
        if len(matches) != 1:
            continue
        contexts: list[ast.AST] = [matches[0]]
        for length in range(1, len(class_path) + 1):
            class_node = classes.get(class_path[:length])
            if class_node is None:
                continue
            contexts.extend(class_node.decorator_list)
            contexts.extend(class_node.bases)
            contexts.extend(
                statement
                for statement in class_node.body
                if not isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            )
        for context in contexts:
            for node in ast.walk(context):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    segment = ast.get_source_segment(source, node) or node.value
                    for word in input_words(node.value):
                        scoped_inputs[word].add(number)
                        for variant in identifier_variants(word):
                            test_symbols[variant].add(number)
                    for key in numeric_escape_keys(segment):
                        scoped_inputs[key].add(number)
                elif isinstance(node, ast.Name):
                    for variant in identifier_variants(node.id):
                        test_symbols[variant].add(number)
                elif isinstance(node, ast.Attribute):
                    for variant in identifier_variants(node.attr):
                        test_symbols[variant].add(number)
        for class_name in class_path:
            for variant in identifier_variants(class_name):
                test_symbols[variant].add(number)


def build_owner_tests(settings: dict) -> dict[str, list[int]]:
    symbols = settings.get("test_symbols", {})
    result: dict[str, set[int]] = defaultdict(set)
    for name, definitions in settings["class_definitions"].items():
        for path, owner in definitions:
            key = f"{path}::{owner}"
            for variant in owner_variants(owner):
                result[key].update(symbols.get(variant, ()))
    return {key: sorted(tests) for key, tests in result.items()}


def build_dependency_map(root: Path, impact: dict, data) -> dict:
    settings = collect_source_references(root, impact, data)
    inputs = collect_test_evidence(root, impact, settings)
    return {
        "settings": {name: sorted(tests) for name, tests in settings["references"].items()},
        "class_inheritance": settings["inheritance"],
        "class_setting_definitions": {
            name: [
                {"file": path, "class": owner}
                for path, owner in sorted(owners)
            ]
            for name, owners in settings["class_definitions"].items()
        },
        "module_constants": {
            name: sorted(tests)
            for name, tests in settings["module_references"].items()
        },
        "module_constant_definitions": {
            name: sorted(paths)
            for name, paths in settings["module_definitions"].items()
        },
        "uncertain_settings": sorted(settings["uncertain"]),
        "dynamic_tests": sorted(settings["dynamic_tests"]),
        "dynamic_settings": {
            name: sorted(tests)
            for name, tests in settings["dynamic_settings"].items()
        },
        "dynamic_by_file": {
            path: sorted(tests)
            for path, tests in settings["dynamic_by_file"].items()
        },
        "uncertain_dynamic_files": sorted(settings["uncertain_dynamic_files"]),
        "uncertain_dynamic": settings["uncertain_dynamic"],
        "inputs": {word: sorted(tests) for word, tests in inputs.items()},
        "scoped_inputs": {
            word: sorted(tests) for word, tests in settings["scoped_inputs"].items()
        },
        "owner_tests": build_owner_tests(settings),
        "equivalence": settings["equivalence"],
    }


def matching_inputs(index: dict[str, list[int]], text: str) -> set[int]:
    words = input_words(text)
    if not words:
        raise ValueError("Provide at least one input word")
    return matching_input_words(index, words)


def evidence_for_test(index: dict, number: int) -> dict:
    return {
        "input_words": sorted(word for word, tests in index["inputs"].items() if number in tests),
        "setting_names": sorted(name for name, tests in index["settings"].items() if number in tests),
        "dynamic_reference": number in index["dynamic_tests"],
    }


def coverage_for_test(impact: dict, number: int) -> dict:
    files: dict[str, dict[str, int | str]] = {}
    for key in ("functions", "statements"):
        for path, entries in impact[key].items():
            count = sum(number in entry[2] for entry in entries)
            if count:
                files.setdefault(path, {"file": path, "functions": 0, "statements": 0})[key] = count
    return {
        "covered_files": sorted(files.values(), key=lambda entry: str(entry["file"])),
        "fixtures": sorted(path for path, tests in impact["fixtures"].items() if number in tests),
        "source_reads": sorted(path for path, tests in impact["source_reads"].items() if number in tests),
        "metadata_sensitive": number in impact["metadata_tests"],
        "untraced_process_fallback": number in impact["process_tests"],
    }


def patch_reasons(
    impact: dict, number: int, changes: dict[str, set[int]] | None, unchanged_code: frozenset[str],
) -> list[str]:
    from selector import selection_evidence

    fallback, evidence = selection_evidence(changes, impact, unchanged_code)
    return [fallback] if fallback is not None else [
        reason for reason, tests in evidence if number in tests
    ]


def evidence_for_patch(
    impact: dict, number: int, patch_path: Path, dependency: dict | None = None,
    root: Path = Path("/source"),
) -> dict:
    from selector import inspect_patch, patch_changes, select_tests, selection_evidence

    text = patch_path.read_text(encoding="utf-8")
    try:
        unchanged_code = inspect_patch(
            text,
            root,
            dependency.get("equivalence") if dependency is not None else None,
        )
        changes = patch_changes(text)
        all_ids = set(impact["tests"])
        selected = select_tests(changes, impact, all_ids, unchanged_code, dependency, text, root)
        fallback, _ = selection_evidence(changes, impact, unchanged_code)
        if impact["tests"][number] not in selected:
            reasons = []
        elif fallback is not None and selected != all_ids:
            reasons = ["selected by static setting/input heuristic or conservative retention"]
        else:
            reasons = patch_reasons(impact, number, changes, unchanged_code)
    except (OSError, UnicodeError, ValueError, SyntaxError) as error:
        reasons = [f"cannot safely analyze patch: {error}; full-suite fallback"]
    return {"selected": bool(reasons), "reasons": reasons}


def build() -> None:
    import coverage

    impact = json.loads(Path("/coverage/impact-map.json").read_text())
    data = coverage.CoverageData(basename="/src/.coverage")
    data.read()
    if not data.measured_files():
        raise RuntimeError("No baseline coverage for dependency map")
    result = build_dependency_map(Path("/src"), impact, data)
    Path("/coverage/dependency-map.json").write_text(json.dumps(result))
    print(f"Indexed {len(result['settings'])} setting names and {len(result['inputs'])} input words")


def inspect(kind: str, query: str, patch_path: Path | None = None) -> None:
    impact = json.loads(Path("/app/impact-map.json").read_text())
    index = json.loads(Path("/app/dependency-map.json").read_text())
    if kind == "test":
        number = impact["tests"].index(query)
        result = {
            "query": query,
            **evidence_for_test(index, number),
            "coverage": coverage_for_test(impact, number),
            "complete": False,
        }
        if patch_path is not None:
            result["patch"] = evidence_for_patch(impact, number, patch_path, index)
    elif kind == "input":
        numbers = matching_inputs(index["inputs"], query)
        result = {
            "query": query,
            "candidate_tests": [impact["tests"][number] for number in sorted(numbers)],
            "complete": False,
        }
    else:
        numbers = index["settings"].get(query, ())
        result = {
            "query": query,
            "candidate_tests": [impact["tests"][number] for number in numbers],
            "unattributed_reference": query in index["uncertain_settings"],
            "dynamic_tests": len(index["dynamic_tests"]),
            "unattributed_dynamic_reference": index["uncertain_dynamic"],
            "complete": False,
        }
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("build", "setting", "input", "test"))
    parser.add_argument("query", nargs="?")
    parser.add_argument("--patch", type=Path, help="Explain a test's selection for this patch")
    args = parser.parse_args()
    if args.patch is not None and args.kind != "test":
        parser.error("--patch requires a test query")
    if args.kind == "build":
        if args.query:
            parser.error("build does not take a query")
        build()
    elif not args.query:
        parser.error(f"{args.kind} requires a query")
    else:
        inspect(args.kind, args.query, args.patch)


if __name__ == "__main__":
    main()
