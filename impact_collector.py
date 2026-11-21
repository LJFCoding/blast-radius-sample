"""Collect per-test coverage and fixture reads while running the pinned suite."""

import ast
import atexit
import concurrent.futures
import json
import linecache
import multiprocessing.process
import multiprocessing.reduction
import os
import subprocess
import sys
import threading
from collections import defaultdict
from functools import wraps
from pathlib import Path

import coverage
import pytest

ROOT = Path("/src")
OUTPUT = Path("/coverage")
SOURCE = ROOT / "sqlglot"
FIXTURES = ROOT / "tests" / "fixtures"
METADATA_ATTRIBUTES = {
    "__code__", "__doc__", "co_firstlineno", "co_lines", "co_positions",
    "currentframe", "extract_tb", "f_code", "f_lineno", "format_exc",
    "format_exception", "format_stack", "getcomments", "getfile", "getsource",
    "getsourcefile", "getsourcelines", "read_bytes", "read_text", "stack",
    "tb_lineno", "_getframe",
}


def source_file(name: str | bytes) -> str | None:
    path = os.fsdecode(name)
    if not path.startswith((str(SOURCE) + "/", "sqlglot/", "./sqlglot/")):
        return None
    path = os.path.abspath(path)
    if path.startswith(str(SOURCE) + "/") and path.endswith(".py"):
        return str(Path(path).relative_to(ROOT))
    return None


def importing_source() -> bool:
    frame = sys._getframe(1)
    while frame:
        if frame.f_code.co_filename.startswith(str(ROOT) + "/"):
            return False
        if frame.f_code.co_filename.startswith("<frozen importlib._bootstrap_external>"):
            return True
        frame = frame.f_back
    return False


def record_source(
    name: str | bytes, context: str, reads: dict[str, set[str]], uncertain: set[str]
) -> None:
    path = source_file(name)
    if path:
        if context:
            reads[path].add(context)
        elif not importing_source():
            uncertain.add(path)


class ChildCollector:
    def __init__(self, worker: str) -> None:
        self.worker = worker
        self.pid = os.getpid()
        self.context = ""
        self.contexts: set[str] = set()
        self.process_contexts: set[str] = set()
        self.fixtures: dict[str, set[str]] = defaultdict(set)
        self.source_reads: dict[str, set[str]] = defaultdict(set)
        self.uncertain_source_reads: set[str] = set()
        self.coverage = coverage.Coverage(
            data_file=str(OUTPUT / f".coverage.{worker}.child.{self.pid}"),
            source=[str(SOURCE)],
            config_file=False,
        )
        sys.addaudithook(self.record_event)
        track_linecache(self)
        self.coverage.start()
        atexit.register(self.finish)

    def record_event(self, event: str, args: tuple) -> None:
        record_event(
            event, args, self.context, self.fixtures, self.process_contexts,
            self.source_reads, self.uncertain_source_reads,
        )

    def set_context(self, context: str) -> None:
        self.context = context
        if context:
            self.contexts.add(context)
        self.coverage.switch_context(context)

    def finish(self) -> None:
        self.coverage.stop()
        self.coverage.save()
        (OUTPUT / f"child.{self.worker}.{self.pid}.json").write_text(json.dumps({
            "contexts": sorted(self.contexts),
            "process_contexts": sorted(self.process_contexts),
            "fixtures": {path: sorted(contexts) for path, contexts in self.fixtures.items()},
            "source_reads": {path: sorted(contexts) for path, contexts in self.source_reads.items()},
            "uncertain_source_reads": sorted(self.uncertain_source_reads),
        }))


_child_collector: ChildCollector | None = None


def _trace_pool_call(worker, context, payload):
    global _child_collector
    if _child_collector is None:
        _child_collector = ChildCollector(worker)
    _child_collector.set_context(context)
    function, args, kwargs = multiprocessing.reduction.ForkingPickler.loads(payload)
    return function(*args, **kwargs)


def record_event(
    event: str,
    args: tuple,
    context: str,
    fixtures: dict[str, set[str]],
    process_contexts: set[str],
    source_reads: dict[str, set[str]],
    uncertain_source_reads: set[str],
) -> None:
    if event in {"subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"}:
        if context:
            process_contexts.add(context)
        return
    if event != "open" or not isinstance(args[0], (str, bytes)):
        return
    path = os.fsdecode(args[0])
    record_source(path, context, source_reads, uncertain_source_reads)
    if not context:
        return
    prefix = str(FIXTURES) + "/"
    if path.startswith(prefix):
        fixtures[str(Path(path).relative_to(ROOT))].add(context)


def track_linecache(collector) -> None:
    original_getlines = linecache.getlines

    @wraps(original_getlines)
    def getlines(filename, module_globals=None):
        if isinstance(filename, (str, bytes)):
            record_source(
                filename, collector.context, collector.source_reads, collector.uncertain_source_reads
            )
        return original_getlines(filename, module_globals)

    linecache.getlines = getlines


class Collector:
    def __init__(self, worker: str) -> None:
        self.worker = worker
        self.coverage = coverage.Coverage(
            data_file=str(OUTPUT / f".coverage.{worker}"),
            source=[str(SOURCE)],
            config_file=False,
        )
        self.tests: set[str] = set()
        self.fixtures: dict[str, set[str]] = defaultdict(set)
        self.source_reads: dict[str, set[str]] = defaultdict(set)
        self.uncertain_source_reads: set[str] = set()
        self.process_contexts: set[str] = set()
        self.synchronous_contexts: set[str] = set()
        self.pool_contexts: set[str] = set()
        self.context = ""
        self.local = threading.local()
        sys.addaudithook(self.record_open)
        track_linecache(self)
        self.track_processes()
        self.coverage.start()

    def record_open(self, event: str, args: tuple) -> None:
        if getattr(self.local, "tracked_launch", False) and event in {
            "subprocess.Popen", "os.fork", "os.posix_spawn", "os.system"
        }:
            return
        record_event(
            event, args, self.context, self.fixtures, self.process_contexts,
            self.source_reads, self.uncertain_source_reads,
        )

    def track_processes(self) -> None:
        original_start = multiprocessing.process.BaseProcess.start
        original_submit = concurrent.futures.ProcessPoolExecutor.submit
        original_run = subprocess.run

        @wraps(original_start)
        def start(process):
            if self.context and not getattr(self.local, "tracked_launch", False):
                self.process_contexts.add(self.context)
            return original_start(process)

        @wraps(original_submit)
        def submit(executor, function, /, *args, **kwargs):
            context = self.context
            if not context:
                raise RuntimeError("Cannot attribute process pool submission to a test")
            self.pool_contexts.add(context)
            payload = bytes(multiprocessing.reduction.ForkingPickler.dumps((function, args, kwargs)))
            self.local.tracked_launch = True
            try:
                return original_submit(executor, _trace_pool_call, self.worker, context, payload)
            finally:
                self.local.tracked_launch = False

        @wraps(original_run)
        def run(*args, **kwargs):
            if self.context:
                self.synchronous_contexts.add(self.context)
            self.local.tracked_launch = True
            try:
                return original_run(*args, **kwargs)
            finally:
                self.local.tracked_launch = False

        multiprocessing.process.BaseProcess.start = start
        concurrent.futures.ProcessPoolExecutor.submit = submit
        subprocess.run = run

    def set_context(self, context: str) -> None:
        self.context = context
        self.coverage.switch_context(context)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item):
        self.tests.add(item.nodeid)
        self.set_context(f"test:{item.nodeid}")
        yield
        self.set_context("")

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_setup(self, item):
        self.set_context(f"setup:{item.nodeid}")
        yield
        self.set_context(f"test:{item.nodeid}")

    def pytest_sessionfinish(self, session, exitstatus):
        self.coverage.stop()
        self.coverage.save()
        (OUTPUT / f"{self.worker}.json").write_text(
            json.dumps(
                {
                    "tests": sorted(self.tests),
                    "fixtures": {
                        path: sorted(contexts) for path, contexts in self.fixtures.items()
                    },
                    "process_contexts": sorted(self.process_contexts),
                    "synchronous_contexts": sorted(self.synchronous_contexts),
                    "pool_contexts": sorted(self.pool_contexts),
                    "source_reads": {
                        path: sorted(contexts) for path, contexts in self.source_reads.items()
                    },
                    "uncertain_source_reads": sorted(self.uncertain_source_reads),
                }
            )
        )


def pytest_configure(config):
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker:
        config.pluginmanager.register(Collector(worker), "impact-collector")


def _decorated_start(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    return min(
        (decorator.lineno for decorator in node.decorator_list),
        default=node.lineno,
    )


def functions_in(path: Path) -> list[tuple[int, int]]:
    tree = ast.parse(path.read_bytes(), filename=str(path))
    functions: list[tuple[int, int]] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append((_decorated_start(child), child.end_lineno))
                visit(child)
            else:
                visit(child)

    visit(tree)
    return functions


def classes_in(tree: ast.AST) -> list[tuple[int, int]]:
    classes: list[tuple[int, int]] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                classes.append((_decorated_start(child), child.end_lineno))
                visit(child)
            else:
                visit(child)

    visit(tree)
    return classes


def expanded_contexts(contexts: set[str], groups: dict[str, set[str]]) -> set[str]:
    result = set()
    for context in contexts:
        phase, separator, test = context.partition(":")
        if not separator or phase not in {"test", "setup"}:
            continue
        if phase == "setup":
            result.update(groups[test.rsplit("::", 1)[0]])
        else:
            result.add(test)
    return result


def metadata_references(tree: ast.AST, tests: set[str]) -> set[str]:
    tests_by_method: dict[str, set[str]] = defaultdict(set)
    for test in tests:
        method = test.split("[", 1)[0].rsplit("::", 1)[-1]
        tests_by_method[method].add(test)
    test_names = set(tests_by_method)
    selected = set()
    uncertain = False
    has_metadata = False

    def visit(node: ast.AST, current: set[str] | None) -> None:
        nonlocal uncertain, has_metadata
        if isinstance(node, ast.ClassDef):
            for child in node.decorator_list + node.bases + node.keywords:
                visit(child, current)
            for child in node.body:
                visit(child, current)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in node.decorator_list + [node.args]:
                visit(child, current)
            if node.returns is not None:
                visit(node.returns, current)
            for child in getattr(node, "type_params", ()):
                visit(child, current)
            own_tests = current if current is not None else tests_by_method.get(node.name)
            for child in node.body:
                visit(child, own_tests)
            return
        metadata = (
            isinstance(node, ast.Attribute) and node.attr in METADATA_ATTRIBUTES
        ) or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in METADATA_ATTRIBUTES
        )
        if metadata:
            has_metadata = True
            if current is None:
                uncertain = True
            else:
                selected.update(current)
        if (
            isinstance(node, ast.Attribute) and node.attr in test_names
        ) or (
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in test_names
        ) or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and (len(node.args) < 2 or not isinstance(node.args[1], ast.Constant)
                 or node.args[1].value in test_names)
        ):
            uncertain = True
        for child in ast.iter_child_nodes(node):
            visit(child, current)

    visit(tree, None)
    return tests if has_metadata and uncertain else selected


def metadata_tests(tests: set[str]) -> set[str]:
    tests_by_file: dict[str, set[str]] = defaultdict(set)
    for test in tests:
        tests_by_file[test.split("::", 1)[0]].add(test)
    selected = set()
    for name, ids in tests_by_file.items():
        path = ROOT / name
        if path.suffix != ".py" or not path.is_file():
            continue
        tree = ast.parse(path.read_bytes(), filename=str(path))
        selected.update(metadata_references(tree, ids))
    return selected


def build_map() -> None:
    records = [json.loads(path.read_text()) for path in sorted(OUTPUT.glob("gw*.json"))]
    children = [json.loads(path.read_text()) for path in sorted(OUTPUT.glob("child.gw*.json"))]
    if not records:
        raise RuntimeError("No coverage workers produced results")
    tests = set().union(*(set(record["tests"]) for record in records))
    if not tests:
        raise RuntimeError("Coverage workers did not execute any tests")
    groups: dict[str, set[str]] = defaultdict(set)
    for test in tests:
        groups[test.rsplit("::", 1)[0]].add(test)

    expected_contexts = set().union(*(set(record["pool_contexts"]) for record in records))
    child_contexts = set().union(*(set(record["contexts"]) for record in children))
    if expected_contexts != child_contexts:
        raise RuntimeError(
            f"Missing coverage for {len(expected_contexts - child_contexts)} "
            f"process-pool test contexts"
        )

    process_tests = set()
    for record in records + children:
        for context in record["process_contexts"]:
            _, _, test = context.partition(":")
            process_tests.update(groups[test.rsplit("::", 1)[0]])
    for record in records:
        process_tests.update(expanded_contexts(set(record["synchronous_contexts"]), groups))

    fixture_contexts: dict[str, set[str]] = defaultdict(set)
    for record in records + children:
        for path, contexts in record["fixtures"].items():
            fixture_contexts[path].update(contexts)
    fixtures = {
        path: sorted(expanded_contexts(contexts, groups))
        for path, contexts in fixture_contexts.items()
    }
    source_contexts: dict[str, set[str]] = defaultdict(set)
    uncertain_source_reads = set()
    for record in records + children:
        for path, contexts in record["source_reads"].items():
            source_contexts[path].update(contexts)
        uncertain_source_reads.update(record["uncertain_source_reads"])
    source_reads = {
        path: sorted(expanded_contexts(contexts, groups))
        for path, contexts in source_contexts.items()
    }

    data = coverage.CoverageData()
    for path in sorted(OUTPUT.glob(".coverage.gw*")):
        worker_data = coverage.CoverageData(basename=str(path))
        worker_data.read()
        data.update(worker_data)
    functions = {}
    classes = {}
    modules = {}
    for filename in sorted(data.measured_files()):
        path = Path(filename)
        if not path.is_relative_to(SOURCE):
            continue
        tree = ast.parse(path.read_bytes(), filename=filename)
        contexts_by_line = data.contexts_by_lineno(filename)
        relative = str(path.relative_to(ROOT))

        def map_ranges(ranges: list[tuple[int, int]]) -> list[list]:
            entries = []
            for start, end in ranges:
                contexts = set()
                for line in range(start, end + 1):
                    contexts.update(contexts_by_line.get(line, ()))
                entries.append([start, end, sorted(expanded_contexts(contexts, groups))])
            return entries

        functions[relative] = map_ranges(functions_in(path))
        classes[relative] = map_ranges(classes_in(tree))
        module_end = max((node.end_lineno or node.lineno for node in tree.body), default=1)
        modules[relative] = map_ranges([(1, module_end)])

    (OUTPUT / "impact-map.json").write_text(
        json.dumps({
            "tests": sorted(tests),
            "process_tests": sorted(process_tests),
            "source_reads": source_reads,
            "uncertain_source_reads": sorted(uncertain_source_reads),
            "metadata_tests": sorted(metadata_tests(tests)),
            "fixtures": fixtures,
            "functions": functions,
            "classes": classes,
            "modules": modules,
        })
    )
    print(
        f"Mapped {len(tests)} tests, {len(functions)} source files and {len(fixtures)} fixtures; "
        f"{len(process_tests)} tests use potentially untraced processes"
    )


if __name__ == "__main__":
    build_map()
