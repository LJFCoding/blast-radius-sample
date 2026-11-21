"""Collect per-test coverage and fixture reads while running the pinned suite."""

import ast
import atexit
import concurrent.futures
import enum
import importlib.abc
import importlib.machinery
import enum
import inspect
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
MAX_TRACED_MAPPING_KEYS = 64
MAX_TRACED_KEYS_PER_TEST = 256
TRACE_MAPPING_NAMES = frozenset({
    "FUNCTIONS",
    "PROPERTY_PARSERS",
    "SHOW_PARSERS",
})
METADATA_ATTRIBUTES = {
    "__code__", "__doc__", "co_firstlineno", "co_lines", "co_positions",
    "currentframe", "extract_tb", "f_code", "f_lineno", "format_exc",
    "format_exception", "format_stack", "stack", "tb_lineno", "_getframe",
}
_TRACE_SPECS: dict[str, dict[int, list[str]]] | None = None
_trace_collector = None


def _reference_path(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        base = _reference_path(node.value)
        return (*base, node.attr) if base else None
    return None


def _mapping_reference(node: ast.AST) -> str | None:
    reference = _reference_path(node)
    if not reference:
        return None
    name = reference[-1].upper()
    return name if name in TRACE_MAPPING_NAMES else None


def _mapping_names(tree: ast.AST) -> set[str]:
    result = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = getattr(node, "value", None)
        if not isinstance(value, ast.Dict) or len(value.keys) > MAX_TRACED_MAPPING_KEYS:
            continue
        result.update(
            name
            for name in assignment_names(node)
            if name in TRACE_MAPPING_NAMES and value.keys
        )
    return result


def assignment_names(node: ast.AST) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    result = set()
    for target in targets:
        reference = _reference_path(target)
        if reference:
            result.add(reference[-1])
    return result


def _trace_specs(source_root: Path) -> dict[str, dict[int, list[str]]]:
    specs: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_bytes(), filename=str(path))
        filename = str(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                container = _mapping_reference(node.value)
                if container:
                    specs[filename][node.lineno].add(container)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"get", "setdefault", "pop"}
            ):
                container = _mapping_reference(node.func.value)
                if container:
                    specs[filename][node.lineno].add(container)
            elif isinstance(node, ast.Compare):
                for operator, comparator in zip(node.ops, node.comparators):
                    if not isinstance(operator, (ast.In, ast.NotIn)):
                        continue
                    container = _mapping_reference(comparator)
                    if container:
                        specs[filename][node.lineno].add(container)
            elif isinstance(node, ast.FunctionDef) and node.name == "_find_parser":
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Subscript)
                        and isinstance(child.value, ast.Name)
                        and child.value.id == "parsers"
                    ):
                        specs[filename][child.lineno].add("AUTO")
    return {
        filename: {
            line: sorted(containers)
            for line, containers in lines.items()
        }
        for filename, lines in specs.items()
        if lines
    }


def _trace_value(value: object, depth: int = 0) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return {
            "__enum__": [
                type(value).__module__,
                type(value).__qualname__,
                _trace_value(value.value, depth + 1),
            ]
        }
    if isinstance(value, tuple) and depth < 2 and len(value) <= 16:
        return {"__tuple__": [_trace_value(item, depth + 1) for item in value]}
    if isinstance(value, list) and depth < 2 and len(value) <= 16:
        return {"__list__": [_trace_value(item, depth + 1) for item in value]}
    return {"__type__": [type(value).__module__, type(value).__qualname__]}


def trace_key(
    container: str,
    key: object,
    owner: object = None,
    mapping: object = None,
) -> object:
    collector = _trace_collector
    if collector is None or not collector.context:
        return key
    if container == "AUTO":
        if owner is None or mapping is None:
            return key
        if getattr(owner, "SHOW_PARSERS", None) is mapping:
            container = "SHOW_PARSERS"
        elif getattr(owner, "SET_PARSERS", None) is mapping:
            container = "SET_PARSERS"
        else:
            return key
    owner_type = owner if inspect.isclass(owner) else type(owner)
    qualified_container = (
        f"{owner_type.__module__}::{owner_type.__qualname__}::{container}"
        if owner is not None
        else container
    )
    count = collector.trace_key_counts[collector.context]
    if count >= MAX_TRACED_KEYS_PER_TEST:
        signature = json.dumps(
            {"__unknown__": True},
            sort_keys=True,
            separators=(",", ":"),
        )
        collector.key_values[(qualified_container, signature)].add(collector.context)
        return key
    identity = (
        qualified_container,
        collector.context,
        type(key).__module__,
        type(key).__qualname__,
        repr(key)[:256],
    )
    if identity in collector.trace_key_seen:
        return key
    collector.trace_key_seen.add(identity)
    collector.trace_key_counts[collector.context] = count + 1
    signature = json.dumps(
        {"value": _trace_value(key)},
        sort_keys=True,
        separators=(",", ":"),
    )
    collector.key_values[(qualified_container, signature)].add(collector.context)
    return key
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
    collector = _trace_collector
    if collector is not None:
        if not context or getattr(collector.local, "instrumenting_import", False):
            return
    path = source_file(name)
    if not path or importing_source():
        return
    if context:
        reads[path].add(context)
    else:
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
        self.member_reads: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
        self.key_values: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.trace_key_seen: set[tuple] = set()
        self.trace_key_counts: dict[str, int] = defaultdict(int)
        self.test_domains: dict[str, dict[str, list]] = {}
        self.uncertain_source_reads: set[str] = set()
        self.local = threading.local()
        setup_value_trace(self)
        sys.addaudithook(self.record_event)
        track_linecache(self)
        track_inspect(self)
        self.coverage = coverage.Coverage(
            data_file=str(OUTPUT / f".coverage.{worker}.child.{self.pid}"),
            source=[str(SOURCE)],
            config_file=False,
        )
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
        finish_value_trace(self)
        self.coverage.stop()
        self.coverage.save()
        (OUTPUT / f"child.{self.worker}.{self.pid}.json").write_text(json.dumps({
            "contexts": sorted(self.contexts),
            "process_contexts": sorted(self.process_contexts),
            "fixtures": {path: sorted(contexts) for path, contexts in self.fixtures.items()},
            "source_reads": {path: sorted(contexts) for path, contexts in self.source_reads.items()},
            "member_reads": [
                {
                    "path": key[0],
                    "member": key[1],
                    "start": key[2],
                    "end": key[3],
                    "contexts": sorted(contexts),
                }
                for key, contexts in self.member_reads.items()
            ],
            "key_values": [
                {
                    "container": key[0],
                    "signature": key[1],
                    "contexts": sorted(contexts),
                }
                for key, contexts in self.key_values.items()
            ],
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


def track_inspect(collector) -> None:
    original_getsourcelines = inspect.getsourcelines
    original_getcomments = inspect.getcomments

    def record_member(obj, start: int, end: int) -> None:
        try:
            filename = inspect.getsourcefile(obj) or inspect.getfile(obj)
        except (OSError, TypeError):
            return
        path = source_file(filename)
        if not path:
            return
        member = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
        collector.member_reads[(path, member, start, end)].add(collector.context)
        if not collector.context:
            collector.uncertain_source_reads.add(path)

    @wraps(original_getsourcelines)
    def getsourcelines(obj):
        result = original_getsourcelines(obj)
        record_member(obj, result[1], result[1] + len(result[0]) - 1)
        return result

    @wraps(original_getcomments)
    def getcomments(obj):
        result = original_getcomments(obj)
        try:
            lines, start = original_getsourcelines(obj)
        except (OSError, TypeError):
            lines, start = [], 1
        if lines:
            record_member(obj, start, start + len(lines) - 1)
        return result

    inspect.getsourcelines = getsourcelines
    inspect.getcomments = getcomments


class TraceTransformer(ast.NodeTransformer):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.specs = _TRACE_SPECS.get(filename, {}) if _TRACE_SPECS else {}
        self.owner_args: list[ast.expr | None] = []
        self.function_names: list[str] = []

    def visit_Module(self, node: ast.Module):
        node = self.generic_visit(node)
        import_node = ast.ImportFrom(
            module="impact_collector",
            names=[ast.alias(name="trace_key", asname="_blast_trace_key")],
            level=0,
        )
        position = 1 if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ) else 0
        while (
            position < len(node.body)
            and isinstance(node.body[position], ast.ImportFrom)
            and node.body[position].module == "__future__"
        ):
            position += 1
        node.body.insert(position, import_node)
        return node

    def _trace_key(
        self,
        line: int,
        container: str,
        key: ast.expr,
        mapping: ast.expr | None = None,
    ) -> ast.expr:
        if container not in self.specs.get(line, ()):
            return key
        traced = ast.Call(
            func=ast.Name(id="_blast_trace_key", ctx=ast.Load()),
            args=[
                ast.Constant(container),
                key,
                self.owner_args[-1] if self.owner_args else ast.Constant(None),
                mapping if mapping is not None else ast.Constant(None),
            ],
            keywords=[],
        )
        ast.copy_location(traced, key)
        return traced

    def visit_Subscript(self, node: ast.Subscript):
        node = self.generic_visit(node)
        container = _mapping_reference(node.value)
        if (
            container is None
            and self.function_names[-1:] == ["_find_parser"]
            and isinstance(node.value, ast.Name)
            and node.value.id == "parsers"
        ):
            container = "AUTO"
        if container:
            node.slice = self._trace_key(
                node.lineno, container, node.slice, node.value
            )
        return node

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "setdefault", "pop"}
            and node.args
        ):
            container = _mapping_reference(node.func.value)
            if container:
                node.args[0] = self._trace_key(
                    node.lineno, container, node.args[0]
                )
        return node

    def visit_Compare(self, node: ast.Compare):
        node = self.generic_visit(node)
        left = node.left
        for operator, comparator in zip(node.ops, node.comparators):
            if not isinstance(operator, (ast.In, ast.NotIn)):
                continue
            container = _mapping_reference(comparator)
            if container:
                left = self._trace_key(node.lineno, container, left)
                break
        node.left = left
        return node

    def _visit_function(self, node):
        positional = [*node.args.posonlyargs, *node.args.args]
        owner = (
            ast.Name(id=positional[0].arg, ctx=ast.Load())
            if positional
            and positional[0].arg in {"self", "cls"}
            else ast.Constant(None)
        )
        self.owner_args.append(owner)
        self.function_names.append(node.name)
        node = self.generic_visit(node)
        self.function_names.pop()
        self.owner_args.pop()
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self._visit_function(node)


def setup_value_trace(collector) -> None:
    global _TRACE_SPECS, _trace_collector
    if _TRACE_SPECS is None:
        _TRACE_SPECS = _trace_specs(SOURCE)
    if _trace_collector is not None:
        raise RuntimeError("Value trace collector is already active")
    _trace_collector = collector

    class TraceLoader(importlib.abc.Loader):
        def __init__(self, wrapped) -> None:
            self.wrapped = wrapped

        def create_module(self, spec):
            create = getattr(self.wrapped, "create_module", None)
            return create(spec) if create is not None else None

        def exec_module(self, module) -> None:
            filename = module.__spec__.origin
            source_getter = getattr(self.wrapped, "get_source", None)
            collector = _trace_collector
            try:
                if collector is not None:
                    collector.local.instrumenting_import = True
                if filename not in _TRACE_SPECS or source_getter is None:
                    self.wrapped.exec_module(module)
                    return
                source = source_getter(module.__name__)
                if source is None:
                    self.wrapped.exec_module(module)
                    return
                tree = ast.parse(source, filename=filename)
                tree = TraceTransformer(filename).visit(tree)
                ast.fix_missing_locations(tree)
                code = compile(tree, filename, "exec")
                exec(code, module.__dict__)
                return
            except (RecursionError, SyntaxError, TypeError) as error:
                print(
                    f"Skipping key trace for {filename}: {error}",
                    file=sys.stderr,
                )
                self.wrapped.exec_module(module)
                return
            finally:
                if collector is not None:
                    collector.local.instrumenting_import = False

    class TraceFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if not fullname.startswith("sqlglot"):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = TraceLoader(spec.loader)
            return spec

    collector.value_trace_finder = TraceFinder()
    sys.meta_path.insert(0, collector.value_trace_finder)


def finish_value_trace(collector) -> None:
    global _trace_collector
    if _trace_collector is collector:
        _trace_collector = None
    finder = getattr(collector, "value_trace_finder", None)
    if finder is not None:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass


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
        self.member_reads: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
        self.key_values: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.trace_key_seen: set[tuple] = set()
        self.trace_key_counts: dict[str, int] = defaultdict(int)
        self.test_domains: dict[str, dict[str, list]] = {}
        self.uncertain_source_reads: set[str] = set()
        self.process_contexts: set[str] = set()
        self.synchronous_contexts: set[str] = set()
        self.pool_contexts: set[str] = set()
        self.context = ""
        self.local = threading.local()
        setup_value_trace(self)
        sys.addaudithook(self.record_open)
        track_linecache(self)
        track_inspect(self)
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
        self.test_domains[item.nodeid] = test_domain(item)
        self.set_context(f"test:{item.nodeid}")

    def pytest_sessionfinish(self, session, exitstatus):
        finish_value_trace(self)
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
                    "member_reads": [
                        {
                            "path": key[0],
                            "member": key[1],
                            "start": key[2],
                            "end": key[3],
                            "contexts": sorted(contexts),
                        }
                        for key, contexts in self.member_reads.items()
                    ],
                    "key_values": [
                        {
                            "container": key[0],
                            "signature": key[1],
                            "contexts": sorted(contexts),
                        }
                        for key, contexts in self.key_values.items()
                    ],
                    "test_domains": self.test_domains,
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


def _domain_value(value: object, depth: int = 0) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return {"__symbol__": value.name}
    if isinstance(value, tuple) and depth < 2 and len(value) <= 16:
        return {"__tuple__": [_domain_value(item, depth + 1) for item in value]}
    if isinstance(value, (list, set)) and depth < 2 and len(value) <= 16:
        return {
            "__list__": [_domain_value(item, depth + 1) for item in value]
        }
    value_type = type(value)
    return {
        "__type__": [
            value_type.__module__,
            value_type.__qualname__,
        ]
    }


def test_domain(item) -> dict[str, list[str]]:
    values: dict[str, set[str]] = defaultdict(set)

    def add(name: str, value: object) -> None:
        for key in {name, name.lower()}:
            encoded = json.dumps(
                _domain_value(value),
                sort_keys=True,
                separators=(",", ":"),
            )
            values[key].add(encoded)

    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        for name, value in callspec.params.items():
            add(name, value)
    for source in (getattr(item, "funcargs", None), getattr(item, "instance", None)):
        try:
            entries = dict(vars(source)) if source is not None else {}
        except TypeError:
            entries = source if isinstance(source, dict) else {}
        for name, value in entries.items():
            if not name.startswith("_"):
                add(name, value)
    item_class = getattr(item, "cls", None)
    if item_class is not None:
        for base in item_class.__mro__:
            for name, value in vars(base).items():
                if (
                    not name.startswith("_")
                    and not callable(value)
                    and isinstance(value, (bool, int, float, str, tuple, list, set, enum.Enum))
                ):
                    add(name, value)
    for source in (getattr(item, "instance", None), getattr(item, "cls", None)):
        for attribute in ("dialect", "version", "read", "write", "generator", "parser"):
            try:
                value = getattr(source, attribute)
            except (AttributeError, TypeError):
                continue
            add(attribute, value)
    return {
        name: sorted(encoded)
        for name, encoded in values.items()
    }


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
    member_contexts: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    key_contexts: dict[tuple[str, str], set[str]] = defaultdict(set)
    test_domains = {}
    uncertain_source_reads = set()
    for record in records + children:
        for path, contexts in record["source_reads"].items():
            source_contexts[path].update(contexts)
        for entry in record.get("member_reads", ()):
            key = (
                entry["path"],
                entry["member"],
                int(entry["start"]),
                int(entry["end"]),
            )
            member_contexts[key].update(entry["contexts"])
        for entry in record.get("key_values", ()):
            key_contexts[(
                entry["container"],
                entry["signature"],
            )].update(entry["contexts"])
        test_domains.update(record.get("test_domains", {}))
        uncertain_source_reads.update(record["uncertain_source_reads"])
    source_reads = {
        path: sorted(expanded_contexts(contexts, groups))
        for path, contexts in source_contexts.items()
    }
    member_reads: dict[str, list[list]] = defaultdict(list)
    for (path, member, start, end), contexts in sorted(member_contexts.items()):
        member_reads[path].append([
            member,
            start,
            end,
            sorted(expanded_contexts(contexts, groups)),
        ])
    key_values = [
        [container, signature, sorted(expanded_contexts(contexts, groups))]
        for (container, signature), contexts in sorted(key_contexts.items())
    ]

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
            "member_reads": dict(member_reads),
            "key_values": key_values,
            "test_domains": [
                test_domains.get(test, {})
                for test in sorted(tests)
            ],
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
