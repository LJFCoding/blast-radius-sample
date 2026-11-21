#!/usr/bin/env python3
"""Categorise baseline tests for one patch and immediately identify eliminated tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from selector import categorize_patch


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patch", type=Path)
    parser.add_argument("--tests", type=Path, default=Path("/data/tests.json"))
    parser.add_argument("--impact", type=Path, default=Path("/app/impact-map.json"))
    parser.add_argument("--dependency", type=Path, default=Path("/app/dependency-map.json"))
    parser.add_argument("--source", type=Path, default=Path("/source"))
    args = parser.parse_args()

    tests = load_json(args.tests)["tests"]
    impact = load_json(args.impact)
    dependency = load_json(args.dependency) if args.dependency.is_file() else None
    all_ids = {test["id"] for test in tests}
    result = categorize_patch(
        args.patch.read_text(encoding="utf-8"),
        impact,
        all_ids,
        dependency,
        args.source,
    )
    result["patch"] = args.patch.name
    json.dump(result, sys.stdout, separators=(",", ":"), sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
