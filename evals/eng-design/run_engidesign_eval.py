#!/usr/bin/env python3
"""
Helper executable that loads a Response_structure model + evaluator function for a
single EngiDesign-Open task and evaluates one serialized LLM response.

The script reads the JSON payload for the response from STDIN and prints a JSON
object with fields: passed, details, score, max_score.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable


def load_attribute(module_path: Path, attr_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_path} has no attribute '{attr_name}'") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a single EngiDesign response.")
    parser.add_argument("--output-structure", required=True, type=Path)
    parser.add_argument("--response-class", default="Response_structure")
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--function", default="evaluate_llm_response")
    return parser.parse_args()


def to_response_object(model_cls, data: Any):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(data)
    return model_cls(**data)


def normalize_result(result):
    passed = False
    details = None
    score = None
    max_score = None

    if isinstance(result, tuple):
        if len(result) >= 1:
            passed = bool(result[0])
        if len(result) >= 2:
            details = result[1]
        if len(result) >= 3:
            score = result[2]
        if len(result) >= 4:
            max_score = result[3]
    elif isinstance(result, dict):
        passed = bool(result.get("passed", False))
        details = result.get("details")
        score = result.get("score")
        max_score = result.get("max_score") or result.get("total")
    elif isinstance(result, bool):
        passed = result

    return {
        "passed": passed,
        "details": details,
        "score": score,
        "max_score": max_score,
    }


def main() -> None:
    args = parse_args()
    raw_payload = sys.stdin.read()
    if not raw_payload.strip():
        raise SystemExit("No JSON payload was provided on stdin.")

    data = json.loads(raw_payload)

    # Ensure the task directory is importable for relative imports like `from output_structure import ...`.
    task_dir = args.evaluator.parent
    for path in {task_dir, args.output_structure.parent}:
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        response_cls = load_attribute(args.output_structure, args.response_class)
        evaluator_fn: Callable[..., Any] = load_attribute(args.evaluator, args.function)
    except Exception as exc:
        normalized = {
            "passed": False,
            "details": f"Import failed: {exc}",
            "score": None,
            "max_score": None,
            "error": "import_error",
        }
        print(json.dumps(normalized, default=str))
        return

    try:
        response_obj = to_response_object(response_cls, data)
    except Exception as exc:
        # Surface validation failures as a normalized failed result instead of crashing.
        normalized = {
            "passed": False,
            "details": f"Validation failed: {exc}",
            "score": None,
            "max_score": None,
            "error": "validation_error",
        }
        print(json.dumps(normalized, default=str))
        return

    result = evaluator_fn(response_obj)
    normalized = normalize_result(result)
    print(json.dumps(normalized, default=str))


if __name__ == "__main__":
    main()
