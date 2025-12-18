#!/usr/bin/env python3
"""
Utility to collapse the EngDesign-Open directory layout into a single JSONL
metadata file that can be consumed by automated evaluation scripts.

Each task directory is expected to contain at least the following files:
  - LLM_prompt.txt
  - output_structure.py (with a `Response_structure` Pydantic model)
  - evaluate.py        (with an `evaluate_llm_response` function)

Optional metadata (domain topic, task provider, rubrics) will be copied into
the resulting JSON entries when available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "README.md").exists() and (candidate / "evals").exists():
            return candidate
    return start


REPO_ROOT = find_repo_root(SCRIPT_DIR)

def path_from_env(var_name: str, default: Path) -> Path:
    override = os.environ.get(var_name)
    if override:
        return Path(override).expanduser()
    return default


ENGIDESIGN_ROOT = path_from_env(
    "ENGIDESIGN_ROOT", REPO_ROOT / "evals" / "eng-design" / "EngDesign-Open"
)
OUTPUT_PATH = path_from_env(
    "ENGIDESIGN_DATASET_PATH", SCRIPT_DIR / "engidesign_open_dataset.jsonl"
)

def read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def build_entry(task_dir: Path) -> Optional[Dict[str, object]]:
    prompt_path = task_dir / "LLM_prompt.txt"
    output_structure = task_dir / "output_structure.py"
    evaluator = task_dir / "evaluate.py"

    if not prompt_path.exists():
        print(f"[WARN] Skipping {task_dir.name}: missing LLM_prompt.txt")
        return None
    if not output_structure.exists():
        print(f"[WARN] Skipping {task_dir.name}: missing output_structure.py")
        return None
    if not evaluator.exists():
        print(f"[WARN] Skipping {task_dir.name}: missing evaluate.py")
        return None

    prompt_text = read_text(prompt_path)
    domain_topic = read_text(task_dir / "domain_topic.txt")
    task_provider = read_text(task_dir / "task_provider.txt")
    evaluation_notes = read_text(task_dir / "evaluation_pipeline.txt")

    # Paths relative to repo root keep JSON portable.
    try:
        rel_output_structure = output_structure.relative_to(REPO_ROOT).as_posix()
        rel_evaluator = evaluator.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel_output_structure = os.path.relpath(output_structure, REPO_ROOT).replace(os.sep, "/")
        rel_evaluator = os.path.relpath(evaluator, REPO_ROOT).replace(os.sep, "/")

    return {
        "id": task_dir.name,
        "prompt": prompt_text,
        "domain_topic": domain_topic,
        "task_provider": task_provider,
        "evaluation_pipeline": evaluation_notes,
        "output_structure": {
            "path": rel_output_structure,
            "class": "Response_structure",
        },
        "evaluator": {
            "path": rel_evaluator,
            "function": "evaluate_llm_response",
        },
    }


def main() -> None:
    if not ENGIDESIGN_ROOT.exists():
        raise SystemExit(f"Could not find EngDesign-Open directory at {ENGIDESIGN_ROOT}")

    entries = []
    for task_dir in sorted(ENGIDESIGN_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue
        entry = build_entry(task_dir)
        if entry is not None:
            entries.append(entry)

    if not entries:
        raise SystemExit("No EngDesign tasks were found; aborting.")

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} tasks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
