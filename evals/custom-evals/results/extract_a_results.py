#!/usr/bin/env python3
"""
Extract responses whose ids start with 'a' from evals/custom-evals/results.jsonl
and write them to a plain text file with separators.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path


# Path to the results.jsonl produced by run_custom_eval.py
RESULTS_PATH = Path(__file__).resolve().parents[3] / "evals" / "custom-evals" / "results" / "results.jsonl"

# Where to write the extracted responses (same folder as this script).
OUTPUT_PATH = Path(__file__).with_name("a_responses.txt")

SEPARATOR = "=" * 19  # "===================" as requested
WRAP_WIDTH = 82

_LIST_PREFIX_RE = re.compile(r"^(\s*(?:[-*]|\d+\.)\s+)(.*)$")


def iter_records(path: Path):
    """Yield decoded JSON objects from a .jsonl file."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc


def wrap_text(text: str, width: int = WRAP_WIDTH) -> str:
    """
    Wrap text to a fixed width while preserving existing line breaks.

    - Leaves fenced code blocks (``` ... ```) untouched.
    - Leaves indented lines untouched (likely code).
    - Wraps list items while preserving their prefix indentation.
    """
    wrapper = textwrap.TextWrapper(
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    wrapped_lines: list[str] = []
    in_fence = False
    for line in text.rstrip().splitlines():
        stripped = line.strip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            wrapped_lines.append(line)
            continue

        if in_fence or line.startswith((" ", "\t")) or len(line) <= width:
            wrapped_lines.append(line)
            continue

        match = _LIST_PREFIX_RE.match(line)
        if match:
            prefix, rest = match.groups()
            list_wrapper = textwrap.TextWrapper(
                width=width,
                initial_indent=prefix,
                subsequent_indent=" " * len(prefix),
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped_lines.extend(list_wrapper.wrap(rest) or [prefix.rstrip()])
        else:
            wrapped_lines.extend(wrapper.wrap(line) or [""])

    return "\n".join(wrapped_lines)


def main() -> None:
    records = [
        rec
        for rec in iter_records(RESULTS_PATH)
        if isinstance(rec, dict) and str(rec.get("id", "")).startswith("a")
    ]

    if not records:
        print("No records with ids starting with 'a' were found.")
        return

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for idx, rec in enumerate(records):
            out.write(f"{rec.get('id', '')}\n")
            response = wrap_text(str(rec.get("response", ""))).rstrip()
            out.write(f"{response}\n")
            if idx < len(records) - 1:
                out.write(f"{SEPARATOR}\n")

    rel_path = OUTPUT_PATH.relative_to(Path.cwd())
    print(f"Wrote {len(records)} responses to {rel_path}")


if __name__ == "__main__":
    main()
