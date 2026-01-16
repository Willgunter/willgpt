#!/usr/bin/env python3
"""
Generate questions for LONG_ANALYSIS blocks in finetune1_formatted_data_attempt2.txt.

For each LONG_ANALYSIS, ask the model for one realistic question that would elicit
that analysis. Output is a Q/A block where the answer is the original LONG_ANALYSIS
paragraph, matching the dataset style:

---WILLGPTSTART---

Q: <generated question>

A:
<original long analysis text>

Use --limit to test a few samples (e.g., 2 or 3). The original dataset is not
modified; results are written to a separate output file.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from openai import OpenAI

SPLIT_TOKEN = "---WILLGPTSTART---"

SYSTEM_PROMPT = (
    "You are an expert extreme-conditions systems engineer. Given a LONG_ANALYSIS paragraph, "
    "write exactly one concise engineering question that would elicit that analysis. "
    "Focus on assumptions, operating limits, or failure mechanisms. No answers, no bullets."
)

USER_PROMPT_TEMPLATE = (
    "Write one realistic engineering question whose detailed answer would match the LONG_ANALYSIS below.\n"
    "- Make the question specific to the scenario and its failure mechanism or assumptions.\n"
    "- Do not include an answer. Output only the question text prefixed by 'Q: '.\n\n"
    "LONG_ANALYSIS:\n"
    "{long_analysis}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate questions for LONG_ANALYSIS blocks.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/finetune1/finetune1_formatted_data_attempt2.txt"),
        help="Input dataset file containing LONG_ANALYSIS blocks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/finetune1/finetune1_long_analysis_QA.txt"),
        help="Where to write the Q/A blocks derived from LONG_ANALYSIS.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Model to use for the chat completion.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of LONG_ANALYSIS items to process (use this to test 2–3 at a time).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep between requests (seconds).",
    )
    return parser.parse_args()


def load_long_blocks(path: Path, split_token: str, limit: int) -> list[str]:
    text = path.read_text(encoding="utf-8")
    parts = [p.strip() for p in text.split(split_token) if p.strip()]
    longs: list[str] = []
    for part in parts:
        if "LONG_ANALYSIS:" not in part:
            continue
        _, after = part.split("LONG_ANALYSIS:", 1)
        long_text = after.strip()
        if not long_text or long_text.upper() == "NONE":
            continue
        longs.append(long_text)
        if len(longs) >= limit:
            break
    return longs


def generate_question(client: OpenAI, model: str, long_text: str) -> str:
    user_prompt = USER_PROMPT_TEMPLATE.format(long_analysis=long_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = (resp.choices[0].message.content or "").strip()
    # Normalize: strip leading "Q:" if present; we'll add it back consistently.
    if content.lower().startswith("q:"):
        content = content.split(":", 1)[1].strip()
    return content


def main() -> None:
    args = parse_args()
    client = OpenAI()

    long_blocks = load_long_blocks(args.input, SPLIT_TOKEN, args.limit)
    if not long_blocks:
        raise SystemExit("No LONG_ANALYSIS blocks found.")

    outputs: list[str] = []
    for idx, long_text in enumerate(long_blocks, 1):
        try:
            question = generate_question(client, args.model, long_text)
            print(f"Generated question #{idx}")
        except Exception as exc:  # noqa: BLE001
            print(f"Error on LONG_ANALYSIS #{idx}: {exc}")
            question = "[ERROR]"
        block = f"{SPLIT_TOKEN}\n\nQ: {question}\n\nA:\n{long_text}"
        outputs.append(block)
        if args.sleep > 0:
            time.sleep(args.sleep)

    args.output.write_text("\n".join(outputs) + "\n", encoding="utf-8")
    print(f"Wrote {len(outputs)} question blocks to {args.output}")


if __name__ == "__main__":
    main()
