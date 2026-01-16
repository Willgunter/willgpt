#!/usr/bin/env python3
"""
Extract 200 examples from finetune1_formatted_data_attempt1.txt (split by ---WILLGPTSTART---),
run a prompt against each (placeholder to be filled in), and write the combined output to
finetune1_formatted_data_attempt2.txt.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

from openai import OpenAI

SPLIT_TOKEN = "---WILLGPTSTART---"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send each split example to OpenAI Chat API and save responses."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("finetune1_formatted_data_attempt1.txt"),
        help="Input file containing raw examples split by the split token.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("finetune1_formatted_data_attempt2_analysis.txt"),
        help="Destination file for processed examples.",
    )
    parser.add_argument(
        "--split-token",
        default=SPLIT_TOKEN,
        help="Delimiter token separating examples in the input file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of examples to process.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI chat model to use.",
    )
    parser.add_argument(
        "--system",
        default="You are an expert extreme-conditions systems engineer. Your task is to extract engineering judgment, not to summarize text. Focus only on: assumptions, limits of validity, failure mechanisms, Ignore background explanations, derivations, and equations unless they directly explain a failure.",
        help="System prompt sent with each request.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.003,
        help="Optional sleep (seconds) between requests to avoid rate limits.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent requests to send.",
    )
    return parser.parse_args()


def load_examples(path: Path, split_token: str, limit: int) -> list[str]:
    text = path.read_text(encoding="utf-8")
    parts = [p.strip() for p in text.split(split_token) if p.strip()]
    return parts[:limit]


def apply_prompt(client: OpenAI, model: str, system_prompt: str, example: str) -> str:
    user_prompt = (
        "I will provide a technical engineering document or excerpt.\n\n"
        "Your task is to distill it into training supervision for a language model, following these rules exactly.\n\n"
        "PART 1: SHORT ENGINEERING Q&A (PRIMARY OUTPUT)\n\n"
        "Extract 2–4 short Q&A examples.\n\n"
        "Each Q&A must:\n\n"
        "reflect a realistic question one engineer would ask another\n\n"
        "focus on assumptions, operating limits, or failure modes\n\n"
        "require engineering judgment, not calculation\n\n"
        "have a short bullet-point answer (5–8 bullets max)\n\n"
        "contain NO equations\n\n"
        "contain NO derivations\n\n"
        "avoid textbook-style explanations\n\n"
        "Format exactly as:\n\n"
        "Q: <engineering question>\n\n"
        "A:\n\n"
        "<assumption or failure mode>\n"
        "<assumption or failure mode>\n"
        "<assumption or failure mode>\n\n"
        "Do NOT reuse phrasing from the document. Abstract and compress.\n\n"
        "PART 2: LONG FAILURE-ANALYSIS Q&A (OPTIONAL)\n\n"
        "If and only if the document contains a meaningful cause → effect → failure chain, output ONE long-form Q&A.\n\n"
        "Rules for the long-form Q&A:\n\n"
        "- The question should match the style and length of the short questions (one concise sentence).\n\n"
        "- Prefix the long Q&A with 'LONG_QA:' on its own line before the question.\n\n"
        "- The answer is a single paragraph explaining the causal chain.\n\n"
        "- The answer must: state original assumptions, explain what changes under extreme conditions, and describe the qualitative failure mechanism.\n\n"
        "- Do NOT include equations or derivations.\n\n"
        "Format exactly as:\n\n"
        "LONG_QA:\n"
        "Q: <engineering question>\n\n"
        "A:\n"
        "<one concise paragraph explaining the failure chain>\n\n"
        "If no such causal chain exists, output nothing for Part 2.\n\n"
        "HARD CONSTRAINTS (DO NOT VIOLATE)\n\n"
        "Do not include equations or symbols\n\n"
        "Do not include multiple-choice questions\n\n"
        "Do not include unrelated physics or chemistry\n\n"
        "Do not add new facts not implied by the text\n\n"
        "Do not exceed the requested number of examples\n\n"
        "DOCUMENT TO PROCESS\n"
        f"{example}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    args = parse_args()
    client = OpenAI()

    examples = load_examples(args.input, args.split_token, args.limit)
    if not examples:
        raise SystemExit("No examples found.")

    processed: list[str] = [""] * len(examples)

    def process_example(idx: int, ex: str) -> tuple[int, str]:
        try:
            processed_ex = apply_prompt(client, args.model, args.system, ex)
            if args.sleep > 0:
                time.sleep(args.sleep)
            print(f"Received response #{idx}")
            return idx, processed_ex
        except Exception as exc:  # noqa: BLE001
            print(f"Error on example {idx}: {exc}")
            return idx, ""

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(process_example, idx, ex) for idx, ex in enumerate(examples, 1)]
        for fut in as_completed(futures):
            idx, processed_ex = fut.result()
            processed[idx - 1] = processed_ex

    output_text = "\n".join(f"{args.split_token}\n\n{ex}" for ex in processed)
    args.output.write_text(output_text + "\n", encoding="utf-8")
    print(f"Wrote {len(processed)} examples to {args.output}")


if __name__ == "__main__":
    main()
