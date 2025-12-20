"""
Quick eval harness tuned to the models and benchmarks documented in README.md.
"""

import argparse
import json
import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable

MODELS = [
    "Qwen/Qwen2.5-3B", # HF native
    "Qwen/Qwen3-4B-Base", # HF native
    "HuggingFaceTB/SmolLM3-3B-Base", # HF native
    "ServiceNow-AI/Apriel-5B-Base", # NOT HF native - needed some bs to run properly
    "ibm-granite/granite-4.0-micro-base",
    "mistralai/Ministral-3-3B-Base-2512", # Ministral is very new so it needs trust_remote_code to be true
                                          # this eval (this file) doesn't properly propagate trust_remote_code down
                                          # so we get this error and do this fix:
                                          # ministral 3 key error - changed line 632 of /usr/local/lib/python3.12/dist-packages/lm_eval/models/huggingface.py
                                          # to be trust_remote_code=True,
    "meta-llama/Llama-3.2-3B",
    "google/gemma-2-2b",
    "deepseek-ai/deepseek-llm-7b-base",
]

DESIRED_BENCHMARKS = [
    "arc_challenge",
    "gsm8k", # for eval, set max_new_tokens to 256, 
    "mathqa",
    "mmlu_electrical_engineering",
    "mmlu_college_computer_science",
    "mmlu_college_physics",
    "mmlu_conceptual_physics",
    "mmlu_college_mathematics"#,
    # "engidesign",  # (OPEN subsection) not in lm_eval_harness
    # "engibench",  # not in lm_eval_harness
]

def _parse_csv_or_repeat(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    parsed: list[str] = []
    for value in values:
        parts = [p.strip() for p in value.split(",")]
        parsed.extend([p for p in parts if p])
    return parsed or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EleutherAI lm-eval-harness benchmarks for HF models."
    )
    parser.add_argument(
        "--models",
        action="append",
        help="Model(s) to eval. Repeat flag or pass comma-separated. Default: built-in list.",
    )
    parser.add_argument(
        "--benchmarks",
        action="append",
        help="Benchmark task(s). Repeat flag or pass comma-separated. Default: built-in list.",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Path to write results JSON (default: results.json).",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help=(
            'How many prompts to run in parallel per forward pass. Use a number (e.g. "4") '
            'to control GPU memory/speed, or "auto" to let lm-eval choose. '
            'If you hit CUDA OOM on a T4, try lowering this (e.g. "1" or "2").'
        ),
    )
    parser.add_argument(
        "--model-args",
        default="",
        help='Extra lm_eval model_args to append, e.g. "dtype=float16,device_map=auto,trust_remote_code=true".',
    )
    args = parser.parse_args()

    models = _parse_csv_or_repeat(args.models) or MODELS
    desired_benchmarks = _parse_csv_or_repeat(args.benchmarks) or DESIRED_BENCHMARKS

    task_manager = TaskManager()
    available_benchmarks = task_manager.match_tasks(desired_benchmarks)
    missing_benchmarks = [
        benchmark
        for benchmark in desired_benchmarks
        if benchmark not in available_benchmarks
    ]

    if missing_benchmarks:
        print(
            "Warning: the following benchmarks were requested but are not "
            "available in the current lm_eval installation:",
            missing_benchmarks,
        )

    if not available_benchmarks:
        raise RuntimeError("No valid benchmarks could be resolved; please add tasks first.")

    extra_model_args = (args.model_args or "").strip().lstrip(",")

    all_results = {}
    for model_name in models:
        model_args = f"pretrained={model_name}"
        if extra_model_args:
            model_args = f"{model_args},{extra_model_args}"
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=available_benchmarks,
            batch_size=args.batch_size,
        )
        all_results[model_name] = results["results"]

    with open(args.output, "w") as f:
        json.dump(all_results, f, default=handle_non_serializable, indent=2)


if __name__ == "__main__":
    main()
