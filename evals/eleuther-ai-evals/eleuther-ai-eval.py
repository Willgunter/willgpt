"""
Quick eval harness tuned to the models and benchmarks documented in README.md.
"""
from typing import TypedDict
import transformers.utils

if not hasattr(transformers.utils, "LossKwargs"):
    class LossKwargs(TypedDict, total=False):
        pass

    transformers.utils.LossKwargs = LossKwargs

import argparse
import json
import lm_eval
from lm_eval.tasks import TaskManager
try:
    from lm_eval.utils import handle_non_serializable
except ImportError:
    def handle_non_serializable(x):
        return x




MODELS = [
    "Qwen/Qwen2.5-3B", # HF native (proxy: Qwen/Qwen2.5-0.5B)
    "Qwen/Qwen3-4B-Base", # HF native (proxy: Qwen/Qwen2.5-0.5B)
    "HuggingFaceTB/SmolLM3-3B-Base", # HF native (proxy: HuggingFaceTB/SmolLM3-135M)
    "ServiceNow-AI/Apriel-5B-Base", # NOT HF native - needed some bs to run properly (proxy: ServiceNow-AI/Apriel-1.5B-Base)
        # ImportError: cannot import name 'LossKwargs' from 'transformers.utils'
        # vim into /usr/local/lib/python3.12/dist-packages/lm_eval/models/huggingface.py and change this code on line 630 ish,
        # then run lm_eval
        # OLD
        # self._model = self.AUTO_MODEL_CLASS.from_pretrained(
        #     pretrained,
        #     revision=revision,
        #     dtype=get_dtype(dtype),
        #     trust_remote_code=trust_remote_code,
        #     gguf_file=gguf_file,
        #     quantization_config=quantization_config,
        #     subfolder=subfolder,
        #     **model_kwargs,
        # )
        # 
        # NEW
        #  fp_dtype = get_dtype(dtype)
        #     if trust_remote_code or dtype in (None, "auto"):
        #         fp_dtype = None
        #     if fp_dtype is not None:
        #         model_kwargs = dict(model_kwargs)
        #         model_kwargs["dtype"] = fp_dtype
        #     self._model = self.AUTO_MODEL_CLASS.from_pretrained(
        #         pretrained,
        #         revision=revision,
        #         trust_remote_code=trust_remote_code,
        #         gguf_file=gguf_file,
        #         quantization_config=quantization_config,
        #         subfolder=subfolder,
        #         **model_kwargs,
        #     )
        
        
    "ibm-granite/granite-4.0-micro-base", # (proxy: itself (3B params))
    "meta-llama/Llama-3.2-3B", # need to be authenticated (proxy: meta-llama/Llama-3.2-1B)
    "google/gemma-2-2b", # need to be authenticated (proxy: itself)
    "deepseek-ai/deepseek-llm-7b-base", # proxy: deepseek-ai/deepseek-llm-1.3b-base
]

DESIRED_BENCHMARKS = [
    "arc_challenge",
    "mathqa",
    "mmlu_electrical_engineering",
    "mmlu_college_computer_science",
    "mmlu_college_physics",
    "mmlu_conceptual_physics",
    "mmlu_college_mathematics"#,
    "gsm8k", # (save for later) for eval, set max_new_tokens to 256, 
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Forward-generation budget (e.g. gsm8k likes 256).",
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

    all_results = {}
    for model_name in models:
        model_args = f"pretrained={model_name}"
        if args.model_args:
            model_args = f"{model_args},{args.model_args}"
        generate_kwargs = {}
        if args.max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = args.max_new_tokens
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=available_benchmarks,
            batch_size=args.batch_size,
            generate_kwargs=generate_kwargs or None,
        )
        all_results[model_name] = results["results"]

    with open(args.output, "w") as f:
        json.dump(all_results, f, default=handle_non_serializable, indent=2)


if __name__ == "__main__":
    main()
