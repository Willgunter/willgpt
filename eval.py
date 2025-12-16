"""
Quick eval harness tuned to the models and benchmarks documented in README.md.
"""

import json
import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable

MODELS = [
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen3-4B-Base",
    "HuggingFaceTB/SmolLM3-3B-Base",
    "ServiceNow-AI/Apriel-5B-Base",
    "ibm-granite/granite-4.0-micro-base",
    "mistralai/Ministral-3-3B-Base-2512",
    "meta-llama/Llama-3.2-3B",
    "google/gemma-2-2b",
    "deepseek-ai/deepseek-llm-7b-base",
]

DESIRED_BENCHMARKS = [
    "arc_challenge",
    "gsm8k",
    "mathqa",
    "mmlu_electrical_engineering",
    "mmlu_college_computer_science",
    "mmlu_college_physics",
    "mmlu_conceptual_physics",
    "mmlu_college_mathematics"#,
    # "engidesign",  # not in lm_eval_harness
    # "engibench",  # not in lm_eval_harness
    # "svamp",  # not in lm_eval_harness
]
    
task_manager = TaskManager()
available_benchmarks = task_manager.match_tasks(DESIRED_BENCHMARKS)
missing_benchmarks = [
    benchmark for benchmark in DESIRED_BENCHMARKS if benchmark not in available_benchmarks
]

if missing_benchmarks:
    print(
        "Warning: the following benchmarks are listed in README.md but are not "
        "available in the current lm_eval installation:",
        missing_benchmarks,
    )

if not available_benchmarks:
    raise RuntimeError("No valid benchmarks could be resolved; please add tasks first.")

all_results = {}

for model_name in MODELS:
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name}",
        tasks=available_benchmarks,
        batch_size="auto",
    )
    all_results[model_name] = results["results"]

# Save results
with open("results.json", "w") as f:
    json.dump(all_results, f, default=handle_non_serializable, indent=2)
