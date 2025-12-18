#!/usr/bin/env python3
"""
vLLM-based evaluation harness for the EngiDesign OPEN benchmark.

The benchmark data lives under evals/EngDesign-Open/<task_id>/ and is summarized
in evals/engidesign_open_dataset.jsonl (generated via build_engidesign_dataset.py).
Each task exposes:
  - an LLM prompt (free-form instructions)
  - a Pydantic Response_structure describing the required JSON output format
  - an evaluate.py module with evaluate_llm_response(response) -> Tuple

For each model we:
  1. Load and cache all task prompts + JSON schemas.
  2. Feed the prompts through vLLM (batch generation).
  3. Extract the JSON payload from each completion.
  4. Run the provided evaluation script in an isolated subprocess to score the response.
  5. Save per-sample transcripts + aggregate metrics for later analysis.

This harness is designed to run comfortably inside Google Colab (T4) by keeping
one model loaded at a time and aggressively cleaning up GPU memory between runs.

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from vllm import LLM, SamplingParams


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
print(REPO_ROOT)
DATASET_PATH = REPO_ROOT / "evals" / "engidesign_open_dataset.jsonl"
print(DATASET_PATH)
EVAL_RUNNER = REPO_ROOT / "evals" / "run_engidesign_eval.py"

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "HuggingFaceTB/SmolLM3-3B-Instruct",
    "ServiceNow-AI/Apriel-5B-Base",
    "ibm-granite/granite-4.0-micro-base",
    "mistralai/Ministral-3-3B-Base-2512",
    "google/gemma-2-2b",
    "deepseek-ai/deepseek-llm-7b-base",
]


def read_json_schema(response_cls) -> str:
    if hasattr(response_cls, "model_json_schema"):
        schema = response_cls.model_json_schema()
    elif hasattr(response_cls, "schema"):
        schema = response_cls.schema()
    else:
        schema = {}
    return json.dumps(schema, indent=2)


def load_response_class(module_path: Path, class_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(f"{module_path} is missing {class_name}") from exc


@dataclass
class EngiDesignTask:
    task_id: str
    prompt: str
    domain_topic: Optional[str]
    task_provider: Optional[str]
    evaluation_pipeline: Optional[str]
    output_structure_path: Path
    output_structure_class: str
    evaluator_path: Path
    evaluator_fn: str
    schema_text: str = field(init=False)

    def __post_init__(self):
        print(self.output_structure_path)
        print(self.output_structure_class) # Response_structure
        response_cls = load_response_class(self.output_structure_path, self.output_structure_class)
        self.schema_text = read_json_schema(response_cls)

    @property
    def task_dir(self) -> Path:
        return self.output_structure_path.parent


def load_tasks(dataset_path: Path, limit: Optional[int]) -> List[EngiDesignTask]:
    tasks: List[EngiDesignTask] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            task = EngiDesignTask(
                task_id=record["id"],
                prompt=record["prompt"],
                domain_topic=record.get("domain_topic"),
                task_provider=record.get("task_provider"),
                evaluation_pipeline=record.get("evaluation_pipeline"),
                output_structure_path=REPO_ROOT / record["output_structure"]["path"],
                output_structure_class=record["output_structure"].get("class", "Response_structure"),
                evaluator_path=REPO_ROOT / record["evaluator"]["path"],
                evaluator_fn=record["evaluator"].get("function", "evaluate_llm_response"),
            )
            tasks.append(task)
            if limit is not None and len(tasks) >= limit:
                break
    if not tasks:
        raise RuntimeError(f"No tasks loaded from {dataset_path}")
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark EngiDesign OPEN tasks using vLLM + task-native evaluators."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to engidesign_open_dataset.jsonl (see build_engidesign_dataset.py).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Hugging Face model identifiers to benchmark.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of EngiDesign tasks to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/engidesign_open"),
        help="Directory where per-model JSON results will be stored.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens generated per task.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to vLLM.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling (only used when temperature > 0).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="half",
        help="Model dtype for vLLM (e.g., half on T4, bfloat16 on A100).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel degree for vLLM.",
    )
    parser.add_argument(
        "--swap-space",
        type=int,
        default=4,
        help="Swap space (GB) allocated by vLLM when VRAM is limited.",
    )
    return parser.parse_args()


def build_prompt(task: EngiDesignTask) -> str:
    context_lines = []
    if task.domain_topic:
        context_lines.append(f"Domain: {task.domain_topic}")
    if task.task_provider:
        context_lines.append(f"Task Provider: {task.task_provider}")
    instructions = (
        "You must respond with a single JSON object that satisfies the schema "
        "below. Do not include markdown fences or extra commentary.\n"
        f"{task.schema_text}\n"
        "Return strictly valid JSON."
    )
    prefix = "\n".join(context_lines)
    return "\n\n".join(filter(None, [prefix, task.prompt, instructions]))


def build_sampling_params(max_tokens: int, temperature: float, top_p: float) -> SamplingParams:
    params = {"max_tokens": max_tokens}
    if temperature and temperature > 0:
        params.update({"temperature": temperature, "top_p": top_p})
    else:
        params["temperature"] = 0.0
        params["top_p"] = top_p
    return SamplingParams(**params)


def extract_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def evaluate_response(task: EngiDesignTask, json_payload: str) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(EVAL_RUNNER),
        "--output-structure",
        str(task.output_structure_path),
        "--response-class",
        task.output_structure_class,
        "--evaluator",
        str(task.evaluator_path),
        "--function",
        task.evaluator_fn,
    ]
    proc = subprocess.run(
        cmd,
        input=json_payload.encode("utf-8"),
        capture_output=True,
        text=True,
        cwd=task.task_dir,
    )
    if proc.returncode != 0:
        return {
            "passed": False,
            "details": proc.stderr.strip() or proc.stdout.strip(),
            "score": None,
            "max_score": None,
            "error": "evaluation_failed",
        }
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "passed": False,
            "details": f"Failed to parse evaluation output: {exc}",
            "score": None,
            "max_score": None,
            "error": "evaluation_parse_error",
        }


def slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


def cleanup_model(llm: Optional[LLM]) -> None:
    if llm is not None:
        del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(args.dataset, args.max_samples)
    prompts = [build_prompt(task) for task in tasks]
    sampling_params = build_sampling_params(args.max_new_tokens, args.temperature, args.top_p)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        print(f"\n=== Starting {model_name} on {len(tasks)} EngiDesign tasks ===")
        llm = None
        try:
            llm = LLM(
                model=model_name,
                dtype=args.dtype,
                tensor_parallel_size=args.tensor_parallel_size,
                swap_space=args.swap_space,
                enforce_eager=True,
            )
            outputs = llm.generate(prompts, sampling_params)

            per_sample = []
            n_passed = 0
            score_sum = 0.0
            score_count = 0

            for task, result in zip(tasks, outputs):
                raw_output = result.outputs[0].text if result.outputs else ""
                json_block = extract_json(raw_output)
                parsed_obj = None
                eval_result = {
                    "passed": False,
                    "details": "No JSON payload detected.",
                    "score": None,
                    "max_score": None,
                    "error": "json_extraction_failed",
                }

                if json_block:
                    try:
                        parsed_obj = json.loads(json_block)
                        eval_result = evaluate_response(task, json_block)
                    except json.JSONDecodeError as exc:
                        eval_result = {
                            "passed": False,
                            "details": f"Invalid JSON: {exc}",
                            "score": None,
                            "max_score": None,
                            "error": "json_parse_error",
                        }

                if eval_result.get("passed"):
                    n_passed += 1
                if eval_result.get("score") is not None:
                    score_sum += float(eval_result["score"])
                    score_count += 1

                per_sample.append(
                    {
                        "task_id": task.task_id,
                        "raw_output": raw_output,
                        "json_payload": json_block,
                        "parsed": parsed_obj,
                        "evaluation": eval_result,
                    }
                )

            avg_score = (score_sum / score_count) if score_count else None
            summary = {
                "model": model_name,
                "n_tasks": len(tasks),
                "n_passed": n_passed,
                "pass_rate": n_passed / len(tasks),
                "avg_score": avg_score,
                "samples": per_sample,
                "generation": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
            }

            output_path = args.output_dir / f"{slugify_model_name(model_name)}.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            print(f"Saved results to {output_path}")
        finally:
            cleanup_model(llm)


if __name__ == "__main__":
    main()
