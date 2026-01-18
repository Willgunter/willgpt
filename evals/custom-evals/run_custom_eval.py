#!/usr/bin/env python3
"""
Run a custom prompt eval against a Hugging Face model and save generations for manual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

HERE = Path(__file__).resolve().parent


def read_prompts(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    suffix = path.suffix.lower()
    raw_items: List[Any] = []

    if suffix == ".jsonl":
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    elif suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        raw_items = data
    else:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw_items.append(line)

    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_items, 1):
        if isinstance(item, str):
            record = {"id": f"q{idx}", "prompt": item}
        elif isinstance(item, dict) and "prompt" in item:
            record = dict(item)
            record.setdefault("id", f"q{idx}")
        else:
            raise ValueError(
                f"Prompt entry {idx} must be a string or an object with a 'prompt' field."
            )
        records.append(record)

    return records


def chunked(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def build_output_dir(path_str: str | None) -> Path:
    if path_str:
        return Path(path_str)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return HERE / "runs" / timestamp


# def slice_generation(output_ids: torch.Tensor, input_len: int, eos_token_id: int | None) -> torch.Tensor:
#     gen_ids = output_ids[input_len:]
#     if eos_token_id is None:
#         return gen_ids
#     eos_positions = (gen_ids == eos_token_id).nonzero(as_tuple=True)[0]
#     if eos_positions.numel():
#         return gen_ids[: eos_positions[0].item()]
#     return gen_ids
def slice_generation(output_ids, input_len, eos_token_id):
    return output_ids[input_len:]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a custom prompt eval for manual review.")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=HERE / "prompts.jsonl",
        help="Path to prompts (.jsonl, .json, or .txt).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6-Base",
        help="Hugging Face model ID to run.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter path. If set, loads base model then adapter.",
    )
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="If set with --adapter-path, merge the LoRA into the base before eval.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write results. Defaults to evals/custom-evals/runs/<timestamp>.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Torch dtype (e.g., float16, bfloat16, float32, auto).",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Transformers device map (e.g., "auto", "cpu", "cuda:0").',
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable if the model requires custom Hugging Face code.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = read_prompts(args.prompts)
    if not prompts:
        raise SystemExit("No prompts found. Add prompts and try again.")

    if args.seed is not None:
        set_seed(args.seed)

    output_dir = build_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = resolve_torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if args.adapter_path:
        peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model = peft_model.merge_and_unload() if args.merge_lora else peft_model
    else:
        model = base_model
    model.eval()

    results_path = output_dir / "results.jsonl"
    review_path = output_dir / "review.csv"
    run_config_path = output_dir / "run_config.json"
    repitition_penalty_param = 1.15
    no_repeat_ngram_size_param = 4
    

    run_config = {
        "model": args.model,
        "prompts": str(args.prompts),
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "repetition_penalty": repitition_penalty_param,
        "no_repeat_ngram_size": no_repeat_ngram_size_param,
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_count": len(prompts),
        "adapter_path": args.adapter_path,
        "merge_lora": args.merge_lora,
    }
    run_config_path.write_text(json.dumps(run_config, indent=2))

    do_sample = args.temperature > 0

    with results_path.open("w", encoding="utf-8") as results_file, review_path.open(
        "w", encoding="utf-8", newline=""
    ) as review_file:
        writer = csv.writer(review_file)
        writer.writerow(["id", "prompt", "response", "score", "notes"])

        completed = 0
        total = len(prompts)

        for batch in chunked(prompts, args.batch_size):
            prompt_texts = [record["prompt"] for record in batch]
            enc = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            )
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=do_sample,
                    repetition_penalty=repitition_penalty_param,
                    no_repeat_ngram_size=no_repeat_ngram_size_param,
                )

            for idx, record in enumerate(batch):
                input_len = int(attention_mask[idx].sum().item())
                gen_ids = slice_generation(outputs[idx], input_len, tokenizer.eos_token_id)
                response_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                result = {
                    "id": record["id"],
                    "prompt": record["prompt"],
                    "response": response_text,
                    "input_tokens": input_len,
                    "output_tokens": int(gen_ids.numel()),
                    "meta": record.get("meta"),
                }
                results_file.write(json.dumps(result, ensure_ascii=True) + "\n")
                writer.writerow([record["id"], record["prompt"], response_text, "", ""])
                print(f"response: {response_text}")
            completed += len(batch)
            print(f"Completed {completed} / {total} prompts")

    print(f"Wrote results to {results_path}")
    print(f"Wrote review sheet to {review_path}")
    print(f"Wrote run config to {run_config_path}")


if __name__ == "__main__":
    main()
