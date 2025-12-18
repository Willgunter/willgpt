# willgpt

Lightweight playground for a college engineering student building and evaluating small-to-mid sized **base LLMs** as foundations for an engineering-focused model.

## Goals

- Start from strong open base models (2B–7B) and adapt them for engineering tasks.
- Track progress with a consistent evaluation suite (reasoning + engineering knowledge).
- Keep setup simple and reproducible.

## Setup

```bash
pip install -U transformers
```

## Benchmarks

Models are evaluated on the following benchmarks:

- engiBench / engDesign (OPEN subsection) — run via a custom vLLM harness using instruct-tuned variants (or closest available siblings) of the base models listed below.
- MMLU (engineering related subsections), specifically:
  - `mmlu_electrical_engineering`
  - `mmlu_college_computer_science`
  - `mmlu_college_physics`
  - `mmlu_conceptual_physics`
  - `mmlu_college_mathematics`
- GSM8K
- SVAMP
- ARC-Challenge
- MathQA

## Base Models

These are the base models used as starting points (all links go to the model cards on Hugging Face):

- [Qwen/Qwen 2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)
- [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base)
- [HuggingFaceTB/SmolLM3-3B-Base](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base)
- [ServiceNow-AI/Apriel-5B-Base](https://huggingface.co/ServiceNow-AI/Apriel-5B-Base)
- [ibm-granite/granite-4.0-micro-base](https://huggingface.co/ibm-granite/granite-4.0-micro-base)
- [mistralai/Ministral-3-3B-Base-2512](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512)
- [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b)
- [deepseek-ai/deepseek-llm-7b-base](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base)

For benchmarking (especially EngiDesign OPEN), the closest available instruct / sibling variants are:

- Qwen/Qwen2.5-3B Base → `Qwen/Qwen2.5-3B-Instruct`
- Qwen/Qwen3-4B-Base → `Qwen/Qwen3-4B-Instruct`
- HuggingFaceTB/SmolLM3-3B-Base → `HuggingFaceTB/SmolLM3-3B-Instruct`
- ServiceNow-AI/Apriel-5B-Base → `ServiceNow-AI/Apriel-5B-Instruct`
- ibm-granite/granite-4.0-micro-base → `ibm-granite/granite-4.0-micro-instruct`
- mistralai/Ministral-3-3B-Base-2512 → `mistralai/Ministral-3-3B-Instruct-2512`
- meta-llama/Llama-3.2-3B → `meta-llama/Llama-3.2-3B-Instruct`
- google/gemma-2-2b → `google/gemma-2-2b-it`
- deepseek-ai/deepseek-llm-7b-base → `deepseek-ai/deepseek-llm-7b-chat`

## EngiDesign OPEN Eval

The EngiDesign-Open tasks live under `evals/EngDesign-Open/`, and there is a vLLM-based harness to batch-evaluate models against the native per-task evaluators.

### 1. Generate the task index

From the repo root:

```bash
python evals/build_engidesign_dataset.py
```

This writes `evals/engidesign_open_dataset.jsonl`, a JSONL metadata file describing each task (prompt, output schema, and evaluation script).

### 2. Install dependencies (Colab / HPC)

At minimum you need:

```bash
pip install vllm transformers
```

Some EngiDesign tasks also require extra libraries (e.g., `numpy`, `scipy`, `opencv-python`); install them as needed in the environment where you run the eval.

### 3. Run a small smoke test

From the repo root, to evaluate a couple of models on the first 5 EngiDesign tasks:

```bash
python evals/engidesign_open_eval.py \
  --dataset evals/engidesign_open_dataset.jsonl \
  --models Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct \
  --max-samples 5 \
  --output-dir evals/eng-design/engidesign-open-smoke
```

### 4. Run the full benchmark

Drop `--max-samples` (or increase it) and point `--output-dir` wherever you want summaries written (e.g., a Google Drive folder in Colab or a scratch directory on HPC):

```bash
python evals/engidesign_open_eval.py \
  --dataset evals/engidesign_open_dataset.jsonl \
  --models Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct \
  --output-dir evals/eng-design/engidesign-open-results
```

The harness loads one model at a time via vLLM, generates JSON responses for each task prompt (constrained by the task’s Pydantic schema), calls the task’s `evaluate_llm_response` function, and saves a per-model JSON summary with pass rate, average score, and per-task transcripts.
