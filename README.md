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

- engiBench / engDesign (OPEN subsection)
- MMMU (currently skipped in benchmarking)
- MMLU (engineering subsection)
- GSM8K
- SVAMP
- ARC-Challenge
- MathQA

## Base Models

These are the base models used as starting points (all links go to the model cards on Hugging Face):

- [Qwen 2.5 3B Base](https://huggingface.co/Qwen/Qwen2.5-3B)
- [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base)
- [SmolLM3-3B-Base](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base)
- [Apriel-5B-Base](https://huggingface.co/ServiceNow-AI/Apriel-5B-Base)
- [granite-4.0-micro-base](https://huggingface.co/ibm-granite/granite-4.0-micro-base)
- [Ministral-3-3B-Base-2512](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512)
- [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- [gemma-2-2b](https://huggingface.co/google/gemma-2-2b)
- [deepseek-llm-7b-base](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base)
