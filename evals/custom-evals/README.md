# Custom manual evals

This folder holds a simple, manual-review eval runner for your own prompt set.
It runs prompts against `Qwen/Qwen3-4B-Base` by default and writes a results file
plus a review sheet you can score by hand.

## Dependencies

```bash
pip install -U torch transformers
```

## 1) Add your prompts

Edit `evals/custom-evals/prompts.jsonl` and replace the sample lines with your
own ~20 prompts.

Each line is JSON with a `prompt` field and optional `id`/`meta`:

```jsonl
{"id": "q1", "prompt": "Your question here"}
{"id": "q2", "prompt": "Another question", "meta": {"topic": "circuits"}}
```

You can also pass a `.json` list or a plain `.txt` file (one prompt per line).

## 2) Run the eval

From the repo root:

```bash
python evals/custom-evals/run_custom_eval.py \
  --prompts evals/custom-evals/prompts.jsonl \
  --model Qwen/Qwen3-4B-Base \
  --max-new-tokens 256 \
  --batch-size 1
```

Notes:
- First run will download the model from Hugging Face. If it is not cached, make
  sure you have network access and any required HF auth set up.
- If the model needs custom code, add `--trust-remote-code`.
- For CPU-only runs, add `--device-map cpu --dtype float32` (slow but works).

## 3) Review outputs

Each run writes to `evals/custom-evals/runs/<timestamp>/`:

- `results.jsonl`: prompt + model response for each item
- `review.csv`: same responses with empty `score` and `notes` columns
- `run_config.json`: run settings snapshot

Open `review.csv` and fill in `score`/`notes` manually.
