import argparse
import re

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments

# -----------------------------
# Config
# -----------------------------
SPLIT_TOKEN = "---WILLGPTSTART---"
MAX_LENGTH = 16384
SYSTEM_PROMPT = (
    "You are an expert extreme-conditions systems engineer who performs complete, "
    "first-principles failure analyses. Follow the numbered checklist: identify explicit "
    "and hidden assumptions, map to governing physics, interdependencies, boundary "
    "conditions, temporal evolution, safety margins, and robustness. Return a structured, "
    "exhaustive analysis in prose with clear headers."
)
QA_USER_PROMPT = (
    "Answer the engineering question with 3–8 concise bullet points focused only on assumptions, "
    "operating limits, or failure modes. No equations, no extra exposition."
)
LONG_USER_PROMPT = (
    "Provide one concise LONG_ANALYSIS paragraph describing the cause → effect → failure chain. "
    "State assumptions, what changes under stress, and the qualitative mechanism."
)

parser = argparse.ArgumentParser(description="Finetune Qwen with Unsloth.")
parser.add_argument(
    "--data-path",
    default="formatted_data.txt",
    help="Path to the raw training text file.",
)
parser.add_argument(
    "--model",
    default="Qwen/Qwen3-0.6B",
    help="Base or merged model ID/path to load with Unsloth.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Print diagnostics about splitting/tokenization.",
)
args = parser.parse_args()
DATA_PATH = args.data_path
MODEL_NAME = args.model
DEBUG = args.debug

# -----------------------------
# Load + split raw text
# -----------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

if DEBUG:
    print(f"Raw text length (chars): {len(raw_text)}")
    print(f"Split token: {SPLIT_TOKEN!r}")
    print(f"Split token occurrences: {raw_text.count(SPLIT_TOKEN)}")

chunks = [
    c.strip()
    for c in re.split(SPLIT_TOKEN, raw_text)
    if c.strip()
]

def parse_chunk(chunk: str):
    # LONG_ANALYSIS block
    if "LONG_ANALYSIS:" in chunk:
        long_text = chunk.split("LONG_ANALYSIS:", 1)[1].strip()
        if long_text:
            return {
                "kind": "long",
                "instruction": LONG_USER_PROMPT,
                "response": long_text,
            }
    # Q/A block
    q_match = re.search(r"Q:\\s*(.+?)(?:\\n\\n|\\nA:|$)", chunk, re.S)
    a_match = re.search(r"A:\\s*(.+)", chunk, re.S)
    if q_match and a_match:
        question = q_match.group(1).strip()
        answer = a_match.group(1).strip()
        if question and answer:
            return {
                "kind": "qa",
                "instruction": question,
                "response": answer,
            }
    # fallback: treat the whole chunk as a response-only example
    return {
        "kind": "qa",
        "instruction": QA_USER_PROMPT,
        "response": chunk.strip(),
    }

parsed = [parse_chunk(c) for c in chunks]
dataset = Dataset.from_dict({
    "kind": [p["kind"] for p in parsed],
    "instruction": [p["instruction"] for p in parsed],
    "response": [p["response"] for p in parsed],
})

print(f"Loaded {len(parsed)} examples")
if DEBUG and parsed:
    print("Sample parsed entries (up to 2):")
    for row in parsed[:2]:
        print(f"kind={row['kind']}")
        print(f"instruction preview: {row['instruction'][:200]}")
        print(f"response preview: {row['response'][:200]}")
        print("---")

# -----------------------------
# Load model with Unsloth
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_LENGTH,
    dtype = None,
    load_in_4bit = True,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Enable LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
)

# -----------------------------
# Tokenization
# -----------------------------
def tokenize_pair(example):
    # Build an instruction-style prompt and only train on the response portion.
    if example["kind"] == "long":
        user_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{LONG_USER_PROMPT}\n\n"
            "LONG_ANALYSIS:\n"
        )
    else:
        user_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{QA_USER_PROMPT}\n\n"
            f"Q: {example['instruction']}\n\n"
            "A:\n"
        )

    prompt_ids = tokenizer(user_prompt, add_special_tokens=False)["input_ids"]
    answer_text = example["response"]
    answer_ids = tokenizer(answer_text + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    # Truncate answer if needed to respect the context window.
    max_answer_len = MAX_LENGTH - len(prompt_ids)
    if max_answer_len <= 0:
        raise ValueError("Prompt is longer than MAX_LENGTH. Reduce prompt text or increase MAX_LENGTH.")
    if len(answer_ids) > max_answer_len:
        answer_ids = answer_ids[:max_answer_len]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


tokenized = dataset.map(tokenize_pair, remove_columns=["kind", "instruction", "response"])

# -----------------------------
# Debug token lengths
# -----------------------------
if DEBUG:
    lengths = [len(x) for x in tokenized["input_ids"]]
    if lengths:
        lengths_sorted = sorted(lengths)
        min_len = lengths_sorted[0]
        max_len = lengths_sorted[-1]
        mid_len = lengths_sorted[len(lengths_sorted) // 2]
        print(f"Tokenized lengths: min={min_len}, median={mid_len}, max={max_len}")
        print(f"First 10 lengths: {lengths_sorted[:10]}")
        print(f"Last 10 lengths: {lengths_sorted[-10:]}")
    else:
        print("Tokenized lengths: no samples")

# -----------------------------
# Training config
# -----------------------------
training_args = TrainingArguments(
    output_dir="./unsloth-qwen",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-4,
    logging_steps=10, # set back to 25
    save_steps=500,
    fp16=True,
    report_to="none",
)

def data_collator(features):
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids = []
    attention_masks = []
    labels = []
    pad_id = tokenizer.pad_token_id
    label_pad = -100
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_id] * pad_len)
        attention_masks.append(f["attention_mask"] + [0] * pad_len)
        labels.append(f["labels"] + [label_pad] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

# if DEBUG:
    # batch = next(iter(trainer.get_train_dataloader()))
    # labels = batch["labels"]
    # input_ids = batch["input_ids"]
    # pad_id = tokenizer.pad_token_id
    # num_pad_labels = (labels == pad_id).sum().item()
    # num_ignore_labels = (labels == -100).sum().item()
    # total_labels = labels.numel()
    # print("Pad labels:", num_pad_labels)
    # print("Ignore labels:", num_ignore_labels)
    # print("Total labels:", total_labels)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Save LoRA adapter
# -----------------------------
model.save_pretrained("./unsloth-qwen-lora")
tokenizer.save_pretrained("./unsloth-qwen-lora")
