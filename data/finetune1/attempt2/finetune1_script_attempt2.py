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
    "You are an extreme-conditions systems engineer."
    "Follow the user’s requested output format exactly. "
    "Never ask questions. Never generate quizzes or multiple-choice items."
    "Stop when the requested sections are complete."
)
QA_USER_PROMPT = (
    "Answer the engineering question in 3–8 short lines. "
    "Rules:"
    "- Do NOT ask follow-up questions."
    "- Do NOT generate multiple-choice options (no A), B), a), b), etc)."
    "- Do NOT include 'Answer:' or 'Q:' or 'A:' or similar in the output."
    "- No bullet symbols. Each line should be a plain sentence."
)

parser = argparse.ArgumentParser(description="Finetune Qwen with Unsloth.")
parser.add_argument(
    "--data-path",
    default="data/finetune1/attempt2/finetune1_formatted_data_attempt2.txt",
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
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Optional cap on number of parsed examples to use.",
)
args = parser.parse_args()
DATA_PATH = args.data_path
MODEL_NAME = args.model
DEBUG = args.debug
LIMIT = args.limit

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
    q_match = re.search(r"Q:\\s*(.+?)(?:\\n\\n|\\nA:|$)", chunk, re.S)
    a_match = re.search(r"A:\\s*(.+)", chunk, re.S)
    if q_match and a_match:
        question = q_match.group(1).strip()
        answer = a_match.group(1).strip()
        if question and answer:
            return {
                "instruction": question,
                "response": answer,
            }
    # fallback: treat the whole chunk as a response-only example
    return {
        "instruction": QA_USER_PROMPT,
        "response": chunk.strip(),
    }

parsed = [parse_chunk(c) for c in chunks]
if LIMIT is not None:
    parsed = parsed[:LIMIT]
    
    
dataset = Dataset.from_dict({
    "instruction": [p["instruction"] for p in parsed],
    "response": [p["response"] for p in parsed],
})

print(f"Loaded {len(parsed)} examples" + (f" (limit={LIMIT})" if LIMIT is not None else ""))
if DEBUG and parsed:
    print("Sample parsed entries (up to 2):")
    for row in parsed[:2]:
        print(f"instruction preview: {row['instruction'][:200]}")
        print(f"response preview: {row['response'][:200]}")
        print("---")

# -----------------------------
# Load model with Unsloth
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_LENGTH,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# Enable LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, #128,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16, # 256, # was 8
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
)
# model.print_trainable_parameters()
# -----------------------------
# Tokenization
# -----------------------------
def tokenize_pair(example):
    # Build an instruction-style prompt and only train on the response portion.
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


tokenized = dataset.map(tokenize_pair, remove_columns=["instruction", "response"])

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
    gradient_accumulation_steps=1,
    num_train_epochs=4,
    learning_rate=3e-4, # 1e-4
    logging_steps=50, # set back to 25
    save_steps=500,
    bf16=True,
    fp16=False,
    report_to="none",
)

# Note to self --> eventually learn how and why this works and what this does
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
# model.save_pretrained("./unsloth-qwen-lora")
# tokenizer.save_pretrained("./unsloth-qwen-lora")


model.save_pretrained("./qwen3-8B-extreme-reasoning-lora")
tokenizer.save_pretrained("./qwen3-8B-extreme-reasoning-lora")