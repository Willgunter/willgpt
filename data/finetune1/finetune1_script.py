import argparse
import re
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B"
SPLIT_TOKEN = "---WILLGPTSTART---"
MAX_LENGTH = 16384

parser = argparse.ArgumentParser(description="Finetune Qwen with Unsloth.")
parser.add_argument(
    "--data-path",
    default="formatted_data.txt",
    help="Path to the raw training text file.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Print diagnostics about splitting/tokenization.",
)
args = parser.parse_args()
DATA_PATH = args.data_path
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

dataset = Dataset.from_dict({"text": chunks})
print(f"Loaded {len(chunks)} examples")
if DEBUG and chunks:
    preview = chunks[0][:500]
    print("First chunk preview (first 500 chars):")
    print(preview)
    if len(chunks) > 1:
        print("Second chunk preview (first 200 chars):")
        print(chunks[1][:200])

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
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    return tokens

tokenized = dataset.map(tokenize, remove_columns=["text"])

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
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=10, # set back to 25
    save_steps=500,
    fp16=True,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
