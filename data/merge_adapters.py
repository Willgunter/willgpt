import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Merge LoRA adapter into a base model.")
parser.add_argument(
    "--model",
    default="Qwen/Qwen3-0.6B",
    help="Base model ID/path to load.",
)
args = parser.parse_args()

base_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype="auto",
    device_map="auto",
)
lora_model = PeftModel.from_pretrained(
    base_model,
    "./unsloth-qwen-lora"
)

merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("./merged-qwen")
