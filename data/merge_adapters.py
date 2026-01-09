from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype="auto",
    device_map="auto"
)

lora_model = PeftModel.from_pretrained(
    base_model,
    "./unsloth-qwen-0.6-lora"
)

merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("./merged-qwen-0.6")
