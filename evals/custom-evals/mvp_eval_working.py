#!/usr/bin/env python3
    
# Original code has been removed temporarily per request.
# ---- old harness placeholder ----
# (previous evaluation logic commented out)

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def main() -> None:
    set_seed(42)
    model_id = "Qwen/Qwen3-0.6B-Base"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
    model.config.pad_token_id = tokenizer.eos_token_id

    prompts = [
        "Explain thermal runaway in lithium-ion batteries.",
        "What is a failure mode in mechanical systems?",
        "Give a short paragraph on stress–strain curves.",
        "Why do materials become brittle at cryogenic temperatures?",
        "What causes voltage sags to trip motors?",
    ]

    for idx, prompt in enumerate(prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = output[0][input_len:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


        print(f"\n[{idx}] Prompt: {prompt}\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()