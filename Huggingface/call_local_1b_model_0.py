"""
Load a small (~1.5B) model on this machine and answer a query.
First run downloads the weights from Hugging Face into the local cache.
Later runs reuse that cache. This does not call a cloud inference API.
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Small instruct model (closest ungated Qwen to "1B").
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
QUERY = "What is Hugging Face? Answer in one sentence."


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    token = os.environ.get("HF_TOKEN")
    device = pick_device()
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

    print(f"Loading local model: {MODEL_ID}")
    print(f"Device: {device}  dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    messages = [{"role": "user", "content": QUERY}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    print(f"\nQuery: {QUERY}\n")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("Answer:")
    print(answer.strip())


if __name__ == "__main__":
    main()
