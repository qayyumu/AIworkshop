"""
Generate with an explicit KV cache (same local Qwen 1.5B as call_local_1b_model_0.py).

Without a cache, every new token would re-run attention over the whole sequence.
With a cache:
  1. Prefill: run the full prompt once, store K and V for every layer.
  2. Decode: feed only the newest token; append its K/V to the cache.

This script does those two stages by hand (not model.generate).
"""

import os
from pathlib import Path

# This env has a broken TensorFlow install; keep Transformers on PyTorch only.
os.environ["USE_TF"] = "0"

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
QUERY = "What is Hugging Face? Answer in one sentence."
MAX_NEW_TOKENS = 40


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cache_length(past_key_values):
    """How many tokens are already stored in the KV cache."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    # Older Transformers: tuple of (key, value) per layer, seq dim = 2.
    return past_key_values[0][0].shape[2]


def cache_memory_mb(past_key_values):
    if past_key_values is None:
        return 0.0
    if hasattr(past_key_values, "key_cache"):
        tensors = list(past_key_values.key_cache) + list(past_key_values.value_cache)
    else:
        tensors = [tensor for kv in past_key_values for tensor in kv]
    nbytes = sum(t.numel() * t.element_size() for t in tensors if t is not None)
    return nbytes / (1024 * 1024)


def next_token_from_logits(logits):
    """Greedy: pick the highest-probability token (keeps the cache demo easy to follow)."""
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


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
    input_ids = inputs["input_ids"].to(device)

    print(f"\nQuery: {QUERY}")
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print("\n--- prefill (full prompt, build KV cache) ---")

    generated = []
    past = DynamicCache()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, past_key_values=past)
        past = out.past_key_values
        token_id = next_token_from_logits(out.logits)
        generated.append(token_id.item())

    print(
        f"  input shape={tuple(input_ids.shape)}  "
        f"cache_len={cache_length(past)}  "
        f"cache≈{cache_memory_mb(past):.1f} MB  "
        f"first token={tokenizer.convert_ids_to_tokens(generated[-1])!r}"
    )

    print("\n--- decode (one token in, cache grows by 1) ---")
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for step in range(1, MAX_NEW_TOKENS):
            if generated[-1] == eos_id:
                break
            out = model(
                input_ids=token_id,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            token_id = next_token_from_logits(out.logits)
            generated.append(token_id.item())
            print(
                f"  step {step:02d}  input shape={tuple(token_id.shape)}  "
                f"cache_len={cache_length(past)}  "
                f"token={tokenizer.convert_ids_to_tokens(generated[-1])!r}"
            )

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(generated)
    )
    print("\nAnswer:")
    print(answer.strip())


if __name__ == "__main__":
    main()
