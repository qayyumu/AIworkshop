"""
List files in a Hugging Face model repo and explain the LLM architecture.

Default: https://huggingface.co/Qwen/Qwen3.8-27B/tree/main

Only small JSON files are downloaded (config + weight index).
The multi-GB safetensors shards are not downloaded.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO_ID = "Qwen/Qwen3.8-27B"

# Short explanation for each repo file.
FILE_ROLES = {
    ".gitattributes": "Git LFS rules for large files (weights, tokenizer.json).",
    "LICENSE": "License for the model weights.",
    "README.md": "Model card: how to use the model.",
    "chat_template.jinja": "Turns chat messages into the token sequence the model expects.",
    "config.json": "Architecture: layer sizes, heads, vocab, vision settings.",
    "crc32.txt": "Checksums from the publisher.",
    "generation_config.json": "Default sampling settings (temperature, top_p, EOS ids).",
    "merges.txt": "BPE merge rules for the tokenizer.",
    "vocab.json": "Token string -> id map.",
    "tokenizer.json": "Full fast tokenizer (vocab + merges + special tokens).",
    "tokenizer_config.json": "Tokenizer settings: special tokens and max length.",
    "preprocessor_config.json": "How images are patched and normalized.",
    "video_preprocessor_config.json": "How video frames are patched and normalized.",
    "model.safetensors.index.json": "Map of tensor name -> which shard file holds it.",
}


def file_size(num_bytes):
    """Turn a byte count into something readable, e.g. 3.69 GB."""
    if not num_bytes:
        return "0 B"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return str(num_bytes)


def load_json(repo_id, filename, token):
    """Download one small JSON file from the Hub and parse it."""
    path = hf_hub_download(repo_id, filename, token=token)
    with open(path) as handle:
        return json.load(handle)


def print_architecture(config):
    """
    config.json is nested for this model:

      config.json
        text_config    <- the LLM (language model)
        vision_config  <- the image/video encoder
    """
    text = config.get("text_config") or config
    vision = config.get("vision_config") or {}
    layer_types = text.get("layer_types") or []
    n_linear = layer_types.count("linear_attention")
    n_full = layer_types.count("full_attention")

    print("\n=== LLM architecture (from config.json -> text_config) ===")
    print(f"  class                    {config.get('architectures')}")
    print(f"  model_type               {config.get('model_type')} / {text.get('model_type')}")
    print(f"  hidden_size              {text.get('hidden_size')}     (embedding / residual width)")
    print(f"  intermediate_size        {text.get('intermediate_size')}    (MLP / FFN width)")
    print(f"  num_hidden_layers        {text.get('num_hidden_layers')}")
    print(f"  layer mix                {n_linear} linear_attention + {n_full} full_attention")
    print(f"  full_attention_interval  {text.get('full_attention_interval')}     (every Nth layer is full attention)")
    print(f"  num_attention_heads      {text.get('num_attention_heads')}      (full-attention Q heads)")
    print(f"  num_key_value_heads      {text.get('num_key_value_heads')}       (GQA: fewer K/V heads)")
    print(f"  head_dim                 {text.get('head_dim')}")
    print(f"  linear_num_key_heads     {text.get('linear_num_key_heads')}")
    print(f"  linear_num_value_heads   {text.get('linear_num_value_heads')}")
    print(f"  vocab_size               {text.get('vocab_size')}")
    print(f"  max_position_embeddings  {text.get('max_position_embeddings')}  (context length)")
    print(f"  hidden_act               {text.get('hidden_act')}")
    print(f"  rms_norm_eps             {text.get('rms_norm_eps')}")
    print(f"  dtype                    {text.get('dtype')}")
    print(f"  mtp_num_hidden_layers    {text.get('mtp_num_hidden_layers')}       (extra next-token prediction head)")
    print(f"  tie_word_embeddings      {text.get('tie_word_embeddings')}")

    rope = text.get("rope_parameters") or {}
    if rope:
        print("\n  RoPE (position encodings):")
        print(f"    rope_theta             {rope.get('rope_theta')}")
        print(f"    rope_type              {rope.get('rope_type')}")
        print(f"    partial_rotary_factor  {rope.get('partial_rotary_factor')}")

    print("\n  One language layer is either:")
    print("    linear_attention  -> Gated DeltaNet / SSM-style (fast, long context)")
    print("    full_attention    -> standard multi-head attention (every 4th layer)")
    print("  Then an MLP (SiLU gated FFN) and RMSNorm. Residual stream width = hidden_size.")

    if vision:
        print("\n=== Vision encoder (from config.json -> vision_config) ===")
        print(f"  depth                    {vision.get('depth')}     (vision transformer layers)")
        print(f"  hidden_size              {vision.get('hidden_size')}")
        print(f"  intermediate_size        {vision.get('intermediate_size')}")
        print(f"  num_heads                {vision.get('num_heads')}")
        print(f"  patch_size               {vision.get('patch_size')}")
        print(f"  out_hidden_size          {vision.get('out_hidden_size')}     (projected into LLM hidden_size)")


def describe_file(name, size, index):
    """One-line description for a Hub file."""
    if name in FILE_ROLES:
        if name == "model.safetensors.index.json":
            weight_map = (index or {}).get("weight_map") or {}
            total = ((index or {}).get("metadata") or {}).get("total_size")
            n_shards = len(set(weight_map.values()))
            return (
                FILE_ROLES[name]
                + f" ({len(weight_map)} tensors, {n_shards} shards, {file_size(total)})."
            )
        return FILE_ROLES[name]

    match = re.match(r"model-(\d+)-of-(\d+)\.safetensors", name)
    if match:
        part, n_parts = match.group(1), match.group(2)
        weight_map = (index or {}).get("weight_map") or {}
        n_tensors = sum(1 for shard in weight_map.values() if shard == name)
        return (
            f"Weight shard {int(part)} of {int(n_parts)} "
            f"({n_tensors} tensors, {file_size(size)})."
        )

    return "Supporting file in the repo."


def main():
    
    print(f"Starting the script and providing the model id {REPO_ID} informaiton")
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    info = api.model_info(REPO_ID)
    print(f"Repo     : {info.id}")
    print(f"Tree URL : https://huggingface.co/{REPO_ID}/tree/main")
    print(f"Task     : {info.pipeline_tag}")
    print(f"Library  : {info.library_name}")

    config = load_json(REPO_ID, "config.json", token)
    index = load_json(REPO_ID, "model.safetensors.index.json", token)
    print_architecture(config)

    print("\n=== Files on the Hub ===")
    total = 0
    files = sorted(api.list_repo_tree(REPO_ID, recursive=True), key=lambda item: item.path)
    for item in files:
        if getattr(item, "type", "file") != "file":
            continue
        size = getattr(item, "size", 0) or 0
        total += size
        print(f"\n{item.path}")
        print(f"  size : {file_size(size)}")
        print(f"  role : {describe_file(item.path, size, index)}")

    print(f"\nTotal listed size: {file_size(total)}")
    print("Weight shards were not downloaded.")


if __name__ == "__main__":
    main()
