"""
Minimal character-level GPT training (pure Python + custom autograd).

What this script does:
1) loads a text dataset (one sample per line)
2) trains a tiny GPT-style language model
3) saves a checkpoint for separate inference
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import urllib.request
from dataclasses import dataclass

random.seed(42)


class Value:
    """Scalar value with autograd metadata."""

    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


@dataclass
class TrainConfig:
    input_path: str = "input.trainfile"
    input_url: str = "https://github.com/qayyumu/AIworkshop/blob/main/Session-04/input.trainfile"
    checkpoint_path: str = "Session-04/llm_checkpoint.pkl"
    num_steps: int = 1000
    learning_rate: float = 0.01
    beta1: float = 0.85
    beta2: float = 0.99
    eps_adam: float = 1e-8
    n_layer: int = 1
    n_embd: int = 16
    n_head: int = 4
    block_size: int = 16
    init_std: float = 0.08
    log_every: int = 50


def maybe_prepare_input_file(path: str, source_url: str) -> None:
    if not os.path.exists(path):
        urllib.request.urlretrieve(source_url, path)


def load_docs(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        docs = [line.strip() for line in file if line.strip()]
    random.shuffle(docs)
    return docs


def build_tokenizer(docs: list[str]):
    chars = sorted(set("".join(docs)))
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for idx, ch in enumerate(chars)}
    bos = len(chars)
    vocab_size = bos + 1
    return chars, stoi, itos, bos, vocab_size


def make_matrix(n_out: int, n_in: int, std: float) -> list[list[Value]]:
    return [[Value(random.gauss(0, std)) for _ in range(n_in)] for _ in range(n_out)]


def init_state_dict(cfg: TrainConfig, vocab_size: int):
    state = {
        "wte": make_matrix(vocab_size, cfg.n_embd, cfg.init_std),
        "wpe": make_matrix(cfg.block_size, cfg.n_embd, cfg.init_std),
        "lm_head": make_matrix(vocab_size, cfg.n_embd, cfg.init_std),
    }
    for layer_idx in range(cfg.n_layer):
        state[f"layer{layer_idx}.attn_wq"] = make_matrix(cfg.n_embd, cfg.n_embd, cfg.init_std)
        state[f"layer{layer_idx}.attn_wk"] = make_matrix(cfg.n_embd, cfg.n_embd, cfg.init_std)
        state[f"layer{layer_idx}.attn_wv"] = make_matrix(cfg.n_embd, cfg.n_embd, cfg.init_std)
        state[f"layer{layer_idx}.attn_wo"] = make_matrix(cfg.n_embd, cfg.n_embd, cfg.init_std)
        state[f"layer{layer_idx}.mlp_fc1"] = make_matrix(4 * cfg.n_embd, cfg.n_embd, cfg.init_std)
        state[f"layer{layer_idx}.mlp_fc2"] = make_matrix(cfg.n_embd, 4 * cfg.n_embd, cfg.init_std)
    return state


def flatten_params(state_dict) -> list[Value]:
    return [p for matrix in state_dict.values() for row in matrix for p in row]


def linear(x, weight):
    return [sum(wi * xi for wi, xi in zip(weight_out, x)) for weight_out in weight]


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [exp_val / total for exp_val in exps]


def rmsnorm(x):
    mean_square = sum(xi * xi for xi in x) / len(x)
    scale = (mean_square + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt_step(token_id, pos_id, keys, values, state_dict, cfg: TrainConfig):
    head_dim = cfg.n_embd // cfg.n_head
    token_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [tok + pos for tok, pos in zip(token_emb, pos_emb)]
    x = rmsnorm(x)

    for layer_idx in range(cfg.n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{layer_idx}.attn_wq"])
        k = linear(x, state_dict[f"layer{layer_idx}.attn_wk"])
        v = linear(x, state_dict[f"layer{layer_idx}.attn_wv"])
        keys[layer_idx].append(k)
        values[layer_idx].append(v)

        x_attention = []
        for head_idx in range(cfg.n_head):
            hs = head_idx * head_dim
            q_head = q[hs : hs + head_dim]
            k_head = [key[hs : hs + head_dim] for key in keys[layer_idx]]
            v_head = [val[hs : hs + head_dim] for val in values[layer_idx]]
            attn_logits = [
                sum(q_head[j] * k_head[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_head))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_head[t][j] for t in range(len(v_head)))
                for j in range(head_dim)
            ]
            x_attention.extend(head_out)

        x = linear(x_attention, state_dict[f"layer{layer_idx}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{layer_idx}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{layer_idx}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, state_dict["lm_head"])


def forward_loss(tokens, state_dict, cfg: TrainConfig):
    context_len = min(cfg.block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(cfg.n_layer)], [[] for _ in range(cfg.n_layer)]
    losses = []
    for pos_id in range(context_len):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        logits = gpt_step(token_id, pos_id, keys, values, state_dict, cfg)
        probs = softmax(logits)
        losses.append(-probs[target_id].log())
    return (1 / context_len) * sum(losses)


def encode_doc(doc: str, stoi: dict[str, int], bos: int) -> list[int]:
    return [bos] + [stoi[ch] for ch in doc] + [bos]


def serialize_state_dict(state_dict):
    return {
        key: [[value.data for value in row] for row in matrix]
        for key, matrix in state_dict.items()
    }


def save_checkpoint(path: str, cfg: TrainConfig, chars: list[str], bos: int, state_dict) -> None:
    payload = {
        "config": {
            "n_layer": cfg.n_layer,
            "n_embd": cfg.n_embd,
            "n_head": cfg.n_head,
            "block_size": cfg.block_size,
        },
        "chars": chars,
        "bos": bos,
        "state_dict": serialize_state_dict(state_dict),
    }
    with open(path, "wb") as file:
        pickle.dump(payload, file)


def train(cfg: TrainConfig) -> None:
    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head.")

    maybe_prepare_input_file(cfg.input_path, cfg.input_url)
    docs = load_docs(cfg.input_path)
    chars, stoi, _itos, bos, vocab_size = build_tokenizer(docs)
    print(f"num docs: {len(docs)}")
    print(f"vocab size: {vocab_size}")

    state_dict = init_state_dict(cfg, vocab_size)
    params = flatten_params(state_dict)
    print(f"num params: {len(params)}")

    m = [0.0] * len(params)
    v = [0.0] * len(params)

    for step in range(cfg.num_steps):
        doc = docs[step % len(docs)]
        tokens = encode_doc(doc, stoi, bos)
        loss = forward_loss(tokens, state_dict, cfg)
        loss.backward()

        lr_t = cfg.learning_rate * (1 - step / cfg.num_steps)
        for i, param in enumerate(params):
            m[i] = cfg.beta1 * m[i] + (1 - cfg.beta1) * param.grad
            v[i] = cfg.beta2 * v[i] + (1 - cfg.beta2) * param.grad**2
            m_hat = m[i] / (1 - cfg.beta1 ** (step + 1))
            v_hat = v[i] / (1 - cfg.beta2 ** (step + 1))
            param.data -= lr_t * m_hat / (v_hat**0.5 + cfg.eps_adam)
            param.grad = 0

        should_log = (step + 1) % cfg.log_every == 0 or step == 0 or (step + 1) == cfg.num_steps
        if should_log:
            print(f"step {step + 1:4d}/{cfg.num_steps:4d} | loss {loss.data:.4f}")

    os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
    save_checkpoint(cfg.checkpoint_path, cfg, chars, bos, state_dict)
    print(f"saved checkpoint: {cfg.checkpoint_path}")
    print("Use `python Session-04/llm_inference.py --checkpoint <path>` for generation.")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a tiny character-level GPT.")
    parser.add_argument("--input", default="input.trainfile", help="Training text file (one sample per line).")
    parser.add_argument("--input-url", default="https://github.com/qayyumu/AIworkshop/blob/main/Session-04/input.trainfile", help="Used when --input does not exist.")
    parser.add_argument("--checkpoint", default="Session-04/llm_checkpoint.pkl", help="Where to save trained weights.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of optimization steps.")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--layers", type=int, default=1, help="Number of transformer blocks.")
    parser.add_argument("--embd", type=int, default=16, help="Embedding size.")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads.")
    parser.add_argument("--block-size", type=int, default=16, help="Max context length.")
    parser.add_argument("--log-every", type=int, default=50, help="Print interval in steps.")
    args = parser.parse_args()
    return TrainConfig(
        input_path=args.input,
        input_url=args.input_url,
        checkpoint_path=args.checkpoint,
        num_steps=args.steps,
        learning_rate=args.lr,
        n_layer=args.layers,
        n_embd=args.embd,
        n_head=args.heads,
        block_size=args.block_size,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    train(parse_args())