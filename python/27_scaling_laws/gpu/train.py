"""Fit a real power law: train char-level Transformers of increasing size and record loss.

Kaplan et al. fit L(N) with N the parameter count. This script trains five models on the
same corpus with the same token budget, so the only thing varying is capacity, and prints
the measured validation loss for each.

Run: ~/.venvs/gpu/bin/python train.py --tokens 3000000
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CTX = 128
BATCH = 32


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class CharGPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, mask):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))


def load(vocab_size):
    text = open(os.path.join(HERE, "corpus.txt"), encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int16)
    split = int(len(data) * 0.95)
    return data[:split], data[split:], chars


def batch_from(arr, rng):
    starts = rng.integers(0, len(arr) - CTX - 1, size=BATCH)
    x = np.stack([arr[s:s + CTX] for s in starts]).astype(np.int64)
    y = np.stack([arr[s + 1:s + 1 + CTX] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def evaluate(model, val, vocab_size, device, batches=40, seed=1234):
    rng = np.random.default_rng(seed)
    model.eval()
    total = 0.0
    mask = torch.triu(torch.ones(CTX, CTX, device=device), diagonal=1).bool()
    for _ in range(batches):
        x, y = batch_from(val, rng)
        logits = model(x.to(device), mask)
        total += F.cross_entropy(logits.reshape(-1, vocab_size), y.to(device).reshape(-1)).item()
    model.train()
    return total / batches


def train_one(d, n_layer, n_head, tokens, device, lr=1e-3, seed=0):
    train_data, val_data, chars = load(0)
    vocab = len(chars)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CharGPT(vocab, d, n_layer, n_head).to(device)
    params = sum(p.numel() for p in model.parameters())
    steps = max(1, tokens // (BATCH * CTX))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    mask = torch.triu(torch.ones(CTX, CTX, device=device), diagonal=1).bool()

    t0 = time.time()
    curve = []
    for step in range(steps):
        x, y = batch_from(train_data, rng)
        logits = model(x.to(device), mask)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.to(device).reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            vl = evaluate(model, val_data, vocab, device, batches=10)
            curve.append((step, round(vl, 4)))
    val_loss = evaluate(model, val_data, vocab, device)
    return {
        "d_model": d, "n_layer": n_layer, "n_head": n_head,
        "params": params, "tokens": steps * BATCH * CTX,
        "val_loss": round(val_loss, 4), "curve": curve,
        "seconds": round(time.time() - t0, 1),
    }


CONFIGS = [
    (64, 2, 4),
    (96, 3, 6),
    (128, 4, 8),
    (160, 5, 8),
    (192, 6, 8),
    (256, 8, 8),
    (320, 10, 8),
    (384, 12, 8),
    (448, 14, 8),
    (512, 16, 8),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=20_000_000)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--only", type=int, default=-1, help="run one config index")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    configs = CONFIGS if args.only < 0 else [CONFIGS[args.only]]
    results = []
    for d, L, h in configs:
        r = train_one(d, L, h, args.tokens, device)
        print(json.dumps(r), flush=True)
        results.append(r)
    if args.only < 0:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
