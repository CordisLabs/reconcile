"""Memory retention: carry the first symbol across a long sequence.

Delayed recall of an arbitrary index turned out to be a binding problem that neither
model could learn at this capacity, so the comparison moves to the simpler question the
RMC paper actually cares about: does information survive a long sequence? The model reads
a sequence and, at the end, emits the symbol it saw first.

Run: ~/.venvs/gpu/bin/python first_symbol.py
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rmc_vs_lstm import LSTMNet, N_SYM  # noqa: E402

DIM = 64
N_SLOTS = 8


def batch(n, length, device, gen):
    seq = torch.randint(1, N_SYM, (n, length), device=device)
    return seq, seq[:, 0]


def make_model(rmc, rmc_step=False):
    """Same controller as the recall task, but the readout is over the sequence only."""
    return LSTMNet(rmc, rmc_step=rmc_step)


def train(rmc, length, steps, device, lr=3e-3, bs=64, seed=0, rmc_step=False,
          rmc_center=False, rmc_self=False):
    torch.manual_seed(seed)
    model = LSTMNet(rmc, rmc_step=rmc_step, rmc_center=rmc_center, rmc_self=rmc_self).to(device)
    # the recall head expects an index; give it a constant zero index
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    for _ in range(steps):
        seq, ans = batch(bs, length, device, gen)
        idx = torch.zeros(bs, dtype=torch.long, device=device)
        loss = F.cross_entropy(model(seq, idx), ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, time.time() - t0


@torch.no_grad()
def evaluate(model, length, device, batches=20, bs=128):
    model.eval()
    correct = total = 0
    gen = torch.Generator(device=device).manual_seed(4321)
    for _ in range(batches):
        seq, ans = batch(bs, length, device, gen)
        idx = torch.zeros(bs, dtype=torch.long, device=device)
        correct += (model(seq, idx).argmax(1) == ans).sum().item()
        total += bs
    model.train()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lengths", type=str, default="5,20,50,100")
    ap.add_argument("--out", default=os.path.join(HERE, "results_first.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    rows = []
    for length in [int(x) for x in args.lengths.split(",")]:
        row = {"length": length}
        for name, rmc, step, center, slf in (("lstm", False, False, False, False),
                                            ("rmc_oneshot", True, False, False, False),
                                            ("rmc_self", True, False, False, True)):
            model, secs = train(rmc, length, args.steps, device, rmc_step=step,
                                rmc_center=center, rmc_self=slf)
            row[name] = round(evaluate(model, length, device), 4)
            row[f"{name}_secs"] = round(secs, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
