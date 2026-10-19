"""Dropout as a regulariser needs a budget: does the effect grow with training?

The slice reported the 1200-step picture: no-dropout reaches a lower minimum and then
degrades, dropout stays flat. This repeats at four budgets to see whether the ordering is
a property of the method or of the stopping point.

Run: python3 budget_probe.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from implementation import DropoutLSTM, encode, train  # noqa: E402

DATA = os.path.join(HERE, "..", "_shared", "data", "alice_20k.txt")


def main():
    text = open(DATA, encoding="utf-8").read()[:2500]
    chars, stoi, _ = encode(text)
    data = np.array([stoi[c] for c in text])
    split = int(len(data) * 0.8)
    train_data, val_data = data[:split], data[split:]
    print(f"{'steps':>6} {'model':>10} {'best':>8} {'final':>8} {'degradation':>12}")
    for steps in (1200, 2500, 5000):
        for name, kwargs in (("none", {}), ("dropout", {"p_in": 0.5, "p_out": 0.5})):
            model = DropoutLSTM(len(chars), 64, len(chars), seed=0, **kwargs)
            best, trace = train(model, train_data, val_data, steps=steps,
                                seq_len=24, batch_size=8, lr=3e-3, seed=0,
                                eval_every=max(1, steps // 6))
            final = trace[-1][2]
            print(f"{steps:>6} {name:>10} {best:>8.3f} {final:>8.3f} {final - best:>12.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
