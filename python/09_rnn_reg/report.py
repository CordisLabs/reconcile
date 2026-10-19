"""Prints the numbers cited in this slice's README.
Run: python3 report.py     (about three minutes: three 4000-step runs)

The protocol is deliberately small enough for a CPU and deliberately long enough to
reach the regime the paper is about: the undropped model must start overfitting, or
the comparison says nothing.
"""

import os
import time

import numpy as np

from implementation import DropoutLSTM, encode, train

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIGS = [
    ("no dropout", {}),
    ("non-recurrent 0.5", {"p_in": 0.5, "p_out": 0.5}),
    ("also recurrent 0.5", {"p_in": 0.5, "p_out": 0.5, "p_rec": 0.5}),
]


def main():
    path = os.path.join(HERE, "..", "_shared", "data", "alice_20k.txt")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()[:6000]
    chars, stoi, _ = encode(text)
    data = np.array([stoi[c] for c in text])
    split = int(len(data) * 0.8)
    train_data, val_data = data[:split], data[split:]

    print(f"corpus {len(text)} chars, {len(chars)} unique, hidden 64, 4000 steps")
    print("  config                best val   final val   shape")
    for name, kwargs in CONFIGS:
        model = DropoutLSTM(len(chars), 64, len(chars), seed=0, **kwargs)
        t0 = time.time()
        best, trace = train(
            model, train_data, val_data, steps=4000, seq_len=32, batch_size=16,
            lr=3e-3, seed=0, eval_every=800,
        )
        final = trace[-1][2]
        shape = " ".join(f"{s}:{v:.2f}" for s, _tr, v in trace)
        print(f"  {name:<20}  {best:8.4f}   {final:9.4f}   {shape}   ({time.time() - t0:.0f}s)")



if __name__ == "__main__":
    main()
