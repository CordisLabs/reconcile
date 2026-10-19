"""Prints the numbers cited in this slice's README.
Run: python3 report.py
"""

import os
import time

import numpy as np

from implementation import CharRNN, encode, order0_entropy, train

HERE = os.path.dirname(os.path.abspath(__file__))


def eval_bits(model, data, seq=48):
    losses = [
        model.loss(data[s:s + seq][None, :], data[s + 1:s + seq + 1][None, :])
        for s in range(0, len(data) - seq - 1, seq)
    ]
    return float(np.mean(losses)) / np.log(2)


def main():
    text = open(os.path.join(HERE, "..", "_shared", "data", "alice_20k.txt"), encoding="utf-8").read()
    chars, stoi, itos = encode(text)
    data = np.array([stoi[c] for c in text])
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    val_text = "".join(itos[int(i)] for i in val_data)

    print("== corpus ==")
    print(f"  characters      : {len(text)}")
    print(f"  unique chars    : {len(chars)}")
    print(f"  train / val     : {split} / {len(data) - split}")
    print(f"  order-0 entropy : {order0_entropy(text):.4f} bits/char (whole text)")
    print(f"  order-0 (val)   : {order0_entropy(val_text):.4f} bits/char")
    print(f"  uniform bound   : {np.log2(len(chars)):.4f} bits/char")
    print()

    print("== capacity ablation (2000 steps each) ==")
    print("  hidden   time     train bits/char   val bits/char")
    results = {}
    for hidden in [16, 32, 64]:
        model = CharRNN(len(chars), hidden, seed=0)
        t0 = time.time()
        trace = train(model, train_data, steps=2000, seq_len=48, batch_size=16, lr=3e-3, seed=0)
        dt = time.time() - t0
        train_bits = float(np.mean(trace[-50:])) / np.log(2)
        val_bits = eval_bits(model, val_data)
        results[hidden] = (model, val_bits)
        print(f"  {hidden:>6}   {dt:5.1f}s   {train_bits:>15.4f}   {val_bits:>13.4f}")
    print()

    model, val_bits = results[64]
    print("== sample from the hidden=64 model (temperature 0.7) ==")
    sample = model.sample(stoi, itos, text[:24], 220, np.random.default_rng(0), temperature=0.7)
    for line in sample.split("\n"):
        print("  " + line)
    print()
    print(f"  held-out {val_bits:.4f} bits/char vs order-0 {order0_entropy(val_text):.4f}")


if __name__ == "__main__":
    main()
