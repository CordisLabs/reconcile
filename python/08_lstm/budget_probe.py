"""Is the LSTM behind the vanilla RNN, or just behind on steps?

The slice reported that at 1500 steps the vanilla RNN copies T=5 perfectly while the LSTM
reaches 0.872. A budget difference is not the paper's claim, so this sweeps the budget.

Run: python3 budget_probe.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import LSTM, VanillaRNN, accuracy, train  # noqa: E402


def run(cls, T, steps, k=10, batch=32, seed=0):
    model = cls(12, 32, 10, seed=seed)
    train(model, steps=steps, T=T, k=k, batch=batch, lr=3e-3, seed=seed)
    return accuracy(model, 256, T, k, np.random.default_rng(1))


if __name__ == "__main__":
    print(f"{'T':>3} {'steps':>7} {'LSTM':>7} {'vanilla':>8}")
    for T in (5, 10):
        for steps in (1500, 4000, 8000):
            a = run(LSTM, T, steps)
            b = run(VanillaRNN, T, steps)
            print(f"{T:>3} {steps:>7} {a:>7.3f} {b:>8.3f}", flush=True)
