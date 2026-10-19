"""Fit L(N, D) jointly: the paper fits loss as a function of both size and data.

Run: ~/.venvs/gpu/bin/python joint_fit.py

Runs a small grid of model sizes against token budgets, then fits
    L = a * N^-alpha + b * D^-beta + L_inf
by least squares on the three unknowns, and reports the exponents next to the paper's.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train import train_one  # noqa: E402

SIZES = [(96, 3, 6), (160, 5, 8), (256, 8, 8), (384, 12, 8)]
BUDGETS = [2_000_000, 8_000_000, 20_000_000]


def main():
    device = "cuda"
    rows = []
    for d, layers, heads in SIZES:
        for tokens in BUDGETS:
            r = train_one(d, layers, heads, tokens, device)
            row = {"params": r["params"], "tokens": r["tokens"], "val_loss": r["val_loss"]}
            print(json.dumps(row), flush=True)
            rows.append(row)
    with open(os.path.join(HERE, "results_joint.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    n = np.array([r["params"] for r in rows], float)
    dd = np.array([r["tokens"] for r in rows], float)
    loss = np.array([r["val_loss"] for r in rows], float)

    best = None
    for linf in np.linspace(0.3, 1.6, 40):
        for alpha in np.linspace(0.01, 0.6, 80):
            for beta in np.linspace(0.01, 0.6, 80):
                A = np.vstack([n ** (-alpha), dd ** (-beta)]).T
                y = loss - linf
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                if coef.min() <= 0:
                    continue
                pred = A @ coef + linf
                res = ((loss - pred) ** 2).sum()
                if best is None or res < best[0]:
                    best = (res, alpha, beta, coef[0], coef[1], linf)
    res, alpha, beta, a, b, linf = best
    r2 = 1 - res / ((loss - loss.mean()) ** 2).sum()
    print(f"\nL = {a:.4f} N^-{alpha:.4f} + {b:.4f} D^-{beta:.4f} + {linf:.3f}   r2={r2:.4f}")


if __name__ == "__main__":
    main()
