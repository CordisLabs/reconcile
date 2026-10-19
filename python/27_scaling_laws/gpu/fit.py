"""Fit L(N) = a * N^-alpha to the measured losses, split by data regime.

Run: python3 fit.py    (numpy only; no GPU needed)

The point of the split: a model trained on fewer tokens than it has parameters is
data-limited, and including those points in a single fit pulls the exponent down and
worsens the fit. Kaplan et al. fit each regime separately for the same reason.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fit(n, loss):
    a_mat = np.vstack([np.log(n), np.ones_like(n)]).T
    (slope, intercept), *_ = np.linalg.lstsq(a_mat, np.log(loss), rcond=None)
    pred = np.exp(intercept) * n ** slope
    r2 = 1.0 - float(((loss - pred) ** 2).sum() / ((loss - loss.mean()) ** 2).sum())
    return -slope, float(np.exp(intercept)), r2


def main():
    with open(os.path.join(HERE, "results.json")) as fh:
        rows = json.load(fh)
    n = np.array([r["params"] for r in rows], float)
    loss = np.array([r["val_loss"] for r in rows], float)
    tokens = rows[0]["tokens"]
    ratio = tokens / n

    print(f"{'N':>12}  {'val loss':>9}  {'D/N':>7}")
    for r in rows:
        print(f"{r['params']:>12,}  {r['val_loss']:>9.4f}  {tokens / r['params']:>7.1f}")

    print(f"\n{'regime':<22} {'points':>6} {'alpha':>8} {'r2':>8}")
    regimes = [("all", np.ones_like(n, bool))]
    for threshold in (2.0, 5.0, 10.0):
        regimes.append((f"D/N > {threshold:g}", ratio > threshold))
    for label, mask in regimes:
        if mask.sum() < 2:
            continue
        alpha, _, r2 = fit(n[mask], loss[mask])
        print(f"{label:<22} {mask.sum():>6} {alpha:>8.4f} {r2:>8.4f}")

    alpha, a, r2 = fit(n, loss)
    print(f"\nall points: L = {a:.4f} * N^-{alpha:.4f}")
    print(f"the paper's exponent for model size is about 0.076")


if __name__ == "__main__":
    main()
