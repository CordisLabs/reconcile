"""Tests for the relational memory cell. Run with: python3 test_relational_rnn.py

The memory is a set of slots, so the attention over them is a distribution and the whole
cell is permutation-equivariant. Both are exact. No LSTM comparison is made — that would
need a training experiment this slice does not contain.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import MHDPA, RelationalCell  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_attention_is_a_distribution():
    rng = np.random.default_rng(0)
    attn = MHDPA(slots=6, dim=8, heads=4, seed=0)
    slots = rng.normal(size=(6, 8))
    _, weights = attn.forward(slots)
    rows = weights.sum(axis=-1)
    check("every head's attention rows sum to 1", np.allclose(rows, 1.0, atol=1e-12),
          f"max |sum-1| = {np.abs(rows - 1).max():.2e}")
    check("attention weights are non-negative", bool((weights >= 0).all()))


def test_slots_interact():
    """With random projections the attention is not the uniform average."""
    rng = np.random.default_rng(1)
    attn = MHDPA(slots=6, dim=8, heads=2, seed=0)
    slots = rng.normal(size=(6, 8))
    _, weights = attn.forward(slots)
    uniform = np.full(weights.shape[-1], 1.0 / weights.shape[-1])
    check("attention is not uniform", np.abs(weights - uniform).max() > 1e-3,
          f"max deviation {np.abs(weights - uniform).max():.4f}")


def test_cell_is_permutation_equivariant():
    rng = np.random.default_rng(2)
    cell = RelationalCell(slots=5, dim=4, heads=2, seed=1)
    x = rng.normal(size=(5, 4))
    out, _ = cell.forward(x)
    worst = 0.0
    for _ in range(10):
        perm = rng.permutation(5)
        out_p, _ = cell.forward(x[perm])
        worst = max(worst, float(np.abs(out_p - out[perm]).max()))
    check("permuting slots permutes the output", worst < 1e-12, f"max |Δ| = {worst:.2e}")


def test_gradient():
    rng = np.random.default_rng(3)
    cell = RelationalCell(slots=4, dim=4, heads=2, seed=1)
    x = rng.normal(size=(4, 4))
    dout = rng.normal(size=(4, 4))

    def loss():
        out, _ = cell.forward(x)
        return float((out * dout).sum())

    loss()
    _, grads = cell.backward(dout)
    eps = 1e-6
    worst = 0.0
    for name, p in cell.params():
        flat = p.reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            plus = loss()
            flat[i] = orig - eps
            minus = loss()
            flat[i] = orig
            worst = max(worst, abs((plus - minus) / (2 * eps) - grads[name].reshape(-1)[i]))
    check("cell gradient vs finite differences", worst < 1e-7, f"max_abs = {worst:.2e}")


if __name__ == "__main__":
    test_attention_is_a_distribution()
    test_slots_interact()
    test_cell_is_permutation_equivariant()
    test_gradient()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
