"""Tests for the attention slice. Run with: python3 test_attention.py

No pytest: the file is its own entry point, so CI has one less moving part.
Every test compares against an independent fact (finite differences, an exact
symmetry, or a normalization identity), never against "it produced numbers".
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from _shared.gradcheck import check_gradients  # noqa: E402
from attention import MultiHeadAttention, sdpa_forward, sdpa_backward  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_attention_weights_normalise():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(5, 4))
    K = rng.normal(size=(7, 4))
    V = rng.normal(size=(7, 4))
    _, cache = sdpa_forward(Q, K, V)
    P = cache[3]
    row_sums = P.sum(axis=-1)
    check(
        "attention rows sum to 1",
        np.allclose(row_sums, 1.0, atol=1e-12),
        f"max |sum-1| = {np.abs(row_sums - 1.0).max():.2e}",
    )
    check("attention weights non-negative", bool((P >= 0).all()))


def test_shapes():
    mha = MultiHeadAttention(d_model=8, n_heads=2, seed=1)
    X = np.random.default_rng(1).normal(size=(4, 8))
    Y = mha.forward(X)
    check("output shape (T, d_model)", Y.shape == (4, 8), str(Y.shape))


def test_permutation_equivariance():
    """No positional encoding yet, so attention must commute with token order."""
    mha = MultiHeadAttention(d_model=8, n_heads=2, seed=2)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(6, 8))
    perm = rng.permutation(6)

    Y = mha.forward(X)
    Y_perm = mha.forward(X[perm])
    err = np.abs(Y_perm - Y[perm]).max()
    check(
        "permutation equivariance (exact symmetry)",
        err < 1e-12,
        f"max |Δ| = {err:.2e}",
    )


def test_sdpa_gradient():
    rng = np.random.default_rng(3)
    Q = rng.normal(size=(4, 3))
    K = rng.normal(size=(4, 3))
    V = rng.normal(size=(4, 3))
    R = rng.normal(size=(4, 3))

    def loss():
        O, _ = sdpa_forward(Q, K, V)
        return float((O * R).sum())

    _, cache = sdpa_forward(Q, K, V)
    dQ, dK, dV = sdpa_backward(R, cache)

    max_abs, max_rel, ok = check_gradients(
        loss, {"Q": Q, "K": K, "V": V}, {"Q": dQ, "K": dK, "V": dV}, eps=1e-6
    )
    check(
        "scaled dot-product attention gradient vs finite differences",
        ok,
        f"max_abs = {max_abs:.2e}, max_rel = {max_rel:.2e}",
    )


def test_key_bias_gradient_is_zero():
    """∂L/∂bk is identically zero.

    Every row of dS sums to zero (softmax is invariant to shifting a row of scores
    by a constant). Summing dK over the key axis therefore cancels exactly. This is
    a *structural* zero, not a small number — and the reason a pure relative-error
    gradient check fails here while the implementation is correct.
    """
    mha = MultiHeadAttention(d_model=8, n_heads=2, seed=5)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(6, 8))
    R = rng.normal(size=(6, 8))
    mha.forward(X)
    grads = mha.backward(R)
    val = float(np.abs(grads["bk"]).max())
    check(
        "key bias gradient is structurally zero",
        val < 1e-15,
        f"max |dL/dbk| = {val:.2e}",
    )


def test_multihead_gradient():
    mha = MultiHeadAttention(d_model=8, n_heads=2, seed=4)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(4, 8))
    R = rng.normal(size=(4, 8))

    def loss():
        return float((mha.forward(X) * R).sum())

    mha.forward(X)
    analytic = mha.backward(R)

    max_abs, max_rel, ok = check_gradients(loss, mha.params, analytic, eps=1e-6)
    n_params = sum(v.size for v in mha.params.values())
    check(
        "multi-head attention gradient vs finite differences",
        ok,
        f"{n_params} params, max_abs = {max_abs:.2e}, max_rel = {max_rel:.2e}",
    )


if __name__ == "__main__":
    test_attention_weights_normalise()
    test_shapes()
    test_permutation_equivariance()
    test_sdpa_gradient()
    test_key_bias_gradient_is_zero()
    test_multihead_gradient()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
