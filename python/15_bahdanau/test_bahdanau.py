"""Tests for the Bahdanau attention slice. Run with: python3 test_bahdanau.py

The task is sequence reversal, so the correct alignment is known exactly: target step t
must attend to source position L-1-t. That makes "the soft alignments agree with our
intuition" (the paper's own phrasing) a number rather than a heat map.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from implementation import (  # noqa: E402
    BahdanauTranslator,
    evaluate,
    make_pair,
    train,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_gradient():
    rng = np.random.default_rng(0)
    model = BahdanauTranslator(10, emb=6, hid=5, att=5, seed=1)
    src, tin, tout = make_pair(rng, 4, 6)
    _, grads = model.loss_and_grad(src, tin, tout)

    def loss():
        return model.loss_and_grad(src, tin, tout)[0]

    max_abs, max_rel, ok = _check_first_entries(model, loss, grads, limit=40)
    check(
        "attention model gradient vs finite differences",
        ok,
        f"max_abs = {max_abs:.2e}, max_rel = {max_rel:.2e} (first 40 entries per parameter)",
    )


def _check_first_entries(model, loss_fn, grads, limit, eps=1e-6):
    worst_abs = 0.0
    worst_rel = 0.0
    for name, p in model.params():
        flat = p.reshape(-1)
        for i in range(min(limit, flat.size)):
            orig = flat[i]
            flat[i] = orig + eps
            plus = loss_fn()
            flat[i] = orig - eps
            minus = loss_fn()
            flat[i] = orig
            num = (plus - minus) / (2 * eps)
            ana = grads[name].reshape(-1)[i]
            d = abs(num - ana)
            worst_abs = max(worst_abs, d)
            scale = max(abs(num), abs(ana))
            if scale > 1e-7:
                worst_rel = max(worst_rel, d / scale)
    # absolute tolerance: relative error on near-zero entries is dominated by the
    # finite-difference roundoff, the same effect documented in _shared/gradcheck.py
    return worst_abs, worst_rel, worst_abs < 1e-7


def test_attention_is_a_distribution():
    rng = np.random.default_rng(1)
    model = BahdanauTranslator(9, emb=8, hid=24, att=24, seed=0)
    src, tin, _ = make_pair(rng, 6, 6)
    _, alphas = model.forward(src, tin)
    rows = alphas.sum(axis=1)
    check(
        "every attention row sums to 1",
        np.allclose(rows, 1.0, atol=1e-12),
        f"max |sum-1| = {np.abs(rows - 1).max():.2e}",
    )
    check("attention weights are non-negative", bool((alphas >= 0).all()))


def test_trained_model_copies_and_aligns():
    model = BahdanauTranslator(9, emb=8, hid=24, att=24, seed=0)
    losses = train(model, steps=300, length=5, n_symbols=6, batch=32, lr=3e-3, seed=0)
    exact, align = evaluate(model, n=80, length=5, n_symbols=6, seed=1)
    check("training loss falls below 0.1", float(np.mean(losses[-20:])) < 0.1,
          f"{np.mean(losses[-20:]):.3f} nats")
    check("greedy decoding reverses unseen length-5 sequences", exact > 0.9, f"exact match {exact:.2f}")
    check("the alignment peak is on the correct anti-diagonal", align > 0.9, f"hit rate {align:.2f}")


def test_length_generalisation_is_limited():
    """Characterisation: training on lengths 3-6 extrapolates one step, then stops.

    This is the honest boundary of the toy result. The attention pattern degrades more
    slowly than the decoder output, which is why both are reported.
    """
    model = BahdanauTranslator(9, emb=8, hid=24, att=24, seed=0)
    train(model, steps=300, length=(3, 6), n_symbols=6, batch=32, lr=3e-3, seed=0)
    ex7, al7 = evaluate(model, n=60, length=7, n_symbols=6, seed=2)
    ex9, al9 = evaluate(model, n=60, length=9, n_symbols=6, seed=2)
    check("generalises one length beyond training", ex7 > 0.8, f"length 7 exact {ex7:.2f}, align {al7:.2f}")
    check("collapses well beyond training", ex9 < 0.5, f"length 9 exact {ex9:.2f}, align {al9:.2f}")


if __name__ == "__main__":
    test_gradient()
    test_attention_is_a_distribution()
    test_trained_model_copies_and_aligns()
    test_length_generalisation_is_limited()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
