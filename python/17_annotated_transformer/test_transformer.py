"""Tests for the Annotated Transformer slice. Run with: python3 test_transformer.py

Three independent judges: finite differences for the whole graph, a causal-mask
property that must hold by construction, and the copy task where the answer is exact.
The label-smoothing check is a derived number rather than an impression.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import (  # noqa: E402
    BOS,
    EOS,
    Transformer,
    evaluate,
    positional_encoding,
    softmax,
    train,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_gradient():
    model = Transformer(8, d_model=8, n_heads=2, d_ff=16, max_len=8, seed=1)
    src, tin, tout = [3, 4, 5], [BOS, 3, 4, 5], [3, 4, 5, EOS]
    _, grads = model.loss_and_grad(src, tin, tout)

    def loss():
        return model.loss_and_grad(src, tin, tout)[0]

    eps = 1e-6
    worst_abs = 0.0
    worst_rel = 0.0
    for name, p in model.params():
        flat = p.reshape(-1)
        for i in range(min(flat.size, 25)):
            orig = flat[i]
            flat[i] = orig + eps
            plus = loss()
            flat[i] = orig - eps
            minus = loss()
            flat[i] = orig
            num = (plus - minus) / (2 * eps)
            ana = grads[name].reshape(-1)[i]
            d = abs(num - ana)
            worst_abs = max(worst_abs, d)
            scale = max(abs(num), abs(ana))
            if scale > 1e-7:
                worst_rel = max(worst_rel, d / scale)
    check(
        "full encoder-decoder gradient vs finite differences",
        worst_abs < 1e-7,
        f"max_abs = {worst_abs:.2e}, max_rel = {worst_rel:.2e} (25 entries per parameter)",
    )


def test_positional_encoding():
    pe = positional_encoding(16, 32)
    check("every position has its own row", len({tuple(np.round(r, 12)) for r in pe}) == 16)
    check("all values lie in [-1, 1]", bool(np.abs(pe).max() <= 1.0 + 1e-12), f"max {np.abs(pe).max():.3f}")
    # column 0 should be sin(pos * 1) since the divisor is 1 for i = 0
    want = np.sin(np.arange(16))
    check("even column 0 is sin(pos)", np.allclose(pe[:, 0], want, atol=1e-12))
    # the smallest frequency column cycles once over 10000^(2*1/32) positions
    check("wavelengths differ per dimension", not np.allclose(pe[:, 0], pe[:, 2]))


def test_causal_mask():
    """Position t must not see position t+1 or later."""
    model = Transformer(9, d_model=16, n_heads=2, d_ff=32, max_len=16, seed=0)
    src = [3, 4, 5, 6]
    a = [BOS, 3, 4, 5]
    b = [BOS, 3, 4, 7]           # differs only at the last position
    la = model.forward(src, a)
    lb = model.forward(src, b)
    check(
        "earlier decoder positions are unaffected by a later token",
        np.allclose(la[:-1], lb[:-1], atol=1e-12),
        f"max |Δ| on positions 0..{len(a) - 2} = {np.abs(la[:-1] - lb[:-1]).max():.2e}",
    )
    check(
        "the last position does change, so the test is not vacuous",
        not np.allclose(la[-1], lb[-1]),
        "",
    )


def test_copy_task():
    model = Transformer(9, d_model=32, n_heads=4, d_ff=64, max_len=16, seed=0)
    losses = train(model, steps=400, length=5, batch=8, lr=2e-3, seed=0)
    exact, ppl = evaluate(model, n=60, length=5, seed=5)
    check("training loss falls below 0.05", float(np.mean(losses[-20:])) < 0.05,
          f"{np.mean(losses[-20:]):.4f} nats")
    check("greedy decoding copies unseen sequences", exact > 0.95, f"exact match {exact:.2f}")
    check("perplexity on hard targets is near 1", ppl < 1.05, f"{ppl:.3f}")


def test_label_smoothing_hurts_perplexity():
    """The paper: "This hurts perplexity, as the model learns to be more unsure."

    The floor is derivable. With smoothing eps the probability of the correct token is
    capped at 1 - eps + eps/V, so the best possible perplexity on hard targets is
    1 / (1 - eps + eps/V). For eps = 0.1 and V = 9 that is 1.0976.
    """
    eps = 0.1
    V = 9
    floor = 1.0 / (1.0 - eps + eps / V)
    model = Transformer(V, d_model=32, n_heads=4, d_ff=64, max_len=16, seed=0)
    train(model, steps=400, length=5, batch=8, lr=2e-3, label_smoothing=eps, seed=0)
    exact, ppl = evaluate(model, n=60, length=5, seed=5)
    check(
        "smoothed model sits on the derived perplexity floor",
        abs(ppl - floor) < 0.02,
        f"ppl {ppl:.4f} vs floor {floor:.4f}",
    )
    check(
        "accuracy is unchanged because the task saturates",
        exact > 0.95,
        f"exact match {exact:.2f} (the paper's accuracy gain is untestable here)",
    )


if __name__ == "__main__":
    test_gradient()
    test_positional_encoding()
    test_causal_mask()
    test_copy_task()
    test_label_smoothing_hurts_perplexity()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
