"""Tests for the ResNet v2 slice. Run with: python3 test_resnet_v2.py

What is measured is the gradient that reaches the input through 4 to 64 residual blocks.
What is *not* measured is trainability: without batch normalisation both variants
explode, so this is a relative-transmission check, not a reproduction of the paper's
1000-layer result.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from _shared.gradcheck import check_gradients  # noqa: E402
from implementation import ResMLP, input_gradient_norm, relu  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_gradients():
    rng = np.random.default_rng(0)
    for variant in ("pre", "post"):
        model = ResMLP(6, 4, blocks=2, width=8, variant=variant, seed=1)
        x = rng.normal(size=(5, 6))
        y = rng.integers(0, 4, size=5)

        def loss():
            return model.loss_and_grad(x, y)[0]

        loss()
        _, grads = model.loss_and_grad(x, y)
        max_abs, max_rel, ok = check_gradients(loss, dict(model.params()), grads, eps=1e-6)
        check(
            f"{variant}-activation gradient vs finite differences",
            ok,
            f"max_abs = {max_abs:.2e}, max_rel = {max_rel:.2e}",
        )


def test_identity_shortcut_transmits_more():
    """The paper's mechanism: the clean shortcut keeps the backward path open."""
    pre = input_gradient_norm("pre", 32)
    post = input_gradient_norm("post", 32)
    check(
        "at 32 blocks the identity shortcut carries >100x more gradient",
        pre > 100.0 * post,
        f"pre {pre:.2e} vs post {post:.2e} (ratio {pre / post:.0f}x)",
    )


def test_ratio_grows_with_depth():
    """The gap is not a constant offset; it widens as depth increases."""
    ratios = []
    for blocks in (4, 8, 16, 32):
        ratios.append(input_gradient_norm("pre", blocks) / input_gradient_norm("post", blocks))
    check(
        "the pre/post ratio grows monotonically with depth",
        all(b > a for a, b in zip(ratios, ratios[1:])),
        " ".join(f"{r:.1f}x" for r in ratios),
    )


def test_both_arms_explode_without_normalisation():
    """Characterisation, not a virtue.

    With no batch normalisation the residual stream accumulates and both gradient norms
    grow without bound. The paper's long-network results depend on BN; this slice cannot
    and does not claim that the identity shortcut makes 1000 layers trainable.
    """
    norms = [input_gradient_norm("pre", b) for b in (4, 16, 64)]
    check(
        "the pre-activation gradient norm grows by orders of magnitude",
        norms[2] > 1000.0 * norms[0],
        " ".join(f"{n:.2e}" for n in norms),
    )


def test_identity_initialisation():
    """Paper's eq. 4: with the residual branch zeroed, the blocks start as identity."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 6))
    for variant in ("pre", "post"):
        model = ResMLP(6, 4, blocks=8, width=8, variant=variant, seed=0, identity_init=True)
        head_only = relu(x @ model.Win + model.bin) @ model.Wout + model.bout
        logits = model.forward(x)
        check(f"{variant}: blocks are the identity at initialisation",
              np.abs(logits - head_only).max() < 1e-12,
              f"max |Δ| = {np.abs(logits - head_only).max():.2e}")


if __name__ == "__main__":
    test_identity_initialisation()
    test_gradients()
    test_identity_shortcut_transmits_more()
    test_ratio_grows_with_depth()
    test_both_arms_explode_without_normalisation()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
