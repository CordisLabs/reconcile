"""Tests for the ResNet slice. Run with: python3 test_resnet.py

The paper's first result is the degradation problem, and it is measured here on the
*training* set — which is the point: the deeper plain network is not overfitting, it is
failing to fit.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from _shared.gradcheck import check_gradients  # noqa: E402
from implementation import MLP, accuracy, make_task, train  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_gradients():
    rng = np.random.default_rng(0)
    for residual in (False, True):
        model = MLP(6, 4, blocks=2, width=8, residual=residual, seed=1)
        x = rng.normal(size=(5, 6))
        y = rng.integers(0, 4, size=5)

        def loss():
            return model.loss_and_grad(x, y)[0]

        loss()
        _, grads = model.loss_and_grad(x, y)
        params = dict(model.params())
        max_abs, max_rel, ok = check_gradients(loss, params, grads, eps=1e-6)
        label = "residual" if residual else "plain"
        check(
            f"{label} network gradient vs finite differences",
            ok,
            f"max_abs = {max_abs:.2e}, max_rel = {max_rel:.2e}",
        )


def fit(blocks, residual, seed=0, steps=400):
    x, y = make_task(n_classes=10, n_per_class=15, dim=10, seed=0)
    model = MLP(10, 10, blocks=blocks, width=32, residual=residual, seed=seed)
    losses = train(model, x, y, steps=steps, lr=1e-2, seed=seed)
    return accuracy(model, x, y), float(np.mean(losses[-20:]))


def test_degradation_on_plain_nets():
    """Deeper plain nets reach higher training error — the paper's anomaly."""
    shallow_acc, _ = fit(blocks=4, residual=False)
    deep_acc, deep_loss = fit(blocks=24, residual=False)
    check(
        "a 4-block plain net fits the training set",
        shallow_acc > 0.95,
        f"training accuracy {shallow_acc:.3f}",
    )
    check(
        "a 24-block plain net does not, and is near chance",
        deep_acc < 0.4,
        f"training accuracy {deep_acc:.3f}, loss {deep_loss:.3f} (chance 0.100)",
    )
    check(
        "degradation is a drop in training accuracy, not overfitting",
        deep_acc < shallow_acc - 0.5,
        f"{shallow_acc:.3f} -> {deep_acc:.3f}",
    )


def test_residual_removes_degradation():
    """The same depth, with the identity path, still fits."""
    deep_plain, _ = fit(blocks=24, residual=False)
    deep_res, _ = fit(blocks=24, residual=True)
    check(
        "a 24-block residual net still fits the training set",
        deep_res > 0.9,
        f"training accuracy {deep_res:.3f}",
    )
    check(
        "the residual net beats the plain net at the same depth",
        deep_res > deep_plain + 0.5,
        f"residual {deep_res:.3f} vs plain {deep_plain:.3f}",
    )


if __name__ == "__main__":
    test_gradients()
    test_degradation_on_plain_nets()
    test_residual_removes_degradation()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
