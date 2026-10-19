"""Tests for the AlexNet slice. Run with: python3 test_alexnet.py

The paper's §3.1 claim is a convergence-speed claim. It is measured here, and — as the
result shows — not reproduced at this scale. The tests pin three separate things:

  1. the conv/pool backward pass is correct (checked with tanh, see below);
  2. why ReLU cannot be finite-difference checked (a kink, demonstrated exactly);
  3. the convergence comparison, recorded as a characterisation rather than a promise.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from _shared.gradcheck import check_gradients  # noqa: E402
from implementation import (  # noqa: E402
    Act,
    SmallConvNet,
    make_task,
    train_arm,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_conv_backward_with_smooth_activation():
    """Conv and pooling backward, checked with tanh.

    ReLU is deliberately not used here: its derivative is piecewise constant, so a
    finite-difference probe near a kink measures the kink, not the gradient (next test).
    """
    xs, ys = make_task(n_per_class=1, seed=0)
    x, y = xs[0], int(ys[0])
    model = SmallConvNet("tanh", seed=0)
    loss, grads = model.loss_and_grads(x, y)

    def loss_fn():
        return model.loss_and_grads(x, y)[0]

    flat_params = {}
    analytic = {}
    for key in model.params:
        for i in range(2):
            name = f"{key}_{i}"
            flat_params[name] = model.params[key][i]
            analytic[name] = grads[key][i]
    max_abs, max_rel, ok = check_gradients(loss_fn, flat_params, analytic, eps=1e-6)
    check(
        "conv/pool/linear backward vs finite differences (tanh)",
        ok,
        f"max_abs = {max_abs:.2e}, max_rel = {max_rel:.2e}",
    )


def test_relu_kink_breaks_finite_differences():
    """Exactly why the check above uses tanh.

    At x = -1e-9 the analytic derivative of ReLU is 0 (the unit is off), but a central
    difference with step 1e-6 straddles the kink at 0 and returns 0.5. This is not a
    bug in the backward pass; it is the reason a ReLU network cannot be gradient-checked
    by perturbing near zero.
    """
    act = Act("relu")
    x = np.array([[0.0]])
    act.forward(x)
    analytic = float(act.backward(np.array([[1.0]]))[0, 0])

    eps = 1e-6
    f = lambda v: max(0.0, v)
    numeric = (f(0.0 + eps) - f(0.0 - eps)) / (2 * eps)

    check("analytic ReLU derivative at 0 is 0", analytic == 0.0, f"{analytic}")
    check(
        "central difference across the kink returns 0.5",
        abs(numeric - 0.5) < 1e-9,
        f"{numeric}",
    )


def test_activation_derivative_bounds():
    """The mechanism the paper is talking about: saturation."""
    relu = Act("relu")
    tangent = Act("tanh")
    xs = np.linspace(-5, 5, 101).reshape(1, -1)
    relu.forward(xs)
    dr = relu.backward(np.ones_like(xs))
    tangent.forward(xs)
    dt = tangent.backward(np.ones_like(xs))
    uniq = np.unique(dr)
    check("ReLU derivative is exactly 0 or 1", bool(np.all(np.isin(uniq, [0.0, 1.0]))), f"{uniq}")
    check("tanh derivative never exceeds 1", float(dt.max()) <= 1.0, f"max {dt.max():.4f}")
    check(
        "tanh derivative collapses where |x| > 3",
        float(dt[0][np.abs(xs[0]) > 3].max()) < 0.02,
        f"max {dt[0][np.abs(xs[0]) > 3].max():.4f}",
    )


def test_convergence_comparison():
    """Characterisation of the measurement, not a promise about the paper's figure.

    On this task both activations fit quickly and tanh is marginally faster. That is the
    opposite of the paper's six-fold ReLU advantage, and the README explains why the two
    settings are not comparable (CIFAR-10, four conv layers, learning rate tuned per
    network).
    """
    xs, ys = make_task(n_classes=10, n_per_class=3, size=8, noise=0.15, seed=0)
    results = {}
    for act in ("relu", "tanh"):
        reach, final_err = train_arm(act, xs, ys, steps=200, lr=0.02, seed=0, init_scale=1.0)
        results[act] = (reach, final_err)

    check(
        "both arms reach 25% training error within the budget",
        results["relu"][0] is not None and results["tanh"][0] is not None,
        f"relu {results['relu'][0]} steps, tanh {results['tanh'][0]} steps",
    )
    check(
        "at this scale tanh is not slower than ReLU (paper's claim not reproduced)",
        results["tanh"][0] <= results["relu"][0],
        f"relu {results['relu'][0]} vs tanh {results['tanh'][0]} steps",
    )


if __name__ == "__main__":
    test_conv_backward_with_smooth_activation()
    test_relu_kink_breaks_finite_differences()
    test_activation_derivative_bounds()
    test_convergence_comparison()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
