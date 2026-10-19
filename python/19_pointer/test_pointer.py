"""Tests for the Pointer Networks slice. Run with: python3 test_pointer.py

The pointer output is a distribution over input positions, so two properties must hold by
construction and one is empirical:
  * masked positions get exactly zero probability,
  * a complete decode is a permutation (no position can be chosen twice),
  * trained on n=6, accuracy holds up on larger n.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import PointerNet, accuracy, make_example, predict, train  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_gradient():
    rng = np.random.default_rng(0)
    model = PointerNet(hid=5, att=5, seed=1)
    xs, order = make_example(rng, 4)

    def loss():
        return model.loss_and_grad(xs, order, order)[0]

    loss()
    _, grads = model.loss_and_grad(xs, order, order)
    eps = 1e-6
    worst_abs = 0.0
    worst_rel = 0.0
    for name, p in model.params():
        flat = p.reshape(-1)
        for i in range(flat.size):
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
    check("pointer gradient vs finite differences", worst_abs < 1e-7,
          f"max_abs = {worst_abs:.2e}, max_rel = {worst_rel:.2e}")


def test_masking_is_exact():
    """A position already chosen must have probability exactly zero next step."""
    rng = np.random.default_rng(3)
    model = PointerNet(hid=16, att=16, seed=2)
    xs, order = make_example(rng, 6)
    alphas = model.forward(xs, order)
    checked = 0
    for t in range(1, len(order)):
        for j in order[:t]:
            if alphas[t, j] != 0.0:
                checked = -1
    check(
        "chosen positions get exactly zero attention at later steps",
        checked == 0,
        f"checked {len(order) - 1} steps",
    )


def test_output_is_a_permutation():
    rng = np.random.default_rng(4)
    model = PointerNet(hid=16, att=16, seed=2)
    train(model, steps=300, n=6, batch=16, lr=3e-3, seed=0)
    ok = True
    for _ in range(30):
        xs, _ = make_example(rng, 8)
        pred = predict(model, xs)
        if sorted(pred) != list(range(8)):
            ok = False
    check("decoding 8 elements always yields a permutation", ok)


def test_trained_accuracy():
    model = PointerNet(hid=32, att=32, seed=0)
    losses = train(model, steps=600, n=6, batch=16, lr=3e-3, seed=0)
    acc6 = accuracy(model, n=6, samples=100, seed=2)
    check("training loss falls below 0.1", float(np.mean(losses[-20:])) < 0.1,
          f"{np.mean(losses[-20:]):.3f} nats")
    check("sorts unseen 6-element inputs", acc6 > 0.85, f"accuracy {acc6:.3f}")


def test_generalises_to_larger_n():
    """The paper's claim: train on small n, the pointer mechanism still works on larger n."""
    model = PointerNet(hid=32, att=32, seed=0)
    train(model, steps=600, n=6, batch=16, lr=3e-3, seed=0)
    accs = {n: accuracy(model, n=n, samples=100, seed=2) for n in (6, 8, 10)}
    check(
        "trained on 6, still sorts 10",
        accs[10] > 0.5,
        f"n=6 {accs[6]:.2f}, n=8 {accs[8]:.2f}, n=10 {accs[10]:.2f}",
    )
    check(
        "accuracy degrades with n rather than staying flat",
        accs[6] > accs[10],
        f"{accs[6]:.2f} -> {accs[10]:.2f}",
    )


if __name__ == "__main__":
    test_gradient()
    test_masking_is_exact()
    test_output_is_a_permutation()
    test_trained_accuracy()
    test_generalises_to_larger_n()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
