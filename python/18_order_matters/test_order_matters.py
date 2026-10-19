"""Tests for the Order Matters slice. Run with: python3 test_order_matters.py

The architectural claim is exact and cheap to check: permuting a set must not change the
encoder's pooled context. The empirical claim — that this also trains better — is
measured separately on the sorting task.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import (  # noqa: E402
    SetPointer,
    accuracy,
    encode_permutation_gap,
    make_example,
    train,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_gradients():
    rng = np.random.default_rng(0)
    for encoder in ("set", "lstm"):
        model = SetPointer(hid=5, att=5, encoder=encoder, seed=1)
        xs, order = make_example(rng, 4)

        def loss():
            return model.loss_and_grad(xs, order, order)[0]

        loss()
        _, grads = model.loss_and_grad(xs, order, order)
        eps = 1e-6
        worst = 0.0
        for name, p in model.params():
            flat = p.reshape(-1)
            for i in range(flat.size):
                orig = flat[i]
                flat[i] = orig + eps
                plus = loss()
                flat[i] = orig - eps
                minus = loss()
                flat[i] = orig
                worst = max(worst, abs((plus - minus) / (2 * eps) - grads[name].reshape(-1)[i]))
        check(f"{encoder} encoder gradient vs finite differences", worst < 1e-7,
              f"max_abs = {worst:.2e}")


def test_set_encoder_is_invariant():
    """The paper's structural point, measured: the pooled context cannot depend on order."""
    rng = np.random.default_rng(1)
    xs = rng.random(8)
    set_model = SetPointer(hid=32, att=32, encoder="set", seed=0)
    lstm_model = SetPointer(hid=32, att=32, encoder="lstm", seed=0)
    set_gap = encode_permutation_gap(set_model, xs, rng, trials=30)
    lstm_gap = encode_permutation_gap(lstm_model, xs, rng, trials=30)
    check(
        "set encoder's pooled context is permutation-invariant",
        set_gap < 1e-15,
        f"max change over 30 permutations = {set_gap:.2e}",
    )
    check(
        "lstm encoder's pooled context is not",
        lstm_gap > 1e-3,
        f"max change over 30 permutations = {lstm_gap:.2e}",
    )


def test_decoded_values_are_permutation_equivariant():
    """A permutation of the input permutes the pointers but not the values emitted."""
    rng = np.random.default_rng(2)
    model = SetPointer(hid=32, att=32, encoder="set", seed=0)
    train(model, steps=400, n=6, batch=16, lr=3e-3, seed=0)
    xs, _ = make_example(rng, 6)
    vals_a = [xs[j] for j in predict(model, xs)]
    perm = rng.permutation(6)
    xs_p = xs[perm]
    vals_b = [xs_p[j] for j in predict(model, xs_p)]
    check(
        "the decoder emits the same values for a permuted input",
        np.allclose(vals_a, vals_b, atol=1e-9),
        f"{np.round(vals_a, 4)} vs {np.round(vals_b, 4)}",
    )


def predict(model, xs):
    chosen = []
    for _ in range(len(xs)):
        alphas = model.forward(xs, chosen + [0])
        chosen.append(int(np.argmax(alphas[-1])))
    return chosen


def test_training_comparison():
    results = {}
    for encoder in ("set", "lstm"):
        model = SetPointer(hid=32, att=32, encoder=encoder, seed=0)
        train(model, steps=600, n=6, batch=16, lr=3e-3, seed=0)
        results[encoder] = (accuracy(model, n=6, samples=100, seed=2),
                            accuracy(model, n=8, samples=100, seed=2))
    check(
        "the invariant encoder sorts unseen 6-sets perfectly",
        results["set"][0] > 0.95,
        f"set {results['set'][0]:.3f}",
    )
    check(
        "the invariant encoder extrapolates to 8",
        results["set"][1] > 0.95,
        f"set n=8 {results['set'][1]:.3f}",
    )
    check(
        "it also beats the order-dependent encoder at n=8",
        results["set"][1] > results["lstm"][1] + 0.05,
        f"set {results['set'][1]:.3f} vs lstm {results['lstm'][1]:.3f}",
    )


if __name__ == "__main__":
    test_gradients()
    test_set_encoder_is_invariant()
    test_decoded_values_are_permutation_equivariant()
    test_training_comparison()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
