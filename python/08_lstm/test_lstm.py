"""Tests for the LSTM slice. Run with: python3 test_lstm.py

The blog post's claim is about a mechanism — that the cell state carries information
across steps. That is measured directly as the norm of the gradient reaching back to
the initial hidden state, which needs no training and no benchmark.

The copy-task comparison is included, but as a characterisation test: at this scale the
vanilla RNN wins, and the test pins that so the README cannot quietly drift.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from _shared.gradcheck import check_gradients  # noqa: E402
from implementation import (  # noqa: E402
    LSTM,
    VanillaRNN,
    accuracy,
    copy_batch,
    one_hot,
    train,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def grad_of(model, T=5, seed=0):
    rng = np.random.default_rng(seed)
    seq, targets, mask = copy_batch(2, T, 4, rng)
    x = one_hot(seq, 6)

    def loss():
        return model.loss(x, targets, mask)

    loss()
    return loss, model.backward(targets, mask)


def test_gradients():
    for cls in (LSTM, VanillaRNN):
        rng = np.random.default_rng(0)
        model = cls(6, 4, 4, seed=1)
        for key in model.params:
            model.params[key] = model.params[key] + rng.normal(
                0, 0.1, model.params[key].shape
            )
        loss, grads = grad_of(model)
        max_abs, max_rel, ok = check_gradients(loss, model.params, grads, eps=1e-6)
        check(
            f"{cls.__name__} gradient vs finite differences",
            ok,
            f"max_abs = {max_abs:.2e}, max_rel = {max_rel:.2e}",
        )


def retention(cls, T, seed=0):
    model = cls(12, 32, 10, seed=seed)
    rng = np.random.default_rng(7)
    seq, targets, mask = copy_batch(1, T, 10, rng)
    x = one_hot(seq, 12)
    model.loss(x, targets, mask)
    model.backward(targets, mask)
    return float(np.linalg.norm(model.dh0))


def test_cell_state_carries_signal():
    """The conveyor belt, measured: gradient reaching back to step 0."""
    lstm_20 = retention(LSTM, 20)
    vanilla_20 = retention(VanillaRNN, 20)
    check(
        "at T=20 the LSTM reaches back with >10x the gradient",
        lstm_20 > 10.0 * vanilla_20,
        f"LSTM {lstm_20:.2e} vs vanilla {vanilla_20:.2e} ({lstm_20 / vanilla_20:.0f}x)",
    )

    lstm_40 = retention(LSTM, 40)
    vanilla_40 = retention(VanillaRNN, 40)
    check(
        "at T=40 the vanilla RNN's gradient has effectively vanished",
        vanilla_40 < 1e-9 and lstm_40 > 1e-7,
        f"LSTM {lstm_40:.2e} vs vanilla {vanilla_40:.2e}",
    )


def test_forget_gate_is_a_gate():
    """Forget bias controls how much of the cell state survives."""
    model = LSTM(12, 32, 10, seed=0, forget_bias=1.0)
    h = np.zeros((1, 32))
    x = np.zeros((1, 12))
    i, f, o, g = model._gates(x, h)
    check("forget gate in (0,1)", bool((f > 0).all() and (f < 1).all()), f"mean f = {f.mean():.4f}")
    check("input gate in (0,1)", bool((i > 0).all() and (i < 1).all()), f"mean i = {i.mean():.4f}")
    check("output gate in (0,1)", bool((o > 0).all() and (o < 1).all()), f"mean o = {o.mean():.4f}")

    high = LSTM(12, 8, 10, seed=0, forget_bias=4.0)
    low = LSTM(12, 8, 10, seed=0, forget_bias=-4.0)
    fh = high._gates(np.zeros((1, 12)), np.zeros((1, 8)))[1].mean()
    fl = low._gates(np.zeros((1, 12)), np.zeros((1, 8)))[1].mean()
    check(
        "bias moves the forget gate monotonically",
        fh > 0.9 and fl < 0.1,
        f"bias +4 -> f = {fh:.4f}, bias -4 -> f = {fl:.4f}",
    )


def test_copy_task_short_delay():
    """Can the LSTM actually learn to copy? Chance is 0.1."""
    model = LSTM(12, 32, 10, seed=0)
    train(model, steps=1500, T=5, k=10, batch=32, lr=3e-3, seed=0)
    acc = accuracy(model, 256, 5, 10, np.random.default_rng(1))
    check("LSTM learns the T=5 copy task", acc > 0.7, f"accuracy {acc:.3f} vs chance 0.100")


def test_vanilla_wins_at_this_scale():
    """Characterisation test, not an endorsement.

    With the same budget the vanilla RNN copies better than the LSTM, even though the
    LSTM reaches back with ~400x more gradient. Gradient retention is necessary for a
    long delay, not sufficient at a short one: at T=5 the vanilla RNN still has plenty
    of gradient and a quarter of the parameters to fit.

    If this ever flips, the README's 差异归因 needs rewriting.
    """
    results = {}
    for cls in (LSTM, VanillaRNN):
        model = cls(12, 32, 10, seed=0)
        train(model, steps=1500, T=5, k=10, batch=32, lr=3e-3, seed=0)
        results[cls.__name__] = accuracy(model, 256, 5, 10, np.random.default_rng(1))
    check(
        "at T=5 the vanilla RNN is not worse than the LSTM",
        results["VanillaRNN"] >= results["LSTM"],
        f"LSTM {results['LSTM']:.3f} vs vanilla {results['VanillaRNN']:.3f}",
    )


if __name__ == "__main__":
    test_gradients()
    test_cell_state_carries_signal()
    test_forget_gate_is_a_gate()
    test_copy_task_short_delay()
    test_vanilla_wins_at_this_scale()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
