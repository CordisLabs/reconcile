"""Tests for the RLHF slice. Run with: python3 test_rlhf.py

The judge here is the Bayes rate: preferences are sampled through a known Bradley-Terry
model, so the best any model can do is computable. Reaching it means the reward model has
recovered the ordering the labels actually carry; falling short means it has not.

Comparing accuracy to a fixed threshold alone would hide the difference between "the
model is weak" and "the labels are noisy", so both numbers are reported.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import (  # noqa: E402
    RewardModel,
    bradley_terry_prob,
    candidate_set_accuracy,
    empirical_bayes_rate,
    make_dataset,
    preference_accuracy,
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
    model = RewardModel(5, hidden=6, seed=1)
    a, b = rng.normal(size=5), rng.normal(size=5)
    for label in (1, 0):
        _, grads = model.loss_and_grad(a, b, label)

        def loss():
            return model.loss_and_grad(a, b, label)[0]

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
        check(f"Bradley-Terry gradient (label {label})", worst < 1e-7, f"max_abs = {worst:.2e}")


def test_bradley_terry_shape():
    check("equal rewards give probability 1/2", abs(bradley_terry_prob(0.0) - 0.5) < 1e-12)
    check("a large advantage saturates high", bradley_terry_prob(20.0) > 0.999)
    check("a large disadvantage saturates low", bradley_terry_prob(-20.0) < 1e-8)


def run_regime(temperature, seed=0):
    rng = np.random.default_rng(seed)
    d = 10
    w = rng.normal(size=d)
    train_data = make_dataset(2000, d, w, temperature, rng)
    test_data = make_dataset(500, d, w, temperature, rng)
    model = RewardModel(d, hidden=32, seed=0)
    losses = train(model, train_data, steps=400, lr=3e-3, seed=0)
    acc = preference_accuracy(model, test_data)
    bayes = empirical_bayes_rate(d, w, temperature, rng, n=4000)
    cand = candidate_set_accuracy(model, w, d, k=5, n=200, rng=np.random.default_rng(1))
    return losses, acc, bayes, cand


def test_preference_accuracy_beats_threshold():
    """The syllabus asks for > 70% on held-out pairs."""
    losses, acc, bayes, _ = run_regime(0.3)
    check("training loss falls", float(np.mean(losses[-20:])) < 0.3,
          f"{losses[0]:.3f} -> {np.mean(losses[-20:]):.3f}")
    check("held-out preference accuracy > 0.70", acc > 0.70, f"{acc:.3f}")


def test_it_reaches_the_bayes_ceiling():
    """The labels are noisy by construction; the model should hit the best possible rate."""
    for temperature in (0.3, 1.0):
        _, acc, bayes, cand = run_regime(temperature)
        check(
            f"temp {temperature}: accuracy within 0.05 of the Bayes rate",
            abs(acc - bayes) < 0.05,
            f"model {acc:.3f} vs Bayes {bayes:.3f}",
        )
        check(
            f"temp {temperature}: picks the best of 5 far above chance",
            cand > 0.5,
            f"top-1 {cand:.3f} vs chance {1 / 5:.2f}",
        )


if __name__ == "__main__":
    test_gradient()
    test_bradley_terry_shape()
    test_preference_accuracy_beats_threshold()
    test_it_reaches_the_bayes_ceiling()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
