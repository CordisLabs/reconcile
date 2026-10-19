"""Tests for the relational reasoning slice. Run with: python3 test_relational.py

The module's structural claim is exact: the sum over ordered pairs cannot see the order of
the objects. The empirical claim is that the module can learn a task that needs relations.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import (  # noqa: E402
    RelationNet,
    accuracy,
    build_test_set,
    make_example,
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
    model = RelationNet(hidden=8, seed=1)
    objects, question, answer = make_example(rng, K=4)

    def loss():
        return model.loss_and_grad(objects, question, answer)[0]

    loss()
    _, grads = model.loss_and_grad(objects, question, answer)
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
    check("relation module gradient vs finite differences", worst < 1e-7, f"max_abs = {worst:.2e}")


def test_exact_permutation_invariance():
    """The sum over ordered pairs makes the module blind to object order."""
    rng = np.random.default_rng(1)
    model = RelationNet(hidden=32, seed=0)
    objects, question, _ = make_example(rng, K=6)
    base = model.forward(objects, question)
    worst = 0.0
    for _ in range(10):
        perm = rng.permutation(6)
        out = model.forward(objects[perm], question)
        worst = max(worst, float(np.abs(out - base).max()))
    check(
        "output is identical under object permutations",
        worst < 1e-12,
        f"max |Δ| over 10 permutations = {worst:.2e}",
    )


def test_it_learns_the_counting_task():
    model = RelationNet(hidden=32, seed=0)
    losses = train(model, steps=1500, batch=32, K=6, lr=3e-3, seed=0)
    test = build_test_set(200, 6, 1)
    acc = accuracy(model, test)
    check("training loss falls below 0.5", float(np.mean(losses[-20:])) < 0.5,
          f"{np.mean(losses[-20:]):.3f} nats")
    check("counts objects of a queried colour", acc > 0.9, f"accuracy {acc:.3f}")


def test_accuracy_is_unchanged_by_permutation():
    """Because the module is invariant, shuffling objects cannot change an answer.

    This is the empirical consequence of the previous test, and it is the reason the
    module is useful for sets: the evaluation does not depend on how the set is listed.
    """
    model = RelationNet(hidden=32, seed=0)
    train(model, steps=1500, batch=32, K=6, lr=3e-3, seed=0)
    test = build_test_set(200, 6, 1)
    normal = accuracy(model, test)
    shuffled = accuracy(model, test, permute_seed=7)
    check(
        "same accuracy with objects shuffled",
        abs(normal - shuffled) < 1e-12,
        f"{normal:.3f} vs {shuffled:.3f}",
    )


if __name__ == "__main__":
    test_gradient()
    test_exact_permutation_invariance()
    test_it_learns_the_counting_task()
    test_accuracy_is_unchanged_by_permutation()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
