"""Diagnostic: can the implementation even memorise a small fixed set of TSP instances?

If it cannot, the negative TSP result is a statement about this implementation, not about
the paper, and must be withdrawn. If it can, the model has the capacity and the failure
is about generalisation.

Run: python3 diag_overfit.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import Adam, PointerNet
from tsp import canonical, greedy_decode, make_example, tour_length


def run(n=6, train_size=64, steps=4000, seed=0):
    rng = np.random.default_rng(seed)
    fixed = [make_example(rng, n) for _ in range(train_size)]
    model = PointerNet(hid=32, att=32, seed=seed, dim=2)
    opt = Adam(model.params(), lr=3e-3)
    for step in range(steps):
        acc = {k: np.zeros_like(v) for k, v in model.params()}
        total = 0.0
        for points, order in fixed:
            loss, g = model.loss_and_grad(points, order, order)
            total += loss
            for k in acc:
                acc[k] += g[k]
        for k in acc:
            acc[k] /= len(fixed)
        opt.step(model.params(), acc)
    exact = sum(1 for points, order in fixed
                if canonical(greedy_decode(model, points)) == order) / len(fixed)
    # held-out instances from the same distribution
    rng2 = np.random.default_rng(999)
    held = [make_example(rng2, n) for _ in range(40)]
    held_acc = sum(1 for points, order in held
                   if canonical(greedy_decode(model, points)) == order) / len(held)
    return total / len(fixed), exact, held_acc


if __name__ == "__main__":
    loss, train_acc, held_acc = run()
    print(f"final training loss {loss:.4f}")
    print(f"memorised instances: {train_acc:.3f}")
    print(f"held-out instances:  {held_acc:.3f}")
