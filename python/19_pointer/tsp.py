"""Pointer networks on TSP, the task the paper is actually about.

The slice could sort a list; sorting is one-dimensional and its answer is a permutation of
a scalar order. TSP is the paper's headline experiment: points in the plane, and the output
is the order in which to visit them.

Run: python3 tsp.py --n 6
"""

import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import PointerNet, Adam  # noqa: E402


def tour_length(points, order):
    p = points[list(order) + [order[0]]]
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def optimal_tour(points):
    """Brute force over all tours; n is small by construction."""
    n = len(points)
    best, best_len = None, float("inf")
    for perm in itertools.permutations(range(1, n)):
        order = (0,) + perm
        length = tour_length(points, order)
        if length < best_len:
            best, best_len = order, length
    return list(best), best_len


def canonical(order):
    """Represent a tour up to rotation and direction."""
    n = len(order)
    i = order.index(0)
    forward = order[i:] + order[:i]
    back = [forward[0]] + forward[1:][::-1]
    return min(forward, back)


def make_example(rng, n):
    points = rng.random((n, 2))
    order, _ = optimal_tour(points)
    return points, canonical(order)


def greedy_decode(model, points):
    n = len(points)
    hs, _ = model.encode(points)
    order = []
    s = model.encode(points)[1]
    for t in range(n):
        allowed = np.ones(n, dtype=bool)
        allowed[order] = False
        pre = hs @ model.Wa + s @ model.Ua
        th = np.tanh(pre)
        e = th @ model.va
        e = np.where(allowed, e, -1e30)
        e = e - e.max()
        alpha = np.exp(e) / np.exp(e).sum()
        pick = int(np.argmax(alpha))
        order.append(pick)
        x = np.concatenate([points[pick], alpha @ hs])
        s = np.tanh(x @ model.Ws + s @ model.Us)
    return order


def beam_decode(model, points, beam=8):
    """Beam search over tours; greedy is beam=1 but keeping ties changes the result."""
    n = len(points)
    hs, h_last = model.encode(points)
    # each beam entry: (logprob, tour, state)
    beams = [(0.0, [], h_last)]
    for t in range(n):
        cand = []
        for lp, tour, s in beams:
            allowed = np.ones(n, dtype=bool)
            allowed[tour] = False
            pre = hs @ model.Wa + s @ model.Ua
            th = np.tanh(pre)
            e = th @ model.va
            e = np.where(allowed, e, -1e30)
            e = e - e.max()
            logp = e - np.log(np.exp(e).sum())
            for j in range(n):
                if allowed[j]:
                    x = np.concatenate([points[j], np.exp(logp) @ hs])
                    s_new = np.tanh(x @ model.Ws + s @ model.Us)
                    cand.append((lp + logp[j], tour + [j], s_new))
        cand.sort(key=lambda kv: -kv[0])
        beams = cand[:beam]
    return beams[0][1]


def train(n, steps, seed=0, batch=32, lr=3e-3):
    rng = np.random.default_rng(seed)
    model = PointerNet(hid=32, att=32, seed=seed, dim=2)
    opt = Adam(model.params(), lr=lr)
    losses = []
    buf = [make_example(rng, n) for _ in range(batch)]
    for _ in range(steps):
        acc = {k: np.zeros_like(v) for k, v in model.params()}
        total = 0.0
        for points, order in buf:
            loss, g = model.loss_and_grad(points, order, order)
            total += loss
            for k in acc:
                acc[k] += g[k]
        for k in acc:
            acc[k] /= len(buf)
        opt.step(model.params(), acc)
        losses.append(total / len(buf))
        buf = [make_example(rng, n) for _ in range(batch)]
    return model, losses


def evaluate(model, n, samples=40, seed=7, beam=1):
    rng = np.random.default_rng(seed)
    exact = 0
    ratios = []
    for _ in range(samples):
        points, order = make_example(rng, n)
        got_tour = beam_decode(model, points, beam) if beam > 1 else greedy_decode(model, points)
        pred = canonical(got_tour)
        if pred == order:
            exact += 1
        best = tour_length(points, order)
        got = tour_length(points, got_tour)
        ratios.append(got / best)
    return exact / samples, float(np.mean(ratios))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--beam", type=int, default=1)
    args = ap.parse_args()
    model, losses = train(args.n, args.steps)
    acc, ratio = evaluate(model, args.n, beam=args.beam)
    print(f"n={args.n} steps={args.steps}: loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    print(f"  exact tour match {acc:.3f}   length / optimal {ratio:.3f}")


if __name__ == "__main__":
    main()
