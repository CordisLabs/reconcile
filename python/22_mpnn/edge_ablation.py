"""Does the edge feature earn its place?

The slice's message function ignored edges. The paper uses them (bond types in QM9). This
runs a task whose answer depends on the edge weights — the target is a sum over edges of
weight times the product of the two node features — and trains the same network with and
without access to them.

Run: python3 edge_ablation.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import MessagePassingNet


def make_batch(rng, n=6, node_dim=4, batch=8):
    out = []
    for _ in range(batch):
        x = rng.normal(size=(n, node_dim))
        adj = (rng.random((n, n)) < 0.6).astype(float)
        adj = np.triu(adj, 1)
        adj = adj + adj.T
        edges = rng.random((n, n, 1))
        # target depends on the edge weights: weight * x_i[0] * x_j[0] over edges
        target = float((adj * edges[:, :, 0] * np.outer(x[:, 0], x[:, 0])).sum())
        out.append((x, adj, edges, target))
    return out


def train(edge_dim, steps=1500, seed=0):
    rng = np.random.default_rng(seed)
    model = MessagePassingNet(node_dim=4, hidden=32, steps=2, seed=seed, edge_dim=edge_dim)
    params = model.params()
    m = {k: np.zeros_like(v) for k, v in params}
    v = {k: np.zeros_like(v) for k, v in params}
    t = 0
    for _ in range(steps):
        acc = {k: np.zeros_like(vv) for k, vv in model.params()}
        loss = 0.0
        for x, adj, e, target in make_batch(rng):
            l, g = model.loss_and_grad(x, adj, target, edges=e)
            loss += l
            for k in acc:
                acc[k] += g[k]
        t += 1
        for k, arr in model.params():
            gr = acc[k] / 8.0
            m[k] = 0.9 * m[k] + 0.1 * gr
            v[k] = 0.999 * v[k] + 0.001 * gr ** 2
            arr -= 3e-3 * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    # held-out error
    errs = []
    for x, adj, e, target in make_batch(np.random.default_rng(999)):
        errs.append(abs(model.forward(x, adj, edges=e) - target))
    return float(np.mean(errs))


def main():
    for name, edge_dim in (("no edges", 0), ("with edges", 1)):
        mae = train(edge_dim)
        print(f"{name:>12}: held-out MAE {mae:.4f}")
    return


if __name__ == "__main__":
    main()
