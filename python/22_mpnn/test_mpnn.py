"""Tests for the message passing slice. Run with: python3 test_mpnn.py

A graph is a set of nodes with edges, so the graph-level readout must not depend on how
the nodes are indexed. That is exact. The training result is on a synthetic regression
whose target is the sum of one node feature.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import MessagePassingNet, make_graph, train  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_gradient():
    rng = np.random.default_rng(0)
    model = MessagePassingNet(node_dim=3, hidden=5, steps=2, seed=1)
    x, adj, target = make_graph(rng, n=4, node_dim=3)

    def loss():
        return model.loss_and_grad(x, adj, target)[0]

    loss()
    _, grads = model.loss_and_grad(x, adj, target)
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
    check("message passing gradient vs finite differences", worst < 1e-7, f"max_abs = {worst:.2e}")


def test_permutation_invariance():
    rng = np.random.default_rng(1)
    model = MessagePassingNet(node_dim=4, hidden=16, steps=3, seed=0)
    x, adj, _ = make_graph(rng, n=7, node_dim=4)
    base = model.forward(x, adj)
    worst = 0.0
    for _ in range(10):
        perm = rng.permutation(7)
        out = model.forward(x[perm], adj[np.ix_(perm, perm)])
        worst = max(worst, abs(out - base))
    check(
        "graph readout is invariant to node ordering",
        worst < 1e-12,
        f"max |Δ| over 10 permutations = {worst:.2e}",
    )


def test_node_states_are_equivariant():
    """Permuting nodes permutes the node states; it does not change them."""
    rng = np.random.default_rng(2)
    model = MessagePassingNet(node_dim=4, hidden=16, steps=2, seed=0)
    x, adj, _ = make_graph(rng, n=6, node_dim=4)
    model.forward(x, adj)
    h = model.cache[3]
    perm = rng.permutation(6)
    model.forward(x[perm], adj[np.ix_(perm, perm)])
    h_perm = model.cache[3]
    check(
        "node states permute with the nodes",
        np.abs(h_perm - h[perm]).max() < 1e-12,
        f"max |Δ| = {np.abs(h_perm - h[perm]).max():.2e}",
    )


def test_it_learns_a_graph_property():
    model = MessagePassingNet(node_dim=4, hidden=32, steps=3, seed=0)
    losses = train(model, steps=1200, batch=8, seed=0)
    rng = np.random.default_rng(5)
    errs, base = [], []
    for _ in range(50):
        x, adj, target = make_graph(rng)
        errs.append(abs(model.forward(x, adj) - target))
        base.append(abs(target))
    check("training loss falls", float(np.mean(losses[-20:])) < 1.0,
          f"{losses[0]:.3f} -> {np.mean(losses[-20:]):.3f}")
    check(
        "beats the zero predictor by at least 2x",
        np.mean(errs) < 0.5 * np.mean(base),
        f"MAE {np.mean(errs):.3f} vs baseline {np.mean(base):.3f}",
    )


def test_edge_features_are_used():
    """With edge features the output must depend on them; without them it cannot."""
    rng = np.random.default_rng(5)
    x, adj, _ = make_graph(rng, n=5, node_dim=3)
    edges = rng.random((5, 5, 1))

    with_edges = MessagePassingNet(node_dim=3, hidden=8, steps=1, seed=0, edge_dim=1)
    a = with_edges.forward(x, adj, edges)
    edges2 = edges * 0.0
    b = with_edges.forward(x, adj, edges2)
    check("output changes when the edge features change", abs(a - b) > 1e-9,
          f"{abs(a - b):.3e}")

    plain = MessagePassingNet(node_dim=3, hidden=8, steps=1, seed=0)
    check("a network without edge inputs ignores them",
          abs(plain.forward(x, adj) - plain.forward(x, adj)) < 1e-15)


def test_edge_gradient():
    rng = np.random.default_rng(6)
    model = MessagePassingNet(node_dim=3, hidden=5, steps=1, seed=1, edge_dim=1)
    x, adj, target = make_graph(rng, n=4, node_dim=3)
    edges = rng.random((4, 4, 1))
    _, grads = model.loss_and_grad(x, adj, target, edges=edges)
    eps = 1e-6
    worst = 0.0
    for name, p in model.params():
        flat = p.reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            plus = model.loss_and_grad(x, adj, target, edges=edges)[0]
            flat[i] = orig - eps
            minus = model.loss_and_grad(x, adj, target, edges=edges)[0]
            flat[i] = orig
            worst = max(worst, abs((plus - minus) / (2 * eps) - grads[name].reshape(-1)[i]))
    check("gradient with edge features vs finite differences", worst < 1e-7, f"max_abs = {worst:.2e}")


if __name__ == "__main__":
    test_gradient()
    test_edge_features_are_used()
    test_edge_gradient()
    test_permutation_invariance()
    test_node_states_are_equivariant()
    test_it_learns_a_graph_property()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
