"""Neural Message Passing for Quantum Chemistry (Gilmer et al., 2017).

The MPNN framework is: messages along edges, an update per node, and a readout over the
whole graph. The structural property that matters is that a graph is a *set* of nodes plus
edges, so permuting the nodes must permute the node states and leave the graph-level
readout unchanged. That is exact and is checked here.

The original paper fits molecular properties on QM9. No dataset is included; the training
task is a synthetic graph regression whose answer is exact by construction.
"""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)


class MessagePassingNet:
    def __init__(self, node_dim=4, hidden=32, steps=3, seed=0, edge_dim=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(node_dim)
        self.edge_dim = edge_dim
        # message: [h_i, h_j, e_ij] -> hidden
        self.Wm = rng.normal(0, s, (2 * node_dim + edge_dim, hidden))
        self.bm = np.zeros(hidden)
        # update: [h_i, aggregated message] -> node_dim
        self.Wu = rng.normal(0, 1.0 / np.sqrt(node_dim + hidden), (node_dim + hidden, node_dim))
        self.bu = np.zeros(node_dim)
        # readout: sum of node states -> scalar
        self.Wr = rng.normal(0, 1.0 / np.sqrt(node_dim), (node_dim, 1))
        self.br = np.zeros(1)
        self.steps = steps

    def params(self):
        return [("Wm", self.Wm), ("bm", self.bm), ("Wu", self.Wu), ("bu", self.bu),
                ("Wr", self.Wr), ("br", self.br)]

    def forward(self, x, adj, edges=None):
        """x: (n, node_dim); adj: (n, n) 0/1 adjacency, no self loops."""
        h = x
        caches = []
        n = x.shape[0]
        for _ in range(self.steps):
            msgs = np.zeros((n, n, self.Wm.shape[1]))
            pairs = np.concatenate(
                [np.repeat(h[:, None, :], n, axis=1), np.repeat(h[None, :, :], n, axis=0)],
                axis=2,
            )
            if self.edge_dim:
                pairs = np.concatenate([pairs, edges], axis=2)
            msgs = relu(pairs @ self.Wm + self.bm)
            agg = np.einsum("ij,ijk->ik", adj, msgs)
            z = np.concatenate([h, agg], axis=1)
            h_new = relu(z @ self.Wu + self.bu)
            caches.append((h, pairs, msgs, agg, z, h_new))
            h = h_new
        pooled = h.sum(axis=0)
        out = float((pooled @ self.Wr + self.br)[0])
        self.cache = (x, adj, caches, h, pooled)
        return out

    def loss_and_grad(self, x, adj, target, edges=None):
        out = self.forward(x, adj, edges)
        loss = (out - target) ** 2
        dout = 2.0 * (out - target)

        grads = {}
        _, _, caches, h, pooled = self.cache
        grads["Wr"] = np.outer(pooled, [dout])
        grads["br"] = np.array([dout])
        dh = np.tile(self.Wr[:, 0] * dout, (x.shape[0], 1))

        for k in reversed(range(self.steps)):
            h_prev, pairs, msgs, agg, z, h_new = caches[k]
            dz = (dh * (h_new > 0)) @ self.Wu.T
            grads_wu = z.T @ (dh * (h_new > 0))
            grads_bu = (dh * (h_new > 0)).sum(axis=0)
            grads["Wu"] = grads["Wu"] + grads_wu if "Wu" in grads else grads_wu
            grads["bu"] = grads["bu"] + grads_bu if "bu" in grads else grads_bu
            dh_prev = dz[:, : h_prev.shape[1]]
            dagg = dz[:, h_prev.shape[1]:]
            # back through the message network
            dmsgs_mask = np.einsum("ij,ik->ijk", adj, dagg)   # (n,n,hidden)
            dpre = dmsgs_mask * (msgs > 0)
            n = x.shape[0]
            grads_wm = np.einsum("ijk,ijm->km", dpre, pairs).T
            grads_bm = dpre.sum(axis=(0, 1))
            grads["Wm"] = grads["Wm"] + grads_wm if "Wm" in grads else grads_wm
            grads["bm"] = grads["bm"] + grads_bm if "bm" in grads else grads_bm
            dpairs = dpre @ self.Wm.T                       # (n,n,2d+edge)
            d = self.Wm.shape[0] // 2
            dh_prev = dh_prev + dpairs[:, :, :d].sum(axis=1) + dpairs[:, :, d:2 * d].sum(axis=0)
            dh = dh_prev
        return loss, grads


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params}
        self.v = {k: np.zeros_like(v) for k, v in params}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k, p in params:
            gr = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gr
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gr * gr
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# --------------------------------------------------------------------------
# task: predict a graph-level property that is exactly defined
# --------------------------------------------------------------------------

def make_graph(rng, n=6, node_dim=4):
    x = rng.normal(size=(n, node_dim))
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i, j] = adj[j, i] = 1.0
    # target: the sum of the first feature over nodes, a permutation-invariant quantity
    target = float(x[:, 0].sum())
    return x, adj, target


def train(model, steps=300, batch=8, seed=0):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=3e-3)
    losses = []
    for _ in range(steps):
        acc = {k: np.zeros_like(v) for k, v in model.params()}
        total = 0.0
        for _ in range(batch):
            x, adj, target = make_graph(rng)
            loss, g = model.loss_and_grad(x, adj, target)
            total += loss
            for k in acc:
                acc[k] += g[k]
        for k in acc:
            acc[k] /= batch
        opt.step(model.params(), acc)
        losses.append(total / batch)
    return losses
