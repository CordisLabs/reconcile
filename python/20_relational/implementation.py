"""A Simple Neural Network Module for Relational Reasoning (Santoro et al., 2017).

The relation module is two functions:

    RN(O) = f_phi( sum_{i,j} g_theta(o_i, o_j) )

`g_theta` sees one pair at a time, the sum aggregates over all ordered pairs (including
i == j, whose diagonal terms carry the individual objects), and `f_phi` maps the
aggregate to an answer.

Two things are checked: the module is exactly invariant to the order of the objects — the
sum cannot see it — and it can actually learn a task that needs relations, namely
counting how many objects share a queried colour.

The task is a small stand-in for sort-of-CLEVR. It is generated, so the answer is exact.
"""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)


class RelationNet:
    def __init__(self, obj_dim=7, q_dim=3, hidden=32, n_answers=7, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(obj_dim * 2)
        # g_theta: pair -> hidden
        self.Wg1 = rng.normal(0, s, (2 * obj_dim, hidden))
        self.bg1 = np.zeros(hidden)
        self.Wg2 = rng.normal(0, s, (hidden, hidden))
        self.bg2 = np.zeros(hidden)
        # f_phi: (aggregate + question) -> answer
        self.Wf1 = rng.normal(0, 1.0 / np.sqrt(hidden + q_dim), (hidden + q_dim, hidden))
        self.bf1 = np.zeros(hidden)
        self.Wf2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, n_answers))
        self.bf2 = np.zeros(n_answers)

    def params(self):
        return [
            ("Wg1", self.Wg1), ("bg1", self.bg1), ("Wg2", self.Wg2), ("bg2", self.bg2),
            ("Wf1", self.Wf1), ("bf1", self.bf1), ("Wf2", self.Wf2), ("bf2", self.bf2),
        ]

    def forward(self, objects, question):
        """objects: (K, obj_dim); question: (q_dim,). Returns logits and a cache."""
        K = objects.shape[0]
        pairs = np.zeros((K, K, 2 * objects.shape[1]))
        for i in range(K):
            for j in range(K):
                pairs[i, j] = np.concatenate([objects[i], objects[j]])
        flat = pairs.reshape(-1, pairs.shape[-1])
        h1 = relu(flat @ self.Wg1 + self.bg1)
        g = relu(h1 @ self.Wg2 + self.bg2)
        agg = g.reshape(K, K, -1).sum(axis=(0, 1))
        z = np.concatenate([agg, question])
        h2 = relu(z @ self.Wf1 + self.bf1)
        logits = h2 @ self.Wf2 + self.bf2
        self.cache = (objects, pairs, flat, h1, g, agg, z, h2, logits, K)
        return logits

    def loss_and_grad(self, objects, question, answer):
        logits = self.forward(objects, question)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        loss = float(-np.log(p[answer] + 1e-12))
        dlogits = p.copy()
        dlogits[answer] -= 1.0

        (objects, pairs, flat, h1, g, agg, z, h2, _, K) = self.cache
        grads = {}
        grads["Wf2"] = np.outer(h2, dlogits)
        grads["bf2"] = dlogits
        dh2 = dlogits @ self.Wf2.T
        dz = (dh2 * (h2 > 0)) @ self.Wf1.T
        grads["Wf1"] = np.outer(z, dh2 * (h2 > 0))
        grads["bf1"] = dh2 * (h2 > 0)
        dagg = dz[: agg.shape[0]]

        # the sum over pairs routes the same gradient to every pair
        dg = np.zeros_like(g)
        for i in range(K):
            for j in range(K):
                dg[i * K + j] = dagg
        da2 = dg * (g > 0)
        grads["Wg2"] = h1.T @ da2
        grads["bg2"] = da2.sum(axis=0)
        da1 = (da2 @ self.Wg2.T) * (h1 > 0)
        grads["Wg1"] = flat.T @ da1
        grads["bg1"] = da1.sum(axis=0)
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
# task: count the objects of a queried colour
# --------------------------------------------------------------------------

N_COLOURS = 3
N_SHAPES = 2
N_ANSWERS = 7


def make_example(rng, K=6):
    colours = rng.integers(0, N_COLOURS, size=K)
    shapes = rng.integers(0, N_SHAPES, size=K)
    xy = rng.random((K, 2))
    objects = np.zeros((K, 7))
    objects[:, :2] = xy
    objects[np.arange(K), 2 + colours] = 1.0
    objects[np.arange(K), 5 + shapes] = 1.0
    q = int(rng.integers(0, N_COLOURS))
    question = np.zeros(3)
    question[q] = 1.0
    return objects, question, int((colours == q).sum())


def train(model, steps, batch=32, K=6, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    losses = []
    for _ in range(steps):
        acc = {k: np.zeros_like(v) for k, v in model.params()}
        total = 0.0
        for _ in range(batch):
            objects, question, answer = make_example(rng, K)
            loss, g = model.loss_and_grad(objects, question, answer)
            total += loss
            for k in acc:
                acc[k] += g[k]
        for k in acc:
            acc[k] /= batch
        opt.step(model.params(), acc)
        losses.append(total / batch)
    return losses


def build_test_set(samples=200, K=6, seed=1):
    """A fixed set of examples, so two evaluations see exactly the same data."""
    rng = np.random.default_rng(seed)
    return [make_example(rng, K) for _ in range(samples)]


def accuracy(model, test_set, permute_seed=None):
    """Accuracy on a fixed test set; `permute_seed` shuffles each example's objects.

    Shuffling uses a separate generator so the examples themselves are unchanged —
    otherwise the two measurements would be over different data.
    """
    prng = np.random.default_rng(permute_seed) if permute_seed is not None else None
    correct = 0
    for objects, question, answer in test_set:
        if prng is not None:
            objects = objects[prng.permutation(objects.shape[0])]
        logits = model.forward(objects, question)
        if int(np.argmax(logits)) == answer:
            correct += 1
    return correct / len(test_set)
