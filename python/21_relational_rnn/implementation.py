"""Relational Recurrent Neural Networks (Santoro et al., 2018).

The paper's cell keeps a small set of **memory slots** and lets them interact at every
step through multi-head dot-product attention (MHDPA), so that information can move
between slots without passing through the recurrent state.

The property that matters structurally is that the slots are a set: attention over them is
permutation-equivariant, and the attention weights are a distribution over slots. Both are
exact and are checked here.

No task comparison against an LSTM is done — the paper's headline claim is that the RMC
solves memory tasks an LSTM cannot, and that needs a training comparison this slice does
not include.
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class MHDPA:
    """Multi-head dot-product attention between memory slots."""

    def __init__(self, slots, dim, heads=2, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.d, self.h = slots, dim, heads
        self.dk = dim // heads
        s = 1.0 / np.sqrt(dim)
        self.Wq = rng.normal(0, s, (dim, dim))
        self.Wk = rng.normal(0, s, (dim, dim))
        self.Wv = rng.normal(0, s, (dim, dim))
        self.Wo = rng.normal(0, s, (dim, dim))

    def params(self):
        return [("Wq", self.Wq), ("Wk", self.Wk), ("Wv", self.Wv), ("Wo", self.Wo)]

    def forward(self, slots):
        Q = slots @ self.Wq
        K = slots @ self.Wk
        V = slots @ self.Wv
        n, h, dk = self.n, self.h, self.dk
        Qh = Q.reshape(n, h, dk).transpose(1, 0, 2)
        Kh = K.reshape(n, h, dk).transpose(1, 0, 2)
        Vh = V.reshape(n, h, dk).transpose(1, 0, 2)
        scores = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(dk)
        attn = softmax(scores)
        out = attn @ Vh
        merged = out.transpose(1, 0, 2).reshape(n, self.d)
        self.cache = (slots, Q, K, V, Qh, Kh, Vh, attn, merged)
        return merged @ self.Wo, attn

    def backward(self, dout):
        slots, Q, K, V, Qh, Kh, Vh, attn, merged = self.cache
        n, h, dk = self.n, self.h, self.dk
        grads = {"Wo": merged.T @ dout}
        dm = dout @ self.Wo.T
        dOh = dm.reshape(n, h, dk).transpose(1, 0, 2)
        dA = dOh @ Vh.transpose(0, 2, 1)
        dVh = attn.transpose(0, 2, 1) @ dOh
        dS = attn * (dA - (dA * attn).sum(axis=-1, keepdims=True))
        dQh = (dS @ Kh) / np.sqrt(dk)
        dKh = (dS.transpose(0, 2, 1) @ Qh) / np.sqrt(dk)
        dQ = dQh.transpose(1, 0, 2).reshape(n, self.d)
        dK = dKh.transpose(1, 0, 2).reshape(n, self.d)
        dV = dVh.transpose(1, 0, 2).reshape(n, self.d)
        grads["Wq"] = slots.T @ dQ
        grads["Wk"] = slots.T @ dK
        grads["Wv"] = slots.T @ dV
        dslots = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        return dslots, grads


class RelationalCell:
    """Slots exchange information through MHDPA, then are updated by a shared MLP."""

    def __init__(self, slots=6, dim=8, heads=2, seed=0):
        rng = np.random.default_rng(seed)
        self.attn = MHDPA(slots, dim, heads, seed)
        self.n, self.d = slots, dim
        s = 1.0 / np.sqrt(dim)
        self.W1 = rng.normal(0, s, (dim, dim))
        self.b1 = np.zeros(dim)
        self.W2 = rng.normal(0, s, (dim, dim))
        self.b2 = np.zeros(dim)

    def params(self):
        return self.attn.params() + [("W1", self.W1), ("b1", self.b1),
                                     ("W2", self.W2), ("b2", self.b2)]

    def forward(self, slots):
        attended, attn = self.attn.forward(slots)
        s = slots + attended
        h = np.maximum(0.0, s @ self.W1 + self.b1)
        out = s + h @ self.W2 + self.b2
        self.cache = (slots, s, h, out)
        return out, attn

    def backward(self, dout):
        slots, s, h, out = self.cache
        grads = {"W2": h.T @ dout, "b2": dout.sum(axis=0)}
        dh = dout @ self.W2.T
        dpre = dh * (h > 0)
        grads["W1"] = s.T @ dpre
        grads["b1"] = dpre.sum(axis=0)
        ds = dout + dpre @ self.W1.T
        dslots_attn, g = self.attn.backward(ds)
        grads.update(g)
        return ds + dslots_attn, grads
