"""Attention Is All You Need (Vaswani et al., 2017) — scaled dot-product and
multi-head attention, forward pass and hand-written backward pass.

Deliberately written against the paper's equations (3.2.1, 3.2.2) rather than any
existing implementation. Forward and backward are both explicit: the point of this
slice is that the analytic gradient can be checked against finite differences.

Shapes: X is (T, d_model), no batch dimension. Positional encoding and masking
belong to the "full Transformer" slice (17_annotated_transformer); this slice is
only the attention block.
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sdpa_forward(Q, K, V):
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V  — paper eq. (1)."""
    d_k = Q.shape[-1]
    scale = 1.0 / np.sqrt(d_k)
    scores = (Q @ K.T) * scale
    P = softmax(scores)
    O = P @ V
    return O, (Q, K, V, P, scale)


def sdpa_backward(dO, cache):
    Q, K, V, P, scale = cache
    dV = P.T @ dO
    dP = dO @ V.T
    # softmax Jacobian applied to dP
    dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True))
    dQ = (dS @ K) * scale
    dK = (dS.T @ Q) * scale
    return dQ, dK, dV


class MultiHeadAttention:
    """h heads of dimension d_k = d_model / h, concatenated and projected (eq. 2)."""

    def __init__(self, d_model, n_heads, seed=0):
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        rng = np.random.default_rng(seed)
        bound = 1.0 / np.sqrt(d_model)
        self.params = {}
        for name in ("Wq", "Wk", "Wv", "Wo"):
            self.params[name] = rng.uniform(-bound, bound, (d_model, d_model))
        for name in ("bq", "bk", "bv", "bo"):
            self.params[name] = np.zeros(d_model)
        self.cache = None

    # -- helpers ---------------------------------------------------------
    def _split(self, x):
        T = x.shape[0]
        return x.reshape(T, self.n_heads, self.d_k).transpose(1, 0, 2)

    def _merge(self, x):
        h, T, d_k = x.shape
        return x.transpose(1, 0, 2).reshape(T, h * d_k)

    # -- forward ---------------------------------------------------------
    def forward(self, X):
        p = self.params
        Q = X @ p["Wq"] + p["bq"]
        K = X @ p["Wk"] + p["bk"]
        V = X @ p["Wv"] + p["bv"]

        Qh, Kh, Vh = self._split(Q), self._split(K), self._split(V)
        scale = 1.0 / np.sqrt(self.d_k)
        scores = (Qh @ Kh.transpose(0, 2, 1)) * scale          # (h, T, T)
        P = softmax(scores, axis=-1)
        Oh = P @ Vh                                             # (h, T, d_k)
        O = self._merge(Oh)                                     # (T, d_model)
        Y = O @ p["Wo"] + p["bo"]

        self.cache = (X, Q, K, V, Qh, Kh, Vh, P, O, Oh, scale)
        return Y

    # -- backward --------------------------------------------------------
    def backward(self, dY):
        X, Q, K, V, Qh, Kh, Vh, P, O, Oh, scale = self.cache
        p = self.params
        grads = {}

        grads["Wo"] = O.T @ dY
        grads["bo"] = dY.sum(axis=0)
        dO = dY @ p["Wo"].T

        dOh = self._split(dO)                                   # (h, T, d_k)
        dP = dOh @ Vh.transpose(0, 2, 1)
        dVh = P.transpose(0, 2, 1) @ dOh
        dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True))
        dQh = (dS @ Kh) * scale
        dKh = (dS.transpose(0, 2, 1) @ Qh) * scale

        dQ = self._merge(dQh)
        dK = self._merge(dKh)
        dV = self._merge(dVh)

        grads["Wq"] = X.T @ dQ
        grads["bq"] = dQ.sum(axis=0)
        grads["Wk"] = X.T @ dK
        grads["bk"] = dK.sum(axis=0)
        grads["Wv"] = X.T @ dV
        grads["bv"] = dV.sum(axis=0)

        return grads
