"""The Unreasonable Effectiveness of Recurrent Neural Networks (Karpathy, 2015) —
a character-level RNN trained from scratch in NumPy.

Written against the blog post's description and the standard BPTT derivation. No
framework: forward and backward are both explicit, so the gradient can be checked
against finite differences.

The claim under test is narrow and therefore checkable: a character-level RNN learns
the statistics of a small corpus well enough to beat the order-0 entropy of the same
text — that is, it uses context, not just character frequency.
"""

import numpy as np


def encode(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return chars, stoi, itos


def order0_entropy(text):
    """Entropy in bits per character of the text's own character distribution."""
    counts = np.array([text.count(c) for c in sorted(set(text))], dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class CharRNN:
    """h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h),  y_t = W_hy h_t + b_y."""

    def __init__(self, vocab, hidden, seed=0):
        rng = np.random.default_rng(seed)
        self.V = vocab
        self.H = hidden
        s = 1.0 / np.sqrt(hidden)
        self.params = {
            "Wxh": rng.uniform(-s, s, (vocab, hidden)),
            "Whh": rng.uniform(-s, s, (hidden, hidden)),
            "Why": rng.uniform(-s, s, (hidden, vocab)),
            "bh": np.zeros(hidden),
            "by": np.zeros(vocab),
        }
        self.cache = None

    # -- forward ---------------------------------------------------------
    def forward(self, idx, h0=None):
        """idx: (B, T) int array. Returns logits (T, B, V) and the final state."""
        p = self.params
        B, T = idx.shape
        h = np.zeros((B, self.H)) if h0 is None else h0
        hs = np.empty((T + 1, B, self.H))
        hs[0] = h
        logits = np.empty((T, B, self.V))
        for t in range(T):
            h = np.tanh(p["Wxh"][idx[:, t]] + h @ p["Whh"].T + p["bh"])
            hs[t + 1] = h
            logits[t] = h @ p["Why"] + p["by"]
        self.cache = (idx, hs, logits)
        return logits, h

    def loss(self, idx, targets):
        logits, _ = self.forward(idx)
        return self._ce(logits, targets)

    @staticmethod
    def _ce(logits, targets):
        T, B, V = logits.shape
        p = softmax(logits)
        picked = p[np.arange(T)[:, None], np.arange(B)[None, :], targets.T]
        return float(-np.log(picked).mean())

    # -- backward --------------------------------------------------------
    def backward(self, idx, targets):
        """Analytic BPTT. Returns a dict of gradients shaped like `params`."""
        idx, hs, logits = self.cache
        T, B, V = logits.shape
        p = softmax(logits)
        p[np.arange(T)[:, None], np.arange(B)[None, :], targets.T] -= 1.0
        p /= T * B

        g = {k: np.zeros_like(v) for k, v in self.params.items()}
        g["Why"] = np.einsum("tbv,tbh->hv", p, hs[1:])
        g["by"] = p.sum(axis=(0, 1))

        dh_next = np.zeros((B, self.H))
        for t in reversed(range(T)):
            dh = p[t] @ self.params["Why"].T + dh_next
            da = dh * (1.0 - hs[t + 1] ** 2)
            g["bh"] += da.sum(axis=0)
            g["Whh"] += da.T @ hs[t]
            np.add.at(g["Wxh"], idx[:, t], da)
            dh_next = da @ self.params["Whh"]
        return g

    # -- sampling --------------------------------------------------------
    def sample(self, stoi, itos, seed_text, length, rng, temperature=1.0):
        h = np.zeros((1, self.H))
        out = []
        for ch in seed_text:
            h = np.tanh(self.params["Wxh"][[stoi[ch]]] + h @ self.params["Whh"].T + self.params["bh"])
        for _ in range(length):
            logits = h @ self.params["Why"] + self.params["by"]
            p = softmax(logits / temperature)[0]
            i = int(rng.choice(len(p), p=p))
            out.append(itos[i])
            h = np.tanh(self.params["Wxh"][[i]] + h @ self.params["Whh"].T + self.params["bh"])
        return "".join(out)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, clip=5.0):
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for k in params:
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def batches(data, seq_len, batch_size, rng):
    """Yield (inputs, targets) pairs of shape (batch, seq_len)."""
    n = len(data) - 1
    per_batch = seq_len * batch_size
    n_batches = n // per_batch
    for _ in range(n_batches):
        starts = rng.integers(0, n - seq_len, size=batch_size)
        x = np.stack([data[s:s + seq_len] for s in starts])
        y = np.stack([data[s + 1:s + seq_len + 1] for s in starts])
        yield x, y


def train(model, data, steps, seq_len=48, batch_size=16, lr=3e-3, seed=0, log_every=0):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params, lr=lr)
    it = batches(data, seq_len, batch_size, rng)
    trace = []
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = batches(data, seq_len, batch_size, rng)
            x, y = next(it)
        loss = model.loss(x, y)
        grads = model.backward(x, y)
        opt.step(model.params, grads)
        trace.append(loss)
        if log_every and step % log_every == 0:
            print(f"step {step:5d}  loss {loss:.4f}  bits/char {loss / np.log(2):.4f}")
    return trace
