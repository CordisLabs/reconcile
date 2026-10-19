"""Recurrent Neural Network Regularization (Zaremba et al., 2014).

The paper's central claim is narrow and therefore testable: dropout belongs on the
*non-recurrent* connections. Applying it to the recurrent connection (the weight that
carries ``h_{t-1}``) destroys the memory the cell exists to provide.

This module implements an LSTM with three independent dropout switches:
  p_in   — on the input to the cell,
  p_out  — on the hidden state before the output projection,
  p_rec  — on the recurrent connection (the variant the paper argues against).

Masks are sampled once per sequence and reused at every timestep, which is the
"variational" dropout the paper describes: dropping a different unit at each step is
not a variational bound and behaves differently.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def drop_mask(rng, shape, p):
    """Inverted-dropout mask: 0 with probability p, else 1/(1-p)."""
    if p <= 0.0:
        return np.ones(shape)
    if p >= 1.0:
        return np.zeros(shape)
    keep = rng.random(shape) >= p
    return keep / (1.0 - p)


class DropoutLSTM:
    def __init__(self, input_dim, hidden, output_dim, p_in=0.0, p_out=0.0, p_rec=0.0,
                 seed=0, forget_bias=1.0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hidden)
        self.I, self.H, self.O = input_dim, hidden, output_dim
        self.p_in, self.p_out, self.p_rec = p_in, p_out, p_rec
        bias = np.zeros(4 * hidden)
        bias[hidden:2 * hidden] = forget_bias
        self.params = {
            "Wx": rng.uniform(-s, s, (input_dim, 4 * hidden)),
            "Wh": rng.uniform(-s, s, (hidden, 4 * hidden)),
            "b": bias,
            "Wy": rng.uniform(-s, s, (hidden, output_dim)),
            "by": np.zeros(output_dim),
        }
        self.cache = None
        self.dh0 = None

    def forward(self, x, rng=None, masks=None):
        """x: one-hot (B, T, I). Samples masks once per sequence when rng is given,
        or uses `masks = (m_in, m_out, m_rec)` when supplied (needed for gradient
        checks, where the same masks must be used by every finite-difference probe).
        """
        p = self.params
        B, T, _ = x.shape
        if masks is not None:
            m_in, m_out, m_rec = masks
        elif rng is not None:
            m_in = drop_mask(rng, (B, self.I), self.p_in)
            m_out = drop_mask(rng, (B, self.H), self.p_out)
            m_rec = drop_mask(rng, (B, self.H), self.p_rec)
        else:
            m_in = np.ones((B, self.I))
            m_out = np.ones((B, self.H))
            m_rec = np.ones((B, self.H))

        h = np.zeros((B, self.H))
        c = np.zeros((B, self.H))
        hs = np.empty((T + 1, B, self.H))
        cs = np.empty((T + 1, B, self.H))
        gates = np.empty((T, B, 4 * self.H))
        xd = np.empty((T, B, self.I))
        hd = np.empty((T, B, self.H))
        logits = np.empty((T, B, self.O))
        hs[0], cs[0] = h, c
        for t in range(T):
            xd[t] = x[:, t, :] * m_in
            z = xd[t] @ p["Wx"] + (h * m_rec) @ p["Wh"] + p["b"]
            H = self.H
            i = sigmoid(z[:, :H])
            f = sigmoid(z[:, H:2 * H])
            o = sigmoid(z[:, 2 * H:3 * H])
            g = np.tanh(z[:, 3 * H:])
            c = f * c + i * g
            h = o * np.tanh(c)
            hs[t + 1], cs[t + 1] = h, c
            gates[t] = np.concatenate([i, f, o, g], axis=1)
            hd[t] = h * m_out
            logits[t] = hd[t] @ p["Wy"] + p["by"]
        self.cache = (x, xd, hd, hs, cs, gates, logits, m_in, m_out, m_rec)
        return logits, h

    def loss(self, x, targets, rng=None, masks=None):
        logits, _ = self.forward(x, rng=rng, masks=masks)
        return masked_ce(logits, targets)

    def backward(self, targets):
        x, xd, hd, hs, cs, gates, logits, m_in, m_out, m_rec = self.cache
        T, B, O = logits.shape
        H = self.H

        p = softmax(logits)
        p[np.arange(T)[:, None], np.arange(B)[None, :], targets.T] -= 1.0
        p /= T * B

        g = {k: np.zeros_like(v) for k, v in self.params.items()}
        g["Wy"] = np.einsum("tbh,tbo->ho", hd, p)
        g["by"] = p.sum(axis=(0, 1))

        dh_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))
        for t in reversed(range(T)):
            dh = (p[t] @ self.params["Wy"].T) * m_out + dh_next
            i = gates[t][:, :H]
            f = gates[t][:, H:2 * H]
            o = gates[t][:, 2 * H:3 * H]
            gg = gates[t][:, 3 * H:]
            tc = np.tanh(cs[t + 1])

            dc = dh * o * (1.0 - tc * tc) + dc_next
            di = dc * gg
            dff = dc * cs[t]
            do = dh * tc
            dgg = dc * i
            dc_next = dc * f

            dz = np.concatenate(
                [
                    di * i * (1.0 - i),
                    dff * f * (1.0 - f),
                    do * o * (1.0 - o),
                    dgg * (1.0 - gg * gg),
                ],
                axis=1,
            )
            g["Wx"] += np.einsum("bi,bj->ij", xd[t], dz)
            g["Wh"] += (hs[t] * m_rec).T @ dz
            g["b"] += dz.sum(axis=0)
            dh_next = (dz @ self.params["Wh"].T) * m_rec
        self.dh0 = dh_next
        return g


def masked_ce(logits, targets):
    T, B, O = logits.shape
    p = softmax(logits)
    picked = p[np.arange(T)[:, None], np.arange(B)[None, :], targets.T]
    return float(-np.log(picked).mean())


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, clip=5.0):
        norm = np.sqrt(sum(float((gr * gr).sum()) for gr in grads.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for k in params:
            gr = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gr
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gr * gr
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# --------------------------------------------------------------------------
# language-model plumbing
# --------------------------------------------------------------------------

def encode(text):
    chars = sorted(set(text))
    return chars, {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}


def one_hot_ids(ids, size):
    out = np.zeros((ids.size, size))
    out[np.arange(ids.size), ids] = 1.0
    return out


def sample_batch(data, seq_len, batch_size, rng):
    starts = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[s:s + seq_len] for s in starts])
    y = np.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def bits_per_char(model, data, seq_len=32, rng=None, seed=0):
    """Mean cross-entropy on whole non-overlapping windows, in bits per character."""
    rng = np.random.default_rng(seed) if rng is None else rng
    losses = []
    for s in range(0, len(data) - seq_len - 1, seq_len):
        x = data[s:s + seq_len][None, :]
        y = data[s + 1:s + seq_len + 1][None, :]
        xo = np.stack([one_hot_ids(row, model.I) for row in x])
        losses.append(model.loss(xo, y, rng=None))
    return float(np.mean(losses)) / np.log(2)


def train(model, train_data, val_data, steps, seq_len=32, batch_size=16, lr=3e-3,
          seed=0, eval_every=0):
    """Returns (best val bits/char, trace of (step, train bpc, val bpc))."""
    rng = np.random.default_rng(seed)
    opt = Adam(model.params, lr=lr)
    best = float("inf")
    trace = []
    for step in range(steps):
        ids, targets = sample_batch(train_data, seq_len, batch_size, rng)
        x = np.stack([one_hot_ids(row, model.I) for row in ids])
        model.loss(x, targets, rng=rng)
        grads = model.backward(targets)
        opt.step(model.params, grads)
        if eval_every and (step % eval_every == 0 or step == steps - 1):
            tr = float(np.mean([
                model.loss(
                    np.stack([one_hot_ids(r, model.I) for r in ids[s:s + 1]]),
                    targets[s:s + 1],
                    rng=None,
                )
                for s in range(0, batch_size, max(1, batch_size // 4))
            ])) / np.log(2)
            vb = bits_per_char(model, val_data, seq_len=seq_len, seed=99)
            best = min(best, vb)
            trace.append((step, tr, vb))
    return best, trace
