"""Understanding LSTM Networks (Olah, 2015) — LSTM cell from scratch in NumPy, plus a
vanilla RNN baseline trained on exactly the same task.

The blog post's claim is qualitative: the LSTM's gated cell state lets information
survive across many steps, which a vanilla RNN cannot do because its hidden state is
overwritten every step. To make that checkable, the slice uses the standard copy task
(Graves et al.): read T symbols, then reproduce them after a delay of T steps. The
answer is known exactly, so accuracy is a real number, not an impression.

Both models get the same data, the same hidden size, and the same number of updates.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# --------------------------------------------------------------------------
# copy task
# --------------------------------------------------------------------------

def copy_batch(batch, T, k, rng):
    """Read T symbols, then a delimiter, then reproduce them.

    Input ids: 0..k-1 symbols, k delimiter, k+1 blank. Sequence length 2T+1.
    Targets are defined only on the last T positions; `mask` marks them.
    """
    symbols = rng.integers(0, k, size=(batch, T))
    seq = np.full((batch, 2 * T + 1), k + 1, dtype=np.int64)  # blanks
    seq[:, :T] = symbols
    seq[:, T] = k
    targets = np.zeros((batch, 2 * T + 1), dtype=np.int64)
    targets[:, T + 1:] = symbols
    mask = np.zeros((2 * T + 1, batch))
    mask[T + 1:] = 1.0
    return seq, targets, mask


def one_hot(idx, size):
    out = np.zeros((idx.shape[0], idx.shape[1], size))
    for t in range(idx.shape[1]):
        out[:, t, :] = np.eye(size)[idx[:, t]]
    return out


def masked_ce(logits, targets, mask):
    """Cross-entropy averaged over the masked positions, in nats."""
    T, B, O = logits.shape
    p = softmax(logits)
    picked = p[np.arange(T)[:, None], np.arange(B)[None, :], targets.T]
    return float(-(np.log(picked) * mask).sum() / mask.sum())


# --------------------------------------------------------------------------
# LSTM
# --------------------------------------------------------------------------

class LSTM:
    """i,f,o = sigmoid(...), g = tanh(...), c = f*c_prev + i*g, h = o*tanh(c)."""

    def __init__(self, input_dim, hidden, output_dim, seed=0, forget_bias=1.0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hidden)
        self.I, self.H, self.O = input_dim, hidden, output_dim
        # Forget-gate bias: the cell must start out *keeping* its state. With a zero
        # bias the forget gate sits at sigmoid(0) = 0.5, so a state has to survive a
        # 0.5^T decay before any gradient can teach it otherwise; at T = 20 that is
        # 1e-6 and the task is unreachable in practice. This is the one line that
        # separates "the cell is correct" from "the cell can learn a long delay".
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

    def _gates(self, x, h):
        z = x @ self.params["Wx"] + h @ self.params["Wh"] + self.params["b"]
        H = self.H
        return (
            sigmoid(z[:, :H]),
            sigmoid(z[:, H:2 * H]),
            sigmoid(z[:, 2 * H:3 * H]),
            np.tanh(z[:, 3 * H:]),
        )

    def forward(self, x):
        """x: one-hot (B, T, I). Returns logits (T, B, O) and the final (h, c)."""
        p = self.params
        B, T, _ = x.shape
        h = np.zeros((B, self.H))
        c = np.zeros((B, self.H))
        hs = np.empty((T + 1, B, self.H))
        cs = np.empty((T + 1, B, self.H))
        gates = np.empty((T, B, 4 * self.H))
        logits = np.empty((T, B, self.O))
        hs[0], cs[0] = h, c
        for t in range(T):
            i, f, o, g = self._gates(x[:, t, :], h)
            c = f * c + i * g
            tc = np.tanh(c)
            h = o * tc
            hs[t + 1], cs[t + 1] = h, c
            gates[t] = np.concatenate([i, f, o, g], axis=1)
            logits[t] = h @ p["Wy"] + p["by"]
        self.cache = (x, hs, cs, gates, logits)
        return logits, h

    def loss(self, x, targets, mask):
        logits, _ = self.forward(x)
        return masked_ce(logits, targets, mask)

    def backward(self, targets, mask):
        x, hs, cs, gates, logits = self.cache
        T, B, O = logits.shape
        H = self.H

        p = softmax(logits)
        p[np.arange(T)[:, None], np.arange(B)[None, :], targets.T] -= 1.0
        p *= mask[:, :, None]
        p /= mask.sum()

        g = {k: np.zeros_like(v) for k, v in self.params.items()}
        g["Wy"] = np.einsum("tbh,tbo->ho", hs[1:], p)
        g["by"] = p.sum(axis=(0, 1))

        dh_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))
        for t in reversed(range(T)):
            dh = p[t] @ self.params["Wy"].T + dh_next
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
            g["Wx"] += np.einsum("bi,bj->ij", x[:, t, :], dz)
            g["Wh"] += hs[t].T @ dz
            g["b"] += dz.sum(axis=0)
            dh_next = dz @ self.params["Wh"].T
        # Gradient with respect to the initial hidden state: the quantity that shows
        # whether information survives across the sequence. Stored on the model so
        # tests can read it without changing the training interface.
        self.dh0 = dh_next
        return g


# --------------------------------------------------------------------------
# vanilla RNN baseline, same interface
# --------------------------------------------------------------------------

class VanillaRNN:
    """h_t = tanh(x_t W_x + h_{t-1} W_h + b), y_t = h_t W_y + b_y."""

    def __init__(self, input_dim, hidden, output_dim, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(hidden)
        self.I, self.H, self.O = input_dim, hidden, output_dim
        self.params = {
            "Wx": rng.uniform(-s, s, (input_dim, hidden)),
            "Wh": rng.uniform(-s, s, (hidden, hidden)),
            "b": np.zeros(hidden),
            "Wy": rng.uniform(-s, s, (hidden, output_dim)),
            "by": np.zeros(output_dim),
        }
        self.cache = None

    def forward(self, x):
        p = self.params
        B, T, _ = x.shape
        h = np.zeros((B, self.H))
        hs = np.empty((T + 1, B, self.H))
        logits = np.empty((T, B, self.O))
        hs[0] = h
        for t in range(T):
            h = np.tanh(x[:, t, :] @ p["Wx"] + h @ p["Wh"] + p["b"])
            hs[t + 1] = h
            logits[t] = h @ p["Wy"] + p["by"]
        self.cache = (x, hs, logits)
        return logits, h

    def loss(self, x, targets, mask):
        logits, _ = self.forward(x)
        return masked_ce(logits, targets, mask)

    def backward(self, targets, mask):
        x, hs, logits = self.cache
        T, B, O = logits.shape
        p = softmax(logits)
        p[np.arange(T)[:, None], np.arange(B)[None, :], targets.T] -= 1.0
        p *= mask[:, :, None]
        p /= mask.sum()

        g = {k: np.zeros_like(v) for k, v in self.params.items()}
        g["Wy"] = np.einsum("tbh,tbo->ho", hs[1:], p)
        g["by"] = p.sum(axis=(0, 1))
        dh_next = np.zeros((B, self.H))
        for t in reversed(range(T)):
            dh = p[t] @ self.params["Wy"].T + dh_next
            da = dh * (1.0 - hs[t + 1] ** 2)
            g["b"] += da.sum(axis=0)
            g["Wh"] += hs[t].T @ da
            g["Wx"] += np.einsum("bi,bh->ih", x[:, t, :], da)
            dh_next = da @ self.params["Wh"].T
        self.dh0 = dh_next
        return g


# --------------------------------------------------------------------------
# training / evaluation
# --------------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, clip=5.0):
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-12))
        self.t += 1
        for k in params:
            gr = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gr
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * gr * gr
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def accuracy(model, batch, T, k, rng):
    """Fraction of correctly copied symbols, averaged over the batch."""
    seq, targets, mask = copy_batch(batch, T, k, rng)
    x = one_hot(seq, k + 2)
    logits, _ = model.forward(x)
    pred = logits.argmax(axis=-1)                     # (T, B)
    want = targets.T                                  # (T, B)
    correct = ((pred == want).astype(float) * mask).sum() / mask.sum()
    return float(correct)


def train(model, steps, T, k, batch, lr=3e-3, seed=0, log_every=0):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params, lr=lr)
    trace = []
    for step in range(steps):
        seq, targets, mask = copy_batch(batch, T, k, rng)
        x = one_hot(seq, k + 2)
        loss = model.loss(x, targets, mask)
        grads = model.backward(targets, mask)
        opt.step(model.params, grads)
        trace.append(loss)
        if log_every and step % log_every == 0:
            print(f"  step {step:4d}  loss {loss:.4f} nats")
    return trace
