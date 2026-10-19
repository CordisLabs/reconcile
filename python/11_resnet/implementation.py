"""Deep Residual Learning for Image Recognition (He et al., 2015).

The paper's first, and most surprising, result is not about accuracy on ImageNet — it is
the *degradation problem*: adding layers to a plain network makes the **training** error
go up. That is not overfitting (training error is what rises), and it is not fixed by
more data. The paper's claim is that residual connections remove it:

    "… when the deeper model starts converging, the training error of the deeper network
     is higher than that of its shallower counterpart."

    "… our deeply residual networks can easily enjoy the benefits of increasing depth."

That is a statement about depth, not about convolution, so it is tested here with plain
and residual MLPs — the same stream of layers, with or without the identity path. The
convolutional setting of the paper is out of reach at this size and that is stated in
the README rather than papered over.
"""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)


class MLP:
    """Stack of 2-layer blocks; `residual=True` adds the identity path to each block."""

    def __init__(self, n_in, n_out, blocks, width=32, residual=False, seed=0):
        rng = np.random.default_rng(seed)
        self.blocks = blocks
        self.width = width
        self.residual = residual
        self.Win = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, width))
        self.bin = np.zeros(width)
        self.Ws = []
        self.bs = []
        for _ in range(blocks):
            s1 = np.sqrt(2.0 / width)
            s2 = np.sqrt(2.0 / width)
            self.Ws.append(
                (
                    rng.normal(0, s1, (width, width)),
                    rng.normal(0, s2, (width, width)),
                )
            )
            self.bs.append((np.zeros(width), np.zeros(width)))
        self.Wout = rng.normal(0, np.sqrt(2.0 / width), (width, n_out))
        self.bout = np.zeros(n_out)

    def params(self):
        out = [("Win", self.Win), ("bin", self.bin)]
        for i, (W1, W2) in enumerate(self.Ws):
            out.append((f"W1_{i}", W1))
            out.append((f"W2_{i}", W2))
            out.append((f"b1_{i}", self.bs[i][0]))
            out.append((f"b2_{i}", self.bs[i][1]))
        out.append(("Wout", self.Wout))
        out.append(("bout", self.bout))
        return out

    def forward(self, x):
        h0 = relu(x @ self.Win + self.bin)
        h = h0
        caches = []
        for i in range(self.blocks):
            W1, W2 = self.Ws[i]
            b1, b2 = self.bs[i]
            pre = h @ W1 + b1
            a = relu(pre)
            z = a @ W2 + b2
            out = h + z if self.residual else relu(z)
            caches.append((h, pre, a, z, out))
            h = out
        logits = h @ self.Wout + self.bout
        self.cache = (x, h0, caches, h)
        return logits

    def loss_and_grad(self, x, y):
        """Mean cross-entropy over the batch and gradients for every parameter."""
        logits = self.forward(x)
        B = x.shape[0]
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        loss = float(-np.log(p[np.arange(B), y] + 1e-12).mean())
        dlogits = p
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B

        grads = {}
        grads["Wout"] = self.cache[3].T @ dlogits
        grads["bout"] = dlogits.sum(axis=0)
        dh = dlogits @ self.Wout.T

        x, h0, caches, _ = self.cache
        for i in reversed(range(self.blocks)):
            h_prev, pre, a, z, out = caches[i]
            W1, W2 = self.Ws[i]
            if self.residual:
                dz = dh
                dh_prev = dh
            else:
                dz = dh * (out > 0)
                dh_prev = np.zeros_like(h_prev)
            grads[f"W2_{i}"] = a.T @ dz
            grads[f"b2_{i}"] = dz.sum(axis=0)
            da = dz @ W2.T
            dpre = da * (pre > 0)
            grads[f"W1_{i}"] = h_prev.T @ dpre
            grads[f"b1_{i}"] = dpre.sum(axis=0)
            dh_prev = dh_prev + dpre @ W1.T
            dh = dh_prev
        # the input projection is followed by ReLU too; that derivative was missing
        dh0 = dh * (h0 > 0)
        grads["Win"] = x.T @ dh0
        grads["bin"] = dh0.sum(axis=0)
        return loss, grads


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params}
        self.v = {k: np.zeros_like(v) for k, v in params}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k, v in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            v -= self.lr * mh / (np.sqrt(vh) + self.eps)


def make_task(n_classes=4, n_per_class=50, dim=10, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1.5, (n_classes, dim))
    xs, ys = [], []
    for c in range(n_classes):
        xs.append(centres[c] + rng.normal(0, 0.7, (n_per_class, dim)))
        ys.append(np.full(n_per_class, c))
    return np.concatenate(xs), np.concatenate(ys)


def accuracy(model, x, y):
    return float((model.forward(x).argmax(axis=1) == y).mean())


def train(model, x, y, steps, lr=1e-2, seed=0):
    opt = Adam(model.params(), lr=lr)
    losses = []
    for _ in range(steps):
        loss, grads = model.loss_and_grad(x, y)
        opt.step(model.params(), grads)
        losses.append(loss)
    return losses
