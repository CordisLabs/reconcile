"""Identity Mappings in Deep Residual Networks (He et al., 2016).

ResNet v1 puts the activation after the addition:  h_{l+1} = relu(h_l + F(h_l)).
v2 keeps the shortcut path clean:                    h_{l+1} = h_l + F(h_l),
with the batch-norm and activation moved *before* the convolution ("pre-activation").

The paper's argument is about the backward path: if the shortcut is an identity, then
dL/dh_l = dL/dh_{l+1} (1 + dF/dh_l), so the gradient reaches the earlier layers without
being multiplied by an activation derivative at every step. The checkable quantity is the
norm of the gradient reaching the input — it needs no training and no dataset.

Batch normalisation is not implemented here; the paper's analysis has it, and its absence
is part of why this is a mechanism check rather than a reproduction.
"""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def bn_forward(x, gamma, beta, eps=1e-5):
    """Batch normalisation over the batch dimension, in training mode."""
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv
    return gamma * xhat + beta, (x, xhat, inv, gamma, eps)


def bn_backward(dy, cache):
    """Standard BN backward; returns dx, dgamma, dbeta."""
    x, xhat, inv, gamma, _ = cache
    n = x.shape[0]
    dgamma = (dy * xhat).sum(axis=0)
    dbeta = dy.sum(axis=0)
    dxhat = dy * gamma
    dx = (inv / n) * (n * dxhat - dxhat.sum(axis=0) - xhat * (dxhat * xhat).sum(axis=0))
    return dx, dgamma, dbeta


class ResMLP:
    """`variant`:
       "pre"  -> h = h + W2 relu(W1 h)          (identity shortcut, activation inside)
       "post" -> h = relu(h + W2 relu(W1 h))    (activation on the shortcut)
    """

    def __init__(self, n_in, n_out, blocks, width=32, variant="pre", seed=0, bn=False,
                 identity_init=False):
        if variant not in ("pre", "post"):
            raise ValueError(variant)
        self.bn = bn
        rng = np.random.default_rng(seed)
        self.blocks = blocks
        self.variant = variant
        self.Win = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, width))
        self.bin = np.zeros(width)
        self.W1 = [rng.normal(0, np.sqrt(2.0 / width), (width, width)) for _ in range(blocks)]
        self.W2 = [rng.normal(0, np.sqrt(2.0 / width), (width, width)) for _ in range(blocks)]
        self.b1 = [np.zeros(width) for _ in range(blocks)]
        self.b2 = [np.zeros(width) for _ in range(blocks)]
        if identity_init:
            # the paper's eq. 4: the residual branch starts as zero, so every block is an
            # identity at initialisation and the deep net starts equivalent to a shallow one
            for i in range(blocks):
                self.W2[i] = np.zeros_like(self.W2[i])
        self.g1 = [np.ones(width) for _ in range(blocks)]
        self.be1 = [np.zeros(width) for _ in range(blocks)]
        self.g2 = [np.ones(width) for _ in range(blocks)]
        self.be2 = [np.zeros(width) for _ in range(blocks)]
        self.Wout = rng.normal(0, np.sqrt(2.0 / width), (width, n_out))
        self.bout = np.zeros(n_out)

    def params(self):
        out = [("Win", self.Win), ("bin", self.bin), ("Wout", self.Wout), ("bout", self.bout)]
        for i in range(self.blocks):
            out += [(f"W1_{i}", self.W1[i]), (f"W2_{i}", self.W2[i]),
                    (f"b1_{i}", self.b1[i]), (f"b2_{i}", self.b2[i])]
            if self.bn:
                out += [(f"g1_{i}", self.g1[i]), (f"be1_{i}", self.be1[i]),
                        (f"g2_{i}", self.g2[i]), (f"be2_{i}", self.be2[i])]
        return out

    def forward(self, x):
        h0 = relu(x @ self.Win + self.bin)
        h = h0
        caches = []
        for i in range(self.blocks):
            pre = h @ self.W1[i] + self.b1[i]
            if self.bn:
                n1, c1 = bn_forward(pre, self.g1[i], self.be1[i])
                a = relu(n1)
                z = a @ self.W2[i] + self.b2[i]
                n2, c2 = bn_forward(z, self.g2[i], self.be2[i])
            else:
                n1 = pre
                c1 = c2 = None
                a = relu(pre)
                z = a @ self.W2[i] + self.b2[i]
                n2 = z
            s = h + n2
            out = s if self.variant == "pre" else relu(s)
            caches.append((h, pre, n1, a, z, n2, c1, c2, s, out))
            h = out
        logits = h @ self.Wout + self.bout
        self.cache = (x, h0, caches, h, logits)
        return logits

    def loss_and_grad(self, x, y):
        logits = self.forward(x)
        B = x.shape[0]
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        loss = float(-np.log(p[np.arange(B), y] + 1e-12).mean())
        dlogits = p
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B

        grads = {"Wout": self.cache[3].T @ dlogits, "bout": dlogits.sum(axis=0)}
        dh = dlogits @ self.Wout.T
        x_, h0, caches, _, _ = self.cache
        for i in reversed(range(self.blocks)):
            h_prev, pre, n1, a, z, n2, c1, c2, s, out = caches[i]
            if self.variant == "pre":
                ds = dh
                dh_prev = dh
            else:
                ds = dh * (s > 0)
                dh_prev = ds.copy()
            if self.bn:
                dz, dg2, dbe2 = bn_backward(ds, c2)
                grads[f"g2_{i}"] = dg2
                grads[f"be2_{i}"] = dbe2
                grads[f"W2_{i}"] = a.T @ dz
                grads[f"b2_{i}"] = dz.sum(axis=0)
                da = dz @ self.W2[i].T
                dpre_raw = da * (n1 > 0)
                dpre, dg1, dbe1 = bn_backward(dpre_raw, c1)
                grads[f"g1_{i}"] = dg1
                grads[f"be1_{i}"] = dbe1
            else:
                grads[f"W2_{i}"] = a.T @ ds
                grads[f"b2_{i}"] = ds.sum(axis=0)
                da = ds @ self.W2[i].T
                dpre = da * (pre > 0)
            grads[f"W1_{i}"] = h_prev.T @ dpre
            grads[f"b1_{i}"] = dpre.sum(axis=0)
            dh = dh_prev + dpre @ self.W1[i].T
        dh0 = dh * (h0 > 0)
        grads["Win"] = x_.T @ dh0
        grads["bin"] = dh0.sum(axis=0)
        self.dh0 = dh0
        return loss, grads


def input_gradient_norm(variant, blocks, width=32, seed=0, batch=8, n_in=10, n_out=10, bn=False):
    """Norm of dL/dx — how much signal survives the trip back to the input."""
    rng = np.random.default_rng(123)
    x = rng.normal(size=(batch, n_in))
    y = rng.integers(0, n_out, size=batch)
    model = ResMLP(n_in, n_out, blocks=blocks, width=width, variant=variant, seed=seed, bn=bn)
    model.loss_and_grad(x, y)
    return float(np.linalg.norm(model.dh0))
