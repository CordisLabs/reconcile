"""Variational Lossy Autoencoder (Chen et al., 2016).

The paper's starting point is **posterior collapse**: if the decoder is expressive enough,
the model learns to ignore the latent entirely, and the KL term goes to zero. The paper
then adds a lossy code and an autoregressive prior/decoder to make the latent useful.

This slice implements the standard VAE and measures collapse directly: per-dimension KL,
the number of "active" dimensions, and how they respond to the KL weight beta. An IAF prior
is not implemented, so the paper's own fix is not tested — what is tested is the phenomenon
that motivates it.

Gaussian decoder, Gaussian latent, analytic KL. No sampling noise in the loss beyond the
reparameterisation, so the curves are smooth.
"""

import numpy as np


class VAE:
    def __init__(self, data_dim, hidden=32, latent=8, seed=0):
        rng = np.random.default_rng(seed)
        self.D, self.H, self.L = data_dim, hidden, latent
        s = 1.0 / np.sqrt(data_dim)
        self.We = rng.normal(0, s, (data_dim, hidden))
        self.be = np.zeros(hidden)
        self.Wmu = rng.normal(0, s, (hidden, latent))
        self.bmu = np.zeros(latent)
        self.Wlv = rng.normal(0, s, (hidden, latent))
        self.blv = np.zeros(latent)
        self.Wd = rng.normal(0, 1.0 / np.sqrt(latent), (latent, hidden))
        self.bd = np.zeros(hidden)
        self.Wo = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, data_dim))
        self.bo = np.zeros(data_dim)

    def params(self):
        return [("We", self.We), ("be", self.be), ("Wmu", self.Wmu), ("bmu", self.bmu),
                ("Wlv", self.Wlv), ("blv", self.blv), ("Wd", self.Wd), ("bd", self.bd),
                ("Wo", self.Wo), ("bo", self.bo)]

    def encode(self, x):
        h = np.tanh(x @ self.We + self.be)
        mu = h @ self.Wmu + self.bmu
        logvar = np.clip(h @ self.Wlv + self.blv, -6.0, 6.0)
        return h, mu, logvar

    def forward(self, x, rng):
        h, mu, logvar = self.encode(x)
        std = np.exp(0.5 * logvar)
        eps = rng.normal(size=mu.shape)
        z = mu + std * eps
        hd = np.tanh(z @ self.Wd + self.bd)
        recon = hd @ self.Wo + self.bo
        self.cache = (x, h, mu, logvar, std, eps, z, hd, recon)
        return recon, mu, logvar, z

    def loss_and_grad(self, x, beta, rng):
        """Gaussian reconstruction + beta * KL, gradients for every parameter."""
        recon, mu, logvar, z = self.forward(x, rng)
        _, h, mu_, logvar_, std, eps, _, hd, recon_ = self.cache
        B = x.shape[0]
        # reconstruction: mean squared error (Gaussian with unit variance)
        diff = recon - x
        rec_loss = float((diff ** 2).sum(axis=1).mean())
        # KL(q(z|x) || N(0,1)), analytically, per dimension
        kl_dim = 0.5 * (mu ** 2 + np.exp(logvar) - 1.0 - logvar)
        kl = float(kl_dim.sum(axis=1).mean())
        loss = rec_loss + beta * kl

        grads = {}
        drec = 2.0 * diff / B
        grads["Wo"] = hd.T @ drec
        grads["bo"] = drec.sum(axis=0)
        dhd = (drec @ self.Wo.T) * (1 - hd * hd)
        grads["Wd"] = z.T @ dhd
        grads["bd"] = dhd.sum(axis=0)
        dz = dhd @ self.Wd.T
        dmu = dz + beta * mu / B
        dlogvar = dz * std * eps * 0.5 + beta * 0.5 * (np.exp(logvar) - 1.0) / B
        grads["Wmu"] = h.T @ dmu
        grads["bmu"] = dmu.sum(axis=0)
        grads["Wlv"] = h.T @ dlogvar
        grads["blv"] = dlogvar.sum(axis=0)
        dh = (dmu @ self.Wmu.T + dlogvar @ self.Wlv.T) * (1 - h * h)
        grads["We"] = x.T @ dh
        grads["be"] = dh.sum(axis=0)
        return loss, grads, kl_dim.mean(axis=0)


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


def make_data(n=512, dim=6, seed=0):
    """Two clusters: a latent code is genuinely useful here."""
    rng = np.random.default_rng(seed)
    half = n // 2
    a = rng.normal(-1.0, 0.3, size=(half, dim))
    b = rng.normal(1.0, 0.3, size=(n - half, dim))
    return np.concatenate([a, b])


def train_vae(model, data, beta, steps=400, batch=64, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr=lr)
    for _ in range(steps):
        idx = rng.integers(0, data.shape[0], size=batch)
        x = data[idx]
        _, grads, _ = model.loss_and_grad(x, beta, rng)
        opt.step(model.params(), grads)
    rng2 = np.random.default_rng(seed + 1)
    recon, mu, logvar, _ = model.forward(data, rng2)
    rec = float(((recon - data) ** 2).sum(axis=1).mean())
    _, _, kl_dim = model.loss_and_grad(data, beta, rng2)
    return kl_dim, rec
