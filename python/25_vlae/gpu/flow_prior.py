"""Does an autoregressive prior buy anything? Diagonal Gaussian against a flow.

The VLAE paper's contribution is a latent whose prior and decoder are autoregressive. The
earlier slice only measured posterior collapse — the problem, not the fix. This replaces
the diagonal Gaussian prior with a masked autoregressive flow (MADE) and compares test
bits per dimension, which is the quantity the paper reports.

The decoder, encoder, latent size and optimiser are identical; only the prior changes.

Run: ~/.venvs/gpu/bin/python flow_prior.py --epochs 3
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LATENT = 16


class MADE(nn.Module):
    """Masked autoencoder: outputs per-dimension scale and shift, respecting order."""

    def __init__(self, dim, hidden=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.deg_in = torch.argsort(torch.rand(dim, generator=g))
        self.deg_h = torch.randint(0, dim - 1, (hidden,), generator=g)
        m1 = (self.deg_in.unsqueeze(1) <= self.deg_h.unsqueeze(0)).float()
        # output j may depend on hidden units whose degree is strictly below j
        deg_out = torch.arange(dim).repeat(2)
        m2 = (self.deg_h.unsqueeze(1) < deg_out.unsqueeze(0)).float()
        self.register_buffer("m1", m1)
        self.register_buffer("m2", m2)
        self.W1 = nn.Parameter(torch.randn(dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.randn(hidden, 2 * dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(2 * dim))

    def forward(self, z):
        h = F.relu(z @ (self.W1 * self.m1) + self.b1)
        out = h @ (self.W2 * self.m2) + self.b2
        s, m = out.chunk(2, dim=1)
        return s.clamp(-3.0, 3.0), m


class FlowPrior(nn.Module):
    """z = z0 * exp(s(z0)) + m(z0), with a standard normal base density."""

    def __init__(self, dim, steps=2):
        super().__init__()
        self.flows = nn.ModuleList([MADE(dim, seed=i) for i in range(steps)])

    def log_prob(self, z):
        log_det = 0.0
        for fl in self.flows:
            s, m = fl(z)
            z = z * torch.exp(s) + m
            log_det = log_det + s.sum(dim=1)
        base = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)
        # change of variables: p(x) = p_z(f(x)) * |det J|, and det J = exp(sum s)
        return base + log_det


class Encoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.mu = nn.Linear(64 * 7 * 7, latent)
        self.lv = nn.Linear(64 * 7 * 7, latent)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.lv(h).clamp(-6, 6)


class Decoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.fc = nn.Linear(latent, 64 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.net(h)          # logits over pixels


def bits_per_dim(log_px, n_pixels):
    return -log_px / (n_pixels * np.log(2))


def loaders(batch=128):
    tf = torchvision.transforms.ToTensor()
    train = torchvision.datasets.MNIST(DATA, train=True, download=True, transform=tf)
    test = torchvision.datasets.MNIST(DATA, train=False, download=True, transform=tf)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=2),
        torch.utils.data.DataLoader(test, batch_size=512, shuffle=False, num_workers=2),
    )


def train(use_flow, epochs, device, seed=0, lr=1e-3):
    torch.manual_seed(seed)
    enc, dec = Encoder().to(device), Decoder().to(device)
    prior = FlowPrior(LATENT).to(device) if use_flow else None
    params = list(enc.parameters()) + list(dec.parameters())
    if prior is not None:
        params += list(prior.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    train_loader, test_loader = loaders()
    stats = []
    for ep in range(epochs):
        for x, _ in train_loader:
            x = x.to(device)
            mu, lv = enc(x)
            std = torch.exp(0.5 * lv)
            z = mu + std * torch.randn_like(std)
            logits = dec(z)
            log_px = -F.binary_cross_entropy_with_logits(
                logits, x, reduction="none").sum(dim=(1, 2, 3))
            if prior is None:
                log_q = -0.5 * ((z - mu) / std).pow(2).sum(1) - lv.sum(1) - 0.5 * LATENT * np.log(2 * np.pi)
                log_p = -0.5 * (z ** 2).sum(1) - 0.5 * LATENT * np.log(2 * np.pi)
            else:
                log_q = -0.5 * ((z - mu) / std).pow(2).sum(1) - lv.sum(1) - 0.5 * LATENT * np.log(2 * np.pi)
                log_p = prior.log_prob(z)
            loss = -(log_px + log_p - log_q).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        bpd = evaluate(enc, dec, prior, test_loader, device)
        stats.append((ep + 1, bpd))
        print(f"  epoch {ep + 1}: test bits/dim {bpd:.4f}", flush=True)
    return stats


@torch.no_grad()
def evaluate(enc, dec, prior, loader, device, batches=8):
    enc.eval(); dec.eval()
    total = n = 0
    for i, (x, _) in enumerate(loader):
        if i >= batches:
            break
        x = x.to(device)
        mu, lv = enc(x)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        logits = dec(z)
        log_px = -F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=(1, 2, 3))
        log_q = -0.5 * ((z - mu) / std).pow(2).sum(1) - lv.sum(1) - 0.5 * LATENT * np.log(2 * np.pi)
        log_p = prior.log_prob(z) if prior is not None else -0.5 * (z ** 2).sum(1) - 0.5 * LATENT * np.log(2 * np.pi)
        total += bits_per_dim(log_px + log_p - log_q, 28 * 28).sum().item()
        n += x.shape[0]
    enc.train(); dec.train()
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    out = {}
    for name, use_flow in (("gaussian", False), ("flow", True)):
        t0 = time.time()
        stats = train(use_flow, args.epochs, device)
        out[name] = {"curve": stats, "seconds": round(time.time() - t0, 1)}
        print(json.dumps(out[name]), flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
