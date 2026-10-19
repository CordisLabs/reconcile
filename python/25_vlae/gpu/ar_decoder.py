"""The other half of the lossy autoencoder: an autoregressive decoder.

The paper's argument is that a VAE only needs to encode the global structure if the
decoder itself models the local texture. This compares a Bernoulli decoder that treats all
784 pixels as independent with one that generates the image row by row, conditioned on the
previous row and on the latent.

Run: ~/.venvs/gpu/bin/python ar_decoder.py --epochs 2
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from flow_prior import FlowPrior  # noqa: E402
DATA = os.path.join(HERE, "data")
LATENT = 16
ROWS = 28


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


class IndependentDecoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.fc = nn.Linear(latent, 64 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.net(h)


class AutoregressiveDecoder(nn.Module):
    """Generates the image row by row: row t depends on z and on row t-1."""

    def __init__(self, latent=LATENT, hidden=128):
        super().__init__()
        self.init = nn.Linear(latent, hidden)
        self.cell = nn.GRUCell(ROWS + latent, hidden)
        self.out = nn.Linear(hidden, ROWS)

    def forward(self, z, target=None, teacher_forcing=True):
        b = z.shape[0]
        h = torch.tanh(self.init(z))
        prev = torch.zeros(b, ROWS, device=z.device)
        logits = []
        for t in range(ROWS):
            h = self.cell(torch.cat([prev, z], dim=1), h)
            logits.append(self.out(h))
            prev = target[:, t] if (teacher_forcing and target is not None) else torch.sigmoid(logits[-1])
        return torch.stack(logits, dim=1)          # (b, ROWS, ROWS)


def loaders(batch=128):
    tf = torchvision.transforms.ToTensor()
    train = torchvision.datasets.MNIST(DATA, train=True, download=True, transform=tf)
    test = torchvision.datasets.MNIST(DATA, train=False, download=True, transform=tf)
    return (torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=2),
            torch.utils.data.DataLoader(test, batch_size=512, shuffle=False, num_workers=2))


def bits_per_dim(log_px, n_pixels=784):
    return -log_px / (n_pixels * np.log(2))


def train(use_ar, epochs, device, seed=0, lr=1e-3, use_flow=False):
    torch.manual_seed(seed)
    enc = Encoder().to(device)
    dec = (AutoregressiveDecoder() if use_ar else IndependentDecoder()).to(device)
    prior = FlowPrior(LATENT).to(device) if use_flow else None
    params = list(enc.parameters()) + list(dec.parameters())
    if prior is not None:
        params += list(prior.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    train_loader, test_loader = loaders()
    out = []
    for ep in range(epochs):
        for x, _ in train_loader:
            x = x.to(device)
            mu, lv = enc(x)
            std = torch.exp(0.5 * lv)
            z = mu + std * torch.randn_like(std)
            if use_ar:
                logits = dec(z, target=x.squeeze(1), teacher_forcing=True)
                log_px = -F.binary_cross_entropy_with_logits(
                    logits, x.squeeze(1), reduction="none").sum(dim=(1, 2))
            else:
                logits = dec(z)
                log_px = -F.binary_cross_entropy_with_logits(
                    logits, x, reduction="none").sum(dim=(1, 2, 3))
            log_q = -0.5 * ((z - mu) / std).pow(2).sum(1) - lv.sum(1) - 0.5 * LATENT * np.log(2 * np.pi)
            log_p = (prior.log_prob(z) if prior is not None
                     else -0.5 * (z ** 2).sum(1) - 0.5 * LATENT * np.log(2 * np.pi))
            loss = -(log_px + log_p - log_q).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        bpd = evaluate(enc, dec, use_ar, test_loader, device, prior=prior)
        out.append((ep + 1, bpd))
        print(f"  epoch {ep+1}: test bits/dim {bpd:.4f}", flush=True)
    return out


@torch.no_grad()
def evaluate(enc, dec, use_ar, loader, device, batches=8, prior=None):
    enc.eval(); dec.eval()
    total = n = 0
    for i, (x, _) in enumerate(loader):
        if i >= batches:
            break
        x = x.to(device)
        mu, lv = enc(x)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        if use_ar:
            logits = dec(z, target=x.squeeze(1), teacher_forcing=True)
            log_px = -F.binary_cross_entropy_with_logits(logits, x.squeeze(1), reduction="none").sum(dim=(1, 2))
        else:
            logits = dec(z)
            log_px = -F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=(1, 2, 3))
        log_q = -0.5 * ((z - mu) / std).pow(2).sum(1) - lv.sum(1) - 0.5 * LATENT * np.log(2 * np.pi)
        log_p = (prior.log_prob(z) if prior is not None
                 else -0.5 * (z ** 2).sum(1) - 0.5 * LATENT * np.log(2 * np.pi))
        total += bits_per_dim(log_px + log_p - log_q).sum().item()
        n += x.shape[0]
    enc.train(); dec.train()
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "results_ar.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    out = {}
    for name, use_ar, use_flow in (("independent", False, False),
                                   ("autoregressive", True, False),
                                   ("ar_plus_flow", True, True)):
        t0 = time.time()
        curve = train(use_ar, args.epochs, device, use_flow=use_flow)
        out[name] = {"curve": curve, "seconds": round(time.time() - t0, 1)}
        print(json.dumps({name: out[name]}), flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
