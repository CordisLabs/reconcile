"""Can a CNN solve a relational question, and does the module help?

The paper's strongest sentence is that powerful convolutional networks do not have a
general capacity to solve relational questions, but can gain it when augmented with a
relation module. The earlier slice showed the module learns a counting task; it never
tested the negative half.

Task: three coloured squares are drawn on a black canvas. The question is whether any two
of them share a colour — a comparison between objects, not a property of any one of them.

CNN:    conv stack -> global pooling -> classifier
CNN+RN: same conv stack -> 4x4 spatial tokens -> pairwise g_theta -> sum -> f_phi

Everything else is identical: same data, same optimiser, same step budget.

Run: ~/.venvs/gpu/bin/python cnn_vs_rn.py
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
COLOURS = 3
SIZE = 32


def make_images(batch, device, gen):
    """Four 5x5 squares on a 4x4 grid of cells; label = the closest pair share a colour.

    Counting colours is not enough: the answer depends on which two objects are nearest
    to each other, so the model has to bind position to colour.
    """
    img = torch.zeros(batch, COLOURS, SIZE, SIZE, device=device)
    colours = torch.randint(0, COLOURS, (batch, 4), device=device)
    cells = torch.stack([torch.randperm(16)[:4] for _ in range(batch)]).to(device)
    ys = (cells // 4) * 8
    xs = (cells % 4) * 8
    for k in range(4):
        y, x = ys[:, k], xs[:, k]
        for dy in range(5):
            for dx in range(5):
                img[torch.arange(batch), colours[:, k], y + dy, x + dx] = 1.0
    # closest pair by centre distance
    pos = torch.stack([ys.float() + 2, xs.float() + 2], dim=2)          # (b, 4, 2)
    d = torch.cdist(pos, pos) + torch.eye(4, device=device) * 1e6
    flat = d.reshape(d.shape[0], -1)
    idx = flat.argmin(dim=1)
    i = idx // 4
    j = idx % 4
    same = (colours[torch.arange(batch), i] == colours[torch.arange(batch), j]).long()
    return img, same


class Encoder(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)                     # (b, width, 4, 4)


class CNN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.enc = Encoder(width)
        self.fc = nn.Sequential(nn.Linear(2 * width, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        f = self.enc(x)
        pooled = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        return self.fc(pooled)


class RelationNet(nn.Module):
    def __init__(self, width=32, hidden=64):
        super().__init__()
        self.enc = Encoder(width)
        self.g = nn.Sequential(nn.Linear(2 * width, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.f = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        f = self.enc(x)                                  # (b, width, 4, 4)
        b = f.shape[0]
        tokens = f.flatten(2).transpose(1, 2)            # (b, 16, width)
        n = tokens.shape[1]
        a = tokens.unsqueeze(2).expand(b, n, n, tokens.shape[2])
        c = tokens.unsqueeze(1).expand(b, n, n, tokens.shape[2])
        pairs = torch.cat([a, c], dim=3).reshape(b, n * n, -1)
        agg = self.g(pairs).sum(dim=1)
        return self.f(agg)


def train(model, steps, device, lr=1e-3, batch=64, seed=0):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    for _ in range(steps):
        img, y = make_images(batch, device, gen)
        loss = F.cross_entropy(model(img), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return time.time() - t0


@torch.no_grad()
def accuracy(model, device, batches=20, batch=128):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(777)
    correct = total = 0
    for _ in range(batches):
        img, y = make_images(batch, device, gen)
        correct += (model(img).argmax(1) == y).sum().item()
        total += batch
    model.train()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    rows = []
    for name, ctor in (("cnn", CNN), ("cnn_rn", RelationNet)):
        model = ctor()
        secs = train(model, args.steps, device)
        acc = accuracy(model, device)
        row = {"model": name, "test_acc": round(acc, 4), "seconds": round(secs, 1),
               "params": sum(p.numel() for p in model.parameters())}
        print(json.dumps(row), flush=True)
        rows.append(row)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
