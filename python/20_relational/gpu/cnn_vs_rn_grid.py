"""Relational reasoning where the encoder can actually see objects.

The earlier attempts fed the module 4x4 spatial tokens of a blurred feature map, so the
comparison measured guessing. Here the objects sit in the cells of a 4x4 grid, and the
convolutional trunk keeps a 4x4 resolution, so each token corresponds to exactly one cell.
Tokens carry a positional embedding, which is what lets any model compute "nearest".

Two sanity checks run before the comparison:
  * an oracle model given the true cell and colour of every object must solve the task;
  * the label must be near balanced.

Run: ~/.venvs/gpu/bin/python cnn_vs_rn_grid.py --steps 4000
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
COLOURS = 4
N_OBJ = 5
GRID = 4
CELL = 8
SIZE = GRID * CELL


def make_batch(batch, device, gen, balanced=True):
    """Objects centred in distinct cells; label = the nearest pair share a colour.

    Balanced by rejection sampling so that the majority-class rate is 0.5; without it the
    label is 75/25 and every model scores the majority rate.
    """
    if balanced:
        half = batch // 2
        pools = {0: [], 1: []}
        have = {0: 0, 1: 0}
        guard = 0
        while (have[0] < half or have[1] < half) and guard < 100:
            guard += 1
            img, y, _c, _col, _p = make_batch(batch, device, gen, balanced=False)
            for cls in (0, 1):
                need = half - have[cls]
                if need > 0:
                    idx = (y == cls).nonzero(as_tuple=True)[0][:need]
                    if len(idx):
                        pools[cls].append(img[idx])
                        have[cls] += len(idx)
        x0 = torch.cat(pools[0])[:half]
        x1 = torch.cat(pools[1])[:half]
        img = torch.cat([x0, x1])
        y = torch.cat([torch.zeros(half, dtype=torch.long, device=device),
                       torch.ones(half, dtype=torch.long, device=device)])
        perm = torch.randperm(img.shape[0], device=device)
        return img[perm], y[perm], None, None, None
    cells = torch.stack([torch.randperm(GRID * GRID, generator=None)[:N_OBJ]
                         for _ in range(batch)])
    colours = torch.randint(0, COLOURS, (batch, N_OBJ), generator=gen)
    img = torch.zeros(batch, COLOURS + 1, SIZE, SIZE)
    # channel c lights up for colour c; the last channel marks "occupied"
    pos = torch.zeros(batch, N_OBJ, 2)
    for k in range(N_OBJ):
        c = cells[:, k]
        row = (c // GRID)
        col = (c % GRID)
        pos[:, k, 0] = row.float()
        pos[:, k, 1] = col.float()
        for dr in range(3):
            for dc in range(3):
                r = row * CELL + CELL // 2 - 1 + dr
                cc = col * CELL + CELL // 2 - 1 + dc
                img[torch.arange(batch), colours[:, k], r, cc] = 1.0
                img[torch.arange(batch), COLOURS, r, cc] = 1.0
    d = torch.cdist(pos, pos) + torch.eye(N_OBJ) * 1e6
    flat = d.reshape(batch, -1)
    idx = flat.argmin(dim=1)
    i, j = idx // N_OBJ, idx % N_OBJ
    label = (colours[torch.arange(batch), i] == colours[torch.arange(batch), j]).long()
    return img.to(device), label.to(device), cells, colours, pos


class Trunk(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(COLOURS + 1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)                       # (b, width, 8, 8) -> pool to 4x4


class CNN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.trunk = Trunk(width)
        self.fc = nn.Sequential(nn.Linear(2 * width, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        f = self.trunk(x)
        f = F.max_pool2d(f, 2)                   # 4x4, one token per cell
        return self.fc(torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1))


class RN(nn.Module):
    def __init__(self, width=32, hidden=64):
        super().__init__()
        self.trunk = Trunk(width)
        self.pos = nn.Parameter(torch.randn(GRID * GRID, width) * 0.1)
        self.g = nn.Sequential(nn.Linear(2 * width, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.f = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        f = F.max_pool2d(self.trunk(x), 2)       # (b, width, 4, 4)
        b = f.shape[0]
        tokens = f.flatten(2).transpose(1, 2) + self.pos[None]
        n = tokens.shape[1]
        a = tokens.unsqueeze(2).expand(b, n, n, tokens.shape[2])
        c = tokens.unsqueeze(1).expand(b, n, n, tokens.shape[2])
        pairs = torch.cat([a, c], dim=3).reshape(b, n * n, -1)
        return self.f(self.g(pairs).sum(dim=1))


class Oracle(nn.Module):
    """Given the true cell and colour of each object: can the task be learned at all?"""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_OBJ * (COLOURS + GRID * GRID), hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, cells, colours):
        one_c = F.one_hot(cells, GRID * GRID).float()
        one_col = F.one_hot(colours, COLOURS).float()
        return self.net(torch.cat([one_c, one_col], dim=2).flatten(1))


def train(model, steps, device, lr=1e-3, batch=64, seed=0, oracle=False, balanced=True):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(steps):
        img, y, cells, colours, pos = make_batch(batch, device, gen, balanced=balanced)
        logits = model(cells.to(device), colours.to(device)) if oracle else model(img)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return time.time() - t0


@torch.no_grad()
def accuracy(model, device, batches=20, batch=128, oracle=False, balanced=True):
    model.eval()
    gen = torch.Generator().manual_seed(777)
    correct = total = 0
    for _ in range(batches):
        img, y, cells, colours, _ = make_batch(batch, device, gen, balanced=balanced)
        logits = model(cells.to(device), colours.to(device)) if oracle else model(img)
        correct += (logits.argmax(1) == y).sum().item()
        total += batch
    model.train()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(HERE, "results_grid.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    gen = torch.Generator().manual_seed(0)
    _, y, _, _, _ = make_batch(4000, device, gen, balanced=False)
    print(f"sanity: label balance {float((y == 1).float().mean()):.3f}", flush=True)

    rows = []
    oracle = Oracle()
    train(oracle, args.steps, device, oracle=True, balanced=False)
    acc = accuracy(oracle, device, oracle=True, balanced=False)
    print(f"sanity: oracle on true features reaches {acc:.3f}", flush=True)
    rows.append({"model": "oracle", "test_acc": round(acc, 4)})

    for name, ctor in (("cnn", CNN), ("cnn_rn", RN)):
        model = ctor()
        secs = train(model, args.steps, device)
        acc = accuracy(model, device)
        rows.append({"model": name, "test_acc": round(acc, 4), "seconds": round(secs, 1)})
        print(json.dumps(rows[-1]), flush=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
