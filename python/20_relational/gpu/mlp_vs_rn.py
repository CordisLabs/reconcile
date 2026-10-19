"""Do relations help, holding the input fixed?

The image-based comparison kept failing for a reason unrelated to the claim: a global-pool
CNN can answer most of these questions by counting, and the tasks hard enough to need
relations were also too hard for anything to learn. This removes the image pipeline
entirely. Both models get the same per-object features — colour one-hot and grid position —
and the question needs a pairwise comparison.

  MLP: concatenate all objects, two hidden layers.
  RN:  pairwise g, sum, f.

The MLP can in principle represent any function of the concatenation; the question is
whether it learns this one as easily.

Run: ~/.venvs/gpu/bin/python mlp_vs_rn.py
"""

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
FEAT = COLOURS + 2          # colour one-hot + (row, col)


def make_batch(batch, device, gen, balanced=True):
    """Balanced by rejection sampling: half the batch has label 1, half has 0."""
    if not balanced:
        return _make_batch_raw(batch, device, gen)
    half = batch // 2
    pools = {0: [], 1: []}
    have = {0: 0, 1: 0}
    guard = 0
    while (have[0] < half or have[1] < half) and guard < 100:
        guard += 1
        x, y = _make_batch_raw(batch, device, gen)
        for cls in (0, 1):
            need = half - have[cls]
            if need > 0:
                idx = (y == cls).nonzero(as_tuple=True)[0][:need]
                if len(idx):
                    pools[cls].append(x[idx])
                    have[cls] += len(idx)
    x0 = torch.cat(pools[0])[:half]
    x1 = torch.cat(pools[1])[:half]
    x = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(half, dtype=torch.long, device=device),
                   torch.ones(half, dtype=torch.long, device=device)])
    perm = torch.randperm(x.shape[0], device=device)
    return x[perm], y[perm]


def _make_batch_raw(batch, device, gen):
    cells = torch.stack([torch.randperm(GRID * GRID)[:N_OBJ] for _ in range(batch)])
    colours = torch.randint(0, COLOURS, (batch, N_OBJ), generator=gen)
    rows = (cells // GRID).float()
    cols = (cells % GRID).float()
    one_c = F.one_hot(colours, COLOURS).float()
    pos = torch.stack([rows, cols], dim=2) / (GRID - 1)
    feats = torch.cat([one_c, pos], dim=2).to(device)          # (b, N, FEAT)
    d = torch.cdist(pos, pos) + torch.eye(N_OBJ) * 1e6
    idx = d.reshape(batch, -1).argmin(dim=1)
    i, j = idx // N_OBJ, idx % N_OBJ
    label = (colours[torch.arange(batch), i] == colours[torch.arange(batch), j]).long().to(device)
    return feats, label


class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_OBJ * FEAT, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x.flatten(1))


class RN(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(2 * FEAT, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.f = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        b, n, _ = x.shape
        a = x.unsqueeze(2).expand(b, n, n, x.shape[2])
        c = x.unsqueeze(1).expand(b, n, n, x.shape[2])
        pairs = torch.cat([a, c], dim=3).reshape(b, n * n, -1)
        return self.f(self.g(pairs).sum(dim=1))


def train(model, steps, device, lr=1e-3, batch=64, seed=0):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(steps):
        x, y = make_batch(batch, device, gen)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return time.time() - t0


@torch.no_grad()
def accuracy(model, device, batches=20, batch=256):
    model.eval()
    gen = torch.Generator().manual_seed(999)
    correct = total = 0
    for _ in range(batches):
        x, y = make_batch(batch, device, gen)
        correct += (model(x).argmax(1) == y).sum().item()
        total += batch
    model.train()
    return correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(0)
    _, y = make_batch(4000, device, gen)
    print(f"label balance {float((y == 1).float().mean()):.3f}", flush=True)
    rows = []
    for name, ctor in (("mlp", MLP), ("rn", RN)):
        model = ctor()
        secs = train(model, 4000, device)
        acc = accuracy(model, device)
        rows.append({"model": name, "test_acc": round(acc, 4), "seconds": round(secs, 1)})
        print(json.dumps(rows[-1]), flush=True)
    with open(os.path.join(HERE, "results_mlp_rn.json"), "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
