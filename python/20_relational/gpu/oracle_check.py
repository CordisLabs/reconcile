"""Sanity check: is the two-hop task learnable at all, given perfect object features?

If a model with the true object attributes cannot solve it, the task is broken and no
image-based comparison means anything. This trains a small MLP on (colour, shape, x, y)
for each object.

Run: ~/.venvs/gpu/bin/python oracle_check.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_OBJ, COLOURS, SHAPES = 6, 4, 2


def make(batch, device, gen):
    colours = torch.randint(0, COLOURS, (batch, N_OBJ), device=device)
    colours[:, 0] = 0                                  # object 0 is red
    shapes = torch.randint(0, SHAPES, (batch, N_OBJ), device=device)
    pos = torch.rand(batch, N_OBJ, 2, device=device)
    d = torch.cdist(pos, pos) + torch.eye(N_OBJ, device=device) * 1e6
    red = (colours == 0).float().argmax(dim=1)
    nearest = d[torch.arange(batch), red].argmin(dim=1)
    leftmost = pos[:, :, 1].argmin(dim=1)
    same = (shapes[torch.arange(batch), nearest] == shapes[torch.arange(batch), leftmost]).long()
    feats = torch.cat([
        F.one_hot(colours, COLOURS).float(),
        F.one_hot(shapes, SHAPES).float(),
        pos,
    ], dim=2).flatten(1)
    return feats, same


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(0)
    model = nn.Sequential(nn.Linear(8 * N_OBJ, 128), nn.ReLU(), nn.Linear(128, 2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3000):
        x, y = make(128, device, gen)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        x, y = make(1000, device, gen)
        acc = (model(x).argmax(1) == y).float().mean().item()
    print(f"oracle-feature MLP accuracy: {acc:.3f}  (task is learnable if ~1.0)")


if __name__ == "__main__":
    main()
