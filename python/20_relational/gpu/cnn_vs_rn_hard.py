"""A harder relational question, where the CNN has to fail if the paper is right.

The earlier task (closest pair same colour) was solved by the CNN at 0.871, so it could not
discriminate. This one is a two-hop question:

    "the shape of the object closest to the red one — is it the same as the shape of the
     leftmost object?"

That needs: find the red object, find its nearest neighbour, read its shape, find the
leftmost object, read its shape, compare. Global pooling cannot do it; pairwise relations
can.

Run: ~/.venvs/gpu/bin/python cnn_vs_rn_hard.py --steps 6000
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
SHAPES = 2
N_OBJ = 6
SIZE = 32
CELL = 8


def make_images(batch, device, gen):
    """Objects on a 4x4 grid of cells; each cell holds colour + shape."""
    # channels: colour, shape, and two positional ramps — a convolution is
    # translation-invariant, so without coordinates no model can compute "nearest"
    img = torch.zeros(batch, 4, SIZE, SIZE, device=device)
    ys_ramp = torch.arange(SIZE, device=device).float() / (SIZE - 1)
    img[:, 2, :, :] = ys_ramp[:, None]
    img[:, 3, :, :] = ys_ramp[None, :]
    colours = torch.randint(0, COLOURS, (batch, N_OBJ), device=device)
    shapes = torch.randint(0, SHAPES, (batch, N_OBJ), device=device)
    cells = torch.stack([torch.randperm(16)[:N_OBJ] for _ in range(batch)]).to(device)
    ys = (cells // 4) * CELL
    xs = (cells % 4) * CELL
    # encode colour in the intensity of channel 0, shape in channel 1
    for k in range(N_OBJ):
        y, x = ys[:, k], xs[:, k]
        for dy in range(4):
            for dx in range(4):
                img[torch.arange(batch), 0, y + dy, x + dx] = (colours[:, k] + 1) / COLOURS
                img[torch.arange(batch), 1, y + dy, x + dx] = (shapes[:, k] + 1) / SHAPES

    pos = torch.stack([ys.float() + 2, xs.float() + 2], dim=2)
    # red object = colour 0 (at least one exists by construction if we force it)
    red = torch.zeros(batch, dtype=torch.long, device=device)
    # force object 0 to be red so the question always has an answer
    colours[:, 0] = 0
    red_mask = colours == 0
    red_idx = red_mask.float().argmax(dim=1)
    d = torch.cdist(pos, pos) + torch.eye(N_OBJ, device=device) * 1e6
    nearest = d[torch.arange(batch), red_idx].argmin(dim=1)
    leftmost = pos[:, :, 1].argmin(dim=1)
    same = (shapes[torch.arange(batch), nearest] == shapes[torch.arange(batch), leftmost]).long()
    return img, same


class Encoder(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.enc = Encoder(width)
        self.fc = nn.Sequential(nn.Linear(2 * width, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        f = self.enc(x)
        return self.fc(torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1))


class RelationNet(nn.Module):
    """Object slots, then relations between them.

    The paper's relation module operates on per-object features. Feeding it 4x4 spatial
    tokens instead (the earlier version) blurs object identity, so this adds a small slot
    attention stage that pools the feature map into a fixed number of object slots. The CNN
    baseline shares the same convolutional trunk.
    """

    def __init__(self, width=32, hidden=64, slots=6):
        super().__init__()
        self.enc = Encoder(width)
        self.slots = nn.Parameter(torch.randn(1, slots, width) * 0.1)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.g = nn.Sequential(nn.Linear(2 * width, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.f = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        f = self.enc(x)
        b = f.shape[0]
        tokens = f.flatten(2).transpose(1, 2)          # (b, 16, width)
        q = self.slots.expand(b, -1, -1)
        obj, _ = self.attn(q, tokens, tokens)
        obj = self.norm(obj + self.slots)
        n = obj.shape[1]
        a = obj.unsqueeze(2).expand(b, n, n, obj.shape[2])
        c = obj.unsqueeze(1).expand(b, n, n, obj.shape[2])
        pairs = torch.cat([a, c], dim=3).reshape(b, n * n, -1)
        return self.f(self.g(pairs).sum(dim=1))


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
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", default=os.path.join(HERE, "results_hard.json"))
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
