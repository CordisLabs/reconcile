"""Plain against residual networks on CIFAR-10 at real depth.

The ResNet paper's central observation is that adding layers to a plain network makes
training *worse*, not just unable to improve — the degradation problem. The toy version of
this slice showed it with 24 blocks on synthetic data; this runs the comparison on
CIFAR-10 with 20 conv layers, which is much closer to the paper's setting.

Run: ~/.venvs/gpu/bin/python cifar_depth.py --blocks 10 --epochs 4
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


class PlainBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(h)))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(h)) + x)


class Net(nn.Module):
    def __init__(self, blocks, residual, width=32):
        super().__init__()
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        Block = ResBlock if residual else PlainBlock
        self.blocks = nn.Sequential(*[Block(width) for _ in range(blocks)])
        self.fc = nn.Linear(width, 10)

    def forward(self, x):
        x = F.relu(self.stem(x))
        x = self.blocks(x)
        return self.fc(x.mean(dim=(2, 3)))


def loaders(batch=128):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = torchvision.datasets.CIFAR10(DATA, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(DATA, train=False, download=True, transform=tf)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=4),
        torch.utils.data.DataLoader(test, batch_size=256, shuffle=False, num_workers=4),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total


def run(blocks, residual, epochs, device, seed=0):
    torch.manual_seed(seed)
    model = Net(blocks, residual).to(device)
    train_loader, test_loader = loaders()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    curve = []
    step = 0
    for ep in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 100 == 0:
                curve.append((step, round(loss.item(), 4)))
        curve.append((step, round(loss.item(), 4)))
    acc = evaluate(model, test_loader, device)
    return {"blocks": blocks, "layers": 2 * blocks, "residual": residual,
            "params": sum(p.numel() for p in model.parameters()),
            "test_acc": round(acc, 4), "seconds": round(time.time() - t0, 1),
            "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, f"depth {2 * args.blocks} conv layers", flush=True)
    results = []
    for residual in (False, True):
        r = run(args.blocks, residual, args.epochs, device)
        tag = "residual" if residual else "plain"
        print(json.dumps({k: v for k, v in r.items() if k != "curve"}), flush=True)
        print(f"  {tag}: final loss {r['curve'][-1][1]}", flush=True)
        results.append(r)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
