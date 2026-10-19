"""ReLU against tanh on CIFAR-10, at real image scale.

The AlexNet paper attributes part of its speed to ReLU: "deep convolutional neural
networks with ReLUs train several times faster than their equivalents with tanh units".
The toy version of this slice (small conv net on synthetic data) did not reproduce that.
This script runs the comparison on CIFAR-10 with an identical architecture, identical
initialisation distribution and identical data order — the only difference is the
activation.

Run: ~/.venvs/gpu/bin/python cifar_relu.py --epochs 3
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


class Net(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        a = F.relu if self.act == "relu" else torch.tanh
        x = F.max_pool2d(a(self.c1(x)), 2)
        x = F.max_pool2d(a(self.c2(x)), 2)
        x = F.max_pool2d(a(self.c3(x)), 2)
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


def run(act, epochs, device, seed=0, lr=1e-3):
    torch.manual_seed(seed)
    model = Net(act).to(device)
    train_loader, test_loader = loaders()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = []
    accs = []
    t0 = time.time()
    step = 0
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                steps.append((step, round(loss.item(), 4)))
                if step % 390 == 0:
                    acc = evaluate(model, test_loader, device)
                    accs.append((step, round(acc, 4)))
    acc = evaluate(model, test_loader, device)
    return {"act": act, "seconds": round(time.time() - t0, 1), "test_acc": round(acc, 4),
            "curve": steps, "acc_curve": accs,
            "params": sum(p.numel() for p in model.parameters())}


def first_step_below(curve, target):
    for step, loss in curve:
        if loss <= target:
            return step
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    results = []
    for act in ("relu", "tanh"):
        r = run(act, args.epochs, device, lr=args.lr)
        print(json.dumps({k: v for k, v in r.items() if k != "curve"}), flush=True)
        results.append(r)
    for target in (1.5, 1.3, 1.1):
        a = first_step_below(next(r for r in results if r["act"] == "relu")["curve"], target)
        b = first_step_below(next(r for r in results if r["act"] == "tanh")["curve"], target)
        print(f"loss <= {target}: relu {a} steps, tanh {b} steps, ratio "
              f"{(b / a if a and b else float('nan')):.2f}x", flush=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
