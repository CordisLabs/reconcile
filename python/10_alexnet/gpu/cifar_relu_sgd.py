"""The ReLU speed-up claim, in the paper's own optimisation regime.

The earlier comparison used Adam. AlexNet was trained with SGD and momentum, and the
paper's Figure 1 is about *iterations to 25% training error* on a four-layer CIFAR net
with each network's learning rate tuned separately. This reproduces those conditions.

Run: ~/.venvs/gpu/bin/python cifar_relu_sgd.py
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


class FourLayer(nn.Module):
    """Four convolutional layers, as in the paper's CIFAR experiment."""

    def __init__(self, act):
        super().__init__()
        self.act = act
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.c4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        a = F.relu if self.act == "relu" else torch.tanh
        x = F.max_pool2d(a(self.c1(x)), 2)
        x = F.max_pool2d(a(self.c2(x)), 2)
        x = a(self.c3(x))
        x = F.max_pool2d(a(self.c4(x)), 2)
        return self.fc(x.mean(dim=(2, 3)))


def loaders(batch=128):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR10(DATA, train=False, download=True, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, num_workers=4),
            torch.utils.data.DataLoader(te, batch_size=256, shuffle=False, num_workers=4))


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


def run(act, lr, epochs, device, seed=0):
    torch.manual_seed(seed)
    model = FourLayer(act).to(device)
    train_loader, test_loader = loaders()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    t0 = time.time()
    curve = []
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
                curve.append((step, round(loss.item(), 4)))
    acc = evaluate(model, test_loader, device)
    return {"act": act, "lr": lr, "test_acc": round(acc, 4),
            "curve": curve, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "results_sgd.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    rows = []
    for act in ("relu", "tanh"):
        for lr in (0.01, 0.03, 0.1):
            r = run(act, lr, args.epochs, device)
            print(json.dumps({k: v for k, v in r.items() if k != "curve"}), flush=True)
            rows.append(r)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
