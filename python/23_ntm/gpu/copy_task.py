"""Does the NTM actually learn the copy task?

The addressing slice implements and tests the operators. This adds the missing half: a
controller, backpropagation through the addressing, and the copy task the paper uses as
its first demonstration. A sequence is presented, then a blank delimiter, and the network
must reproduce it.

Run: ~/.venvs/gpu/bin/python copy_task.py --length 6 --steps 4000
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = 8
SLOTS = 16
WIDTH = 8


class NTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.ctrl = nn.LSTM(SYMBOLS + WIDTH, hidden, batch_first=True)
        # heads: read key/beta/g/s/gamma ; write key/beta/erase/add
        self.read_head = nn.Linear(hidden, WIDTH + 1 + 1 + 3 + 1)
        self.write_head = nn.Linear(hidden, WIDTH + 1 + WIDTH + WIDTH)
        self.out = nn.Linear(hidden + WIDTH, SYMBOLS)

    def init_state(self, batch, device):
        mem = torch.randn(batch, SLOTS, WIDTH, device=device) * 0.1
        wr = torch.full((batch, SLOTS), 1.0 / SLOTS, device=device)
        return mem, wr, wr, None

    @staticmethod
    def content(mem, key, beta):
        key = key / (key.norm(dim=1, keepdim=True) + 1e-8)
        m = mem / (mem.norm(dim=2, keepdim=True) + 1e-8)
        sim = torch.einsum("bnd,bd->bn", m, key).clamp(-0.99, 0.99)
        return F.softmax(beta.unsqueeze(1) * sim, dim=1)

    @staticmethod
    def shift(prev, weights):
        # three-way shift (-1, 0, +1)
        shifted = torch.zeros_like(prev)
        shifted += weights[:, 0:1] * torch.roll(prev, 1, dims=1)
        shifted += weights[:, 1:2] * prev
        shifted += weights[:, 2:3] * torch.roll(prev, -1, dims=1)
        return shifted

    def step(self, x_t, mem, w_r, w_w, hx, device):
        r = torch.einsum("bn,bnw->bw", w_r, mem)
        inp = torch.cat([x_t, r], dim=1).unsqueeze(1)
        out, hx = self.ctrl(inp, hx)
        h = out.squeeze(1)

        rh = self.read_head(h)
        key_r = rh[:, :WIDTH]
        beta_r = F.softplus(rh[:, WIDTH].clamp(-4, 3)) + 1e-4
        g = torch.sigmoid(rh[:, WIDTH + 1])
        sw = F.softmax(rh[:, WIDTH + 2:WIDTH + 5], dim=1)
        gamma = torch.exp(rh[:, WIDTH + 5].clamp(-2, 2))

        w_c = self.content(mem, key_r, beta_r)
        # the paper gates between *content addressing* and the shifted previous weight;
        # interpolating with the previous weight instead makes the head unable to lock onto
        # a location and leaves only drift
        w_g = g.unsqueeze(1) * w_c + (1 - g.unsqueeze(1)) * self.shift(w_r, sw)
        w_r_new = (w_g.clamp(min=1e-8) ** gamma.unsqueeze(1))
        w_r_new = w_r_new / w_r_new.sum(dim=1, keepdim=True)

        wh = self.write_head(h)
        key_w = wh[:, :WIDTH]
        beta_w = F.softplus(wh[:, WIDTH].clamp(-4, 3)) + 1e-4
        erase = torch.sigmoid(wh[:, WIDTH + 1:WIDTH + 1 + WIDTH])
        add = torch.tanh(wh[:, WIDTH + 1 + WIDTH:])
        w_w_new = self.content(mem, key_w, beta_w)

        mem = mem * (1 - w_w_new.unsqueeze(2) * erase.unsqueeze(1))
        mem = mem + w_w_new.unsqueeze(2) * add.unsqueeze(1)
        mem = mem.clamp(-3.0, 3.0)          # keep the memory bounded across long sequences

        logits = self.out(torch.cat([h, r], dim=1))
        return logits, mem, w_r_new, w_w_new, hx

    def forward(self, seq):
        """seq: (b, T, SYMBOLS). Returns logits for every step."""
        b, T, _ = seq.shape
        device = seq.device
        mem, w_r, w_w, hx = self.init_state(b, device)
        outs = []
        for t in range(T):
            logits, mem, w_r, w_w, hx = self.step(seq[:, t], mem, w_r, w_w, hx, device)
            outs.append(logits)
        return torch.stack(outs, dim=1)


def make_batch(batch, length, device, gen):
    symbols = torch.randint(0, SYMBOLS, (batch, length), device=device)
    inputs = torch.zeros(batch, 2 * length + 1, SYMBOLS, device=device)
    inputs.scatter_(2, symbols.unsqueeze(2), 1.0)          # copy phase
    target = torch.full((batch, 2 * length + 1), -100, dtype=torch.long, device=device)
    # the model must emit the symbols after the delimiter
    shifted = torch.zeros(batch, length, dtype=torch.long, device=device)
    target[:, length + 1:] = symbols
    return inputs, target


def train(length, steps, device, lr=1e-3, batch=32, seed=0, mixed=False, model=None):
    torch.manual_seed(seed)
    model = model if model is not None else NTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    losses = []
    for i in range(steps):
        L = int(torch.randint(3, 9, (1,), device=device, generator=gen).item()) if mixed else length
        inputs, target = make_batch(batch, L, device, gen)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, SYMBOLS), target.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(loss))
        if (i + 1) % max(1, steps // 4) == 0:
            print(f"  step {i+1}: loss {losses[-1]:.4f}", flush=True)
    return model, losses, time.time() - t0


@torch.no_grad()
def accuracy(model, length, device, batches=20, batch=64):
    """Exact-sequence accuracy at a given length; length may differ from training."""
    gen = torch.Generator(device=device).manual_seed(999)
    model.eval()
    correct = total = 0
    for _ in range(batches):
        inputs, target = make_batch(batch, length, device, gen)
        logits = model(inputs)
        pred = logits[:, length + 1:].argmax(-1)
        gold = target[:, length + 1:]
        correct += (pred == gold).all(dim=1).sum().item()   # exact sequence
        total += batch
    model.train()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=6)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--test-lengths", type=str, default="6,8,10,12")
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--schedule", type=str, default="",
                    help="curriculum, e.g. 5:12000,8:8000,10:8000")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ckpt", type=str, default="", help="save/load the model here")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, f"copy length {args.length}", flush=True)
    if args.schedule:
        # curriculum: the same model, trained on growing lengths
        model = NTM(hidden=args.hidden if hasattr(args, "hidden") else 64).to(device)
        if args.ckpt and os.path.exists(args.ckpt) and not args.fresh:
            model.load_state_dict(torch.load(args.ckpt, map_location=device))
            print(f"resumed from {args.ckpt}", flush=True)
        secs = 0.0
        for part in args.schedule.split(","):
            L, n = part.split(":")
            L, n = int(L), int(n)
            model, losses, s_ = train(L, n, device, model=model, mixed=False, lr=args.lr,
                                      batch=args.batch)
            secs += s_
            print(f"  curriculum length {L}: final loss {losses[-1]:.4f} ({s_:.0f}s)", flush=True)
            if args.ckpt:
                torch.save(model.state_dict(), args.ckpt)
                print(f"  checkpoint saved to {args.ckpt}", flush=True)
    else:
        model, losses, secs = train(args.length, args.steps, device, mixed=args.mixed, lr=args.lr)
    out = {"train_length": args.length, "steps": args.steps,
           "final_loss": round(float(sum(losses[-20:]) / 20), 4), "seconds": round(secs, 1)}
    for L in [int(x) for x in args.test_lengths.split(",")]:
        out[f"acc_len_{L}"] = round(accuracy(model, L, device), 4)
    print(json.dumps(out), flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
