"""Does memory help where an LSTM fails? Delayed recall, LSTM against RMC.

The RMC paper's claim is that letting memory slots exchange information through attention
lets a recurrent network solve tasks an LSTM cannot. The task here is delayed recall: read
a sequence of symbols, then read an index, and emit the symbol that was at that index. The
longer the sequence, the more the answer depends on information that the recurrent state
has to carry.

Both models share the LSTM controller, the embedding and the output layer. The only
difference is whether the controller's hidden state is mixed with a set of memory slots by
multi-head attention before being read out.

Run: ~/.venvs/gpu/bin/python rmc_vs_lstm.py
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
N_SYM = 4          # alphabet size
N_SLOTS = 8
DIM = 64


def make_batch(batch, length, device, rng):
    seq = torch.randint(1, N_SYM, (batch, length), device=device)
    idx = torch.randint(0, length, (batch,), device=device)
    answer = seq[torch.arange(batch), idx]
    return seq, idx, answer


class LSTMNet(nn.Module):
    def __init__(self, rmc, rmc_step=False, rmc_center=False, rmc_self=False):
        super().__init__()
        self.rmc = rmc
        self.rmc_step = rmc_step
        self.rmc_center = rmc_center
        self.rmc_self = rmc_self
        self.emb = nn.Embedding(N_SYM + 1, DIM)
        self.lstm = nn.LSTM(DIM, DIM, batch_first=True)
        if rmc:
            self.slots = nn.Parameter(torch.zeros(1, N_SLOTS, DIM))
            self.attn = nn.MultiheadAttention(DIM, 4, batch_first=True)
            self.norm = nn.LayerNorm(DIM)
        self.idx_emb = nn.Embedding(64, DIM)
        self.out = nn.Linear(DIM, N_SYM)
        # the plain baseline must see the query too, otherwise it can only guess
        self.out_plain = nn.Linear(2 * DIM, N_SYM)

    def forward(self, seq, idx):
        b = seq.shape[0]
        x = self.emb(seq)
        h, _ = self.lstm(x)                       # (b, L, DIM)
        if self.rmc and self.rmc_self:
            # the paper's mechanism as written: the slots attend to each other, with the
            # current hidden state appended as one more token
            slots = self.slots.expand(b, -1, -1)
            for t in range(h.shape[1]):
                ht = h[:, t:t + 1, :]
                seq = torch.cat([slots, ht], dim=1)
                upd, _ = self.attn(seq, seq, seq)
                slots = self.norm(slots + upd[:, :slots.shape[1]])
            query = self.idx_emb(idx).unsqueeze(1)
            ctx, _ = self.attn(query, slots, slots)
            feat = ctx.squeeze(1)
        elif self.rmc and self.rmc_step:
            # the paper's mechanism: slots exchange information at every time step
            slots = self.slots.expand(b, -1, -1)
            for t in range(h.shape[1]):
                ht = h[:, t:t + 1, :]
                upd, _ = self.attn(slots, ht, ht)
                if self.rmc_center:
                    # re-centre on the learned slot parameters instead of accumulating
                    slots = self.norm(upd + self.slots)
                else:
                    slots = self.norm(slots + upd)
            query = self.idx_emb(idx).unsqueeze(1)
            ctx, _ = self.attn(query, slots, slots)
            feat = ctx.squeeze(1)
        elif self.rmc:
            slots = self.slots.expand(b, -1, -1)
            slots, _ = self.attn(slots, h, h)
            slots = self.norm(slots + self.slots)
            query = self.idx_emb(idx).unsqueeze(1)
            ctx, _ = self.attn(query, slots, slots)
            feat = ctx.squeeze(1)
        else:
            feat = torch.cat([h[:, -1, :], self.idx_emb(idx)], dim=1)
            return self.out_plain(feat)
        return self.out(feat)


def evaluate(model, length, device, batches=20, batch=128):
    model.eval()
    correct = total = 0
    rng = torch.Generator(device=device).manual_seed(1234)
    with torch.no_grad():
        for _ in range(batches):
            seq, idx, ans = make_batch(batch, length, device, rng)
            correct += (model(seq, idx).argmax(1) == ans).sum().item()
            total += batch
    model.train()
    return correct / total


def train(rmc, length, steps, device, lr=3e-3, batch=64, seed=0, rmc_step=False,
          rmc_center=False, rmc_self=False):
    torch.manual_seed(seed)
    model = LSTMNet(rmc, rmc_step=rmc_step, rmc_center=rmc_center, rmc_self=rmc_self).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    for _ in range(steps):
        seq, idx, ans = make_batch(batch, length, device, g)
        loss = F.cross_entropy(model(seq, idx), ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lengths", type=str, default="5,10,20,40")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    rows = []
    for length in lengths:
        row = {"length": length}
        for name, rmc, step, center in (("lstm", False, False, False),
                                        ("rmc_oneshot", True, False, False),
                                        ("rmc_step", True, True, False),
                                        ("rmc_center", True, True, True)):
            model, secs = train(rmc, length, args.steps, device, rmc_step=step,
                                rmc_center=center)
            acc = evaluate(model, length, device)
            row[name] = round(acc, 4)
            row[f"{name}_secs"] = round(secs, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
