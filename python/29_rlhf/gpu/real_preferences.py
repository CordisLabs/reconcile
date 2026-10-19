"""Train a reward model on real human preferences instead of a known rule.

The slice's preference data comes from a hidden reward, so it can check recovery but not
whether human comparisons are learnable at all. This uses Anthropic's HH-RLHF, which
contains pairs where a human chose one assistant reply over another.

The reward model is deliberately weak — a linear score over hashed bag-of-words — so the
question is only whether preferences are learnable above chance, not how good a model can
get. Evaluation is pairwise accuracy on held-out conversations.

Run: ~/.venvs/gpu/bin/python real_preferences.py --train 6000 --test 1500
"""

import argparse
import json
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
API = ("https://datasets-server.huggingface.co/rows?dataset=Anthropic/hh-rlhf"
       "&config=default&split={split}&offset={offset}&length={length}")
BUCKETS = 4096


def fetch(split, n, chunk=100, retries=5):
    rows = []
    offset = 0
    while len(rows) < n:
        url = API.format(split=split, offset=offset, length=chunk)
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=60) as fh:
                    payload = json.load(fh)
                break
            except Exception as exc:            # transient 5xx from the datasets server
                if attempt == retries - 1:
                    raise
                print(f"  retry {attempt + 1}: {exc}", flush=True)
                time.sleep(2 ** attempt)
        got = payload.get("rows", [])
        if not got:
            break
        rows.extend(r["row"] for r in got)
        offset += len(got)
    return rows[:n]


def cached(split, n):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"hh_{split}_{n}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    rows = fetch(split, n)
    with open(path, "w") as fh:
        json.dump(rows, fh)
    return rows


def reply(cell):
    """The last assistant turn, which is what the human was judging."""
    parts = cell.split("Assistant:")
    return parts[-1].strip() if len(parts) > 1 else cell


def featurise(text, buckets=BUCKETS):
    """Hashed bag of words, L1-normalised, plus a length feature.

    Normalising matters: with binary features and a handful of nonzeros per example the
    gradients are tiny, and the model sits at ln 2 for every epoch.
    """
    v = np.zeros(buckets, dtype=np.float32)
    tokens = []
    for tok in text.lower().split():
        tok = "".join(c for c in tok if c.isalnum())
        if tok:
            tokens.append(tok)
    for tok in tokens:
        v[hash(tok) % (buckets - 1)] += 1.0
    if tokens:
        v[:buckets - 1] /= np.sqrt(len(tokens))
    # a much longer answer is often the preferred one; give the model that signal
    v[buckets - 1] = np.log1p(len(tokens)) / 5.0
    return v


def build(rows):
    xs, ys = [], []
    for r in rows:
        xs.append(featurise(reply(r["chosen"])))
        xs.append(featurise(reply(r["rejected"])))
        ys.append(1.0)
        ys.append(0.0)
    return np.stack(xs), np.array(ys, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--test", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    t0 = time.time()
    train_rows = cached("train", args.train)
    test_rows = cached("test", args.test)
    print(f"fetched {len(train_rows)} train / {len(test_rows)} test pairs "
          f"({time.time() - t0:.0f}s)", flush=True)

    X, _ = build(train_rows)
    Xt, _ = build(test_rows)
    print(f"features: {X.shape}, nonzeros {(X > 0).mean():.3f}", flush=True)

    model = torch.nn.Linear(BUCKETS, 1, bias=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-6)
    Xt_gpu = torch.from_numpy(X).to(device)
    n = len(train_rows)
    for ep in range(args.epochs):
        perm = torch.randperm(n, device=device)          # shuffle *pairs*, not examples
        total = 0.0
        for i in range(0, n, 128):
            pairs = perm[i:i + 128]
            a, b = 2 * pairs, 2 * pairs + 1             # chosen, rejected
            ra = model(Xt_gpu[a]).squeeze(1)
            rb = model(Xt_gpu[b]).squeeze(1)
            loss = -F.logsigmoid(ra - rb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep + 1}: loss {total / max(1, (2 * n // 256)):.4f}", flush=True)

    # held-out pairwise accuracy: does the score prefer the human's choice?
    with torch.no_grad():
        s = model(torch.from_numpy(Xt).to(device)).squeeze(1).cpu().numpy()
    chosen, rejected = s[0::2], s[1::2]
    acc = float((chosen > rejected).mean())
    ties = float((chosen == rejected).mean())
    print(f"held-out pairwise accuracy: {acc:.4f}  (chance 0.5, ties {ties:.4f})", flush=True)
    with open(os.path.join(HERE, "results_real_prefs.json"), "w") as fh:
        json.dump({"train_pairs": len(train_rows), "test_pairs": len(test_rows),
                   "accuracy": round(acc, 4), "epochs": args.epochs}, fh, indent=2)


if __name__ == "__main__":
    main()
