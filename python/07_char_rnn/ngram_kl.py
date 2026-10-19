"""Does the model's n-gram distribution approach the corpus's?

The syllabus listed this as the second metric and it was never done: sample from the
trained model and compare the distribution of n-grams with the corpus. A model that
memorised the training text would match at all n; a model that learned only local
statistics would match at n=1 and fall apart at n=3.

Run: python3 ngram_kl.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import CharRNN, encode, train  # noqa: E402

DATA = os.path.join(HERE, "..", "_shared", "data", "alice_20k.txt")


def ngram_probs(text, n, smoothing=1e-6):
    counts = {}
    for i in range(len(text) - n + 1):
        g = text[i:i + n]
        counts[g] = counts.get(g, 0) + 1
    total = sum(counts.values())
    return {g: (c + smoothing) / (total + smoothing * len(counts)) for g, c in counts.items()}


def kl(p, q, floor=1e-9):
    """KL(p || q) with q floored, in bits."""
    total = 0.0
    for g, pg in p.items():
        qg = max(q.get(g, 0.0), floor)
        total += pg * np.log2(pg / qg)
    return float(total)


def main():
    text = open(DATA, encoding="utf-8").read()
    chars, stoi, itos = encode(text)
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    rng = np.random.default_rng(0)
    model = CharRNN(len(chars), hidden=48, seed=0)
    corpus = {n: ngram_probs(text, n) for n in (1, 2, 3)}
    print("checkpoint   KL(1)    KL(2)    KL(3)")
    rows = []
    for ckpt in range(1, 5):
        train(model, data, steps=150, seed=ckpt)
        sample = model.sample(stoi, itos, seed_text=text[:20], length=4000, rng=rng)
        kls = [kl(corpus[n], ngram_probs(sample, n)) for n in (1, 2, 3)]
        print(f"{ckpt * 150:>6}      " + "  ".join(f"{k:8.4f}" for k in kls))
        rows.append(kls)
    return rows


if __name__ == "__main__":
    main()
