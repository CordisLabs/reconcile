"""Both halves of the paper's label-smoothing claim.

Vaswani et al. §5.4: label smoothing "hurts perplexity ... but improves accuracy and BLEU".
The slice verified the perplexity half analytically (the smoothed model cannot beat a known
floor). This runs the accuracy half: train the same model twice, once with smoothing and
once without, on a copy task long enough that neither is perfect.

Run: python3 label_smoothing.py
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import Transformer, evaluate, train  # noqa: E402

LENGTH = 20
VOCAB = 9
STEPS = 600


def run(smoothing, seed=0):
    model = Transformer(VOCAB, d_model=32, n_heads=4, d_ff=64, max_len=32, seed=0)
    t0 = time.time()
    losses = train(model, STEPS, length=LENGTH, batch=8, lr=2e-3,
                   label_smoothing=smoothing, seed=seed)
    acc, ppl = evaluate(model, n=100, length=LENGTH, seed=2)
    return {"smoothing": smoothing, "accuracy": round(acc, 4), "ppl": round(ppl, 4),
            "final_loss": round(float(np.mean(losses[-20:])), 4),
            "seconds": round(time.time() - t0, 1)}


def main():
    rows = []
    for s in (0.0, 0.1):
        r = run(s)
        print(f"smoothing {s}: exact-match {r['accuracy']:.3f}  ppl {r['ppl']:.4f}  "
              f"loss {r['final_loss']:.3f}  ({r['seconds']}s)", flush=True)
        rows.append(r)
    return rows


if __name__ == "__main__":
    main()
