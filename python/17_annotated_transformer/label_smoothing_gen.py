"""Label smoothing as a generalisation effect, not a saturated one.

The copy task saturates at length 10-20, so accuracy cannot differ. This trains both models
on length 8 and evaluates on length 8 and on length 14, where the model has to generalise.

Run: python3 label_smoothing_gen.py
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import Transformer, evaluate, train  # noqa: E402

TRAIN_LEN = 8
TEST_LENS = (8, 10, 12, 14)
STEPS = 800
VOCAB = 9


def run(smoothing, seed=0):
    model = Transformer(VOCAB, d_model=32, n_heads=4, d_ff=64, max_len=32, seed=0)
    t0 = time.time()
    train(model, STEPS, length=TRAIN_LEN, batch=8, lr=2e-3,
          label_smoothing=smoothing, seed=seed)
    out = {}
    for L in TEST_LENS:
        acc, ppl = evaluate(model, n=100, length=L, seed=2)
        out[L] = (round(acc, 4), round(ppl, 4))
    return out, round(time.time() - t0, 1)


def main():
    for s in (0.0, 0.1):
        out, secs = run(s)
        print(f"smoothing {s}: " + "  ".join(
            f"len {L}: acc {a:.3f} ppl {p:.4f}" for L, (a, p) in out.items()) + f"  ({secs}s)",
            flush=True)
    return


if __name__ == "__main__":
    main()
