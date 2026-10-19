"""Prints the numbers cited in this slice's README.
Run: python3 report.py
"""

import time

import numpy as np

from implementation import LSTM, VanillaRNN, accuracy, copy_batch, one_hot, train


def retention(model_cls, T, seed=0):
    """Norm of the gradient with respect to the initial hidden state.

    This is the quantity the blog post is really talking about: how much signal from
    the start of the sequence still reaches the parameters after T steps.
    """
    model = model_cls(12, 32, 10, seed=seed)
    rng = np.random.default_rng(7)
    seq, targets, mask = copy_batch(1, T, 10, rng)
    x = one_hot(seq, 12)
    model.loss(x, targets, mask)
    model.backward(targets, mask)
    return float(np.linalg.norm(model.dh0))


def main():
    print("== gradient reaching back to the initial hidden state ==")
    print("  T      LSTM        vanilla RNN   ratio")
    for T in (1, 5, 10, 20, 40):
        a = retention(LSTM, T)
        b = retention(VanillaRNN, T)
        print(f"  {T:>2}   {a:.3e}   {b:.3e}   {a / b:>8.1f}x")
    print()

    print("== copy task, same budget for both models (hidden 32, lr 3e-3) ==")
    print("  T     steps   LSTM acc   vanilla acc   chance")
    for T, steps in [(5, 1500), (10, 4000)]:
        row = []
        for cls in (LSTM, VanillaRNN):
            model = cls(12, 32, 10, seed=0)
            t0 = time.time()
            train(model, steps=steps, T=T, k=10, batch=32, lr=3e-3, seed=0)
            acc = accuracy(model, 256, T, 10, np.random.default_rng(1))
            row.append((acc, time.time() - t0))
        print(
            f"  {T:>2}   {steps:>5}   {row[0][0]:>8.3f}   {row[1][0]:>11.3f}   {1 / 10:.3f}"
            f"   ({row[0][1]:.1f}s / {row[1][1]:.1f}s)"
        )
    print()

    print("== can the LSTM at least memorise? (single fixed batch, T=10) ==")
    rng = np.random.default_rng(0)
    model = LSTM(12, 32, 10, seed=0)
    seq, targets, mask = copy_batch(32, 10, 10, rng)
    x = one_hot(seq, 12)
    from implementation import Adam

    opt = Adam(model.params, lr=3e-3)
    for _ in range(3000):
        model.loss(x, targets, mask)
        g = model.backward(targets, mask)
        opt.step(model.params, g)
    logits, _ = model.forward(x)
    pred = logits.argmax(axis=-1)
    acc = float(((pred == targets.T).astype(float) * mask).sum() / mask.sum())
    print(f"  loss {model.loss(x, targets, mask):.4f} nats, accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
