"""Tests for the character-level RNN. Run with: python3 test_rnn.py

Three independent judges, none of them "it produced numbers":
  1. the analytic BPTT gradient against finite differences;
  2. a memorisation task where the right answer is known exactly;
  3. held-out bits/char against the order-0 entropy of the same text.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from _shared.gradcheck import check_gradients  # noqa: E402
from implementation import CharRNN, encode, order0_entropy, train  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_gradient():
    """BPTT against central differences on a model small enough to perturb."""
    rng = np.random.default_rng(0)
    model = CharRNN(5, 4, seed=1)
    for k in model.params:
        model.params[k] = model.params[k] + rng.normal(0, 0.1, model.params[k].shape)

    x = rng.integers(0, 5, size=(2, 6))
    y = rng.integers(0, 5, size=(2, 6))

    def loss():
        return model.loss(x, y)

    loss()
    grads = model.backward(x, y)
    max_abs, max_rel, ok = check_gradients(loss, model.params, grads, eps=1e-6)
    check(
        "BPTT gradient vs finite differences",
        ok,
        f"max_abs = {max_abs:.2e}, max_rel = {max_rel:.2e}",
    )


def test_learns_a_repeating_pattern():
    """If the implementation can learn at all, a period-3 pattern is trivial."""
    pattern = "abc"
    text = (pattern * 80)[:240]
    chars, stoi, _ = encode(text)
    data = np.array([stoi[c] for c in text])

    model = CharRNN(len(chars), 16, seed=0)
    trace = train(model, data, steps=400, seq_len=24, batch_size=8, lr=3e-3, seed=0)
    final = float(np.mean(trace[-20:]))
    check("memorises a period-3 pattern", final < 0.05, f"final loss = {final:.4f} nats")

    # Greedy decoding must reproduce the pattern exactly.
    _, stoi, itos = encode(text)
    window = data[:24][None, :]
    logits, _ = model.forward(window)
    pred = logits[:, 0, :].argmax(axis=-1)
    check(
        "greedy decoding reproduces the pattern",
        bool((pred[1:] == data[2:25]).all()),
        f"first chars: {''.join(itos[int(i)] for i in pred[:9])}",
    )


def test_beats_order0_entropy():
    """The paper's point: context, not just frequency.

    A model that only learned the marginal character distribution would land at the
    order-0 entropy. Beating it by a full bit means the hidden state is carrying
    context.
    """
    path = os.path.join(HERE, "..", "_shared", "data", "alice_20k.txt")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    chars, stoi, itos = encode(text)
    data = np.array([stoi[c] for c in text])

    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]

    model = CharRNN(len(chars), 64, seed=0)
    train(model, train_data, steps=2000, seq_len=48, batch_size=16, lr=3e-3, seed=0)

    losses = [
        model.loss(val_data[s:s + 48][None, :], val_data[s + 1:s + 49][None, :])
        for s in range(0, len(val_data) - 49, 48)
    ]
    val_bits = float(np.mean(losses)) / np.log(2)
    order0 = order0_entropy("".join(itos[int(i)] for i in val_data))

    check(
        "held-out bits/char beats the order-0 bound by > 1 bit",
        val_bits < order0 - 1.0,
        f"val {val_bits:.3f} vs order-0 {order0:.3f} bits/char",
    )
    check(
        "held-out bits/char is below the log2(67) uniform bound",
        val_bits < np.log2(len(chars)),
        f"{val_bits:.3f} vs {np.log2(len(chars)):.3f}",
    )


if __name__ == "__main__":
    test_gradient()
    test_learns_a_repeating_pattern()
    test_beats_order0_entropy()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
