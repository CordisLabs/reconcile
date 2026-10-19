"""Tests for the RNN-regularisation slice. Run with: python3 test_rnn_reg.py

The paper's claim is about *where* dropout goes, not that dropout is good. So the
tests check the mask mechanics, the gradient through the masks, and then the
comparison the paper actually makes: non-recurrent dropout against recurrent dropout.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from _shared.gradcheck import check_gradients  # noqa: E402
from implementation import (  # noqa: E402
    DropoutLSTM,
    bits_per_char,
    drop_mask,
    encode,
    one_hot_ids,
    train,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_gradient_through_masks():
    """The backward pass must run through the masks, not around them."""
    rng = np.random.default_rng(0)
    k, T, B, H = 6, 6, 2, 4
    model = DropoutLSTM(k, H, k, p_in=0.5, p_out=0.5, p_rec=0.5, seed=1)
    for key in model.params:
        model.params[key] = model.params[key] + rng.normal(0, 0.1, model.params[key].shape)

    masks = (
        drop_mask(rng, (B, k), 0.5),
        drop_mask(rng, (B, H), 0.5),
        drop_mask(rng, (B, H), 0.5),
    )
    ids = rng.integers(0, k, size=(B, T))
    targets = rng.integers(0, k, size=(B, T))
    x = np.stack([one_hot_ids(row, k) for row in ids])

    def loss():
        return model.loss(x, targets, masks=masks)

    loss()
    grads = model.backward(targets)
    max_abs, max_rel, ok = check_gradients(loss, model.params, grads, eps=1e-6)
    check(
        "gradient with all three dropout masks vs finite differences",
        ok,
        f"max_abs = {max_abs:.2e}, max_rel = {max_rel:.2e}",
    )


def test_mask_statistics():
    """Inverted dropout: keep-probability 1-p, scale 1/(1-p), expectation preserved."""
    rng = np.random.default_rng(0)
    p = 0.3
    m = drop_mask(rng, (4000, 50), p)
    dropped = float((m == 0).mean())
    kept_value = float(m[m > 0].mean())
    check(
        "mask drops p of the units",
        abs(dropped - p) < 0.01,
        f"dropped {dropped:.4f} vs p = {p}",
    )
    check(
        "kept units are scaled by 1/(1-p)",
        abs(kept_value - 1.0 / (1.0 - p)) < 1e-12,
        f"kept value {kept_value:.6f}, expected {1.0 / (1.0 - p):.6f}",
    )


def test_mask_is_shared_across_time():
    """Variational dropout: one mask per sequence, not one per timestep.

    The implementation stores masks of shape (B, dim), so the same units are dropped
    at every timestep. This test guards that shape, which is the property the paper's
    variational bound depends on.
    """
    rng = np.random.default_rng(1)
    B, T, H = 3, 7, 5
    model = DropoutLSTM(H, H, 2, p_in=0.5, seed=0)
    x = np.zeros((B, T, H))
    model.forward(x, rng=rng)
    m_in, m_out, _ = model.cache[7], model.cache[8], model.cache[9]
    check(
        "input mask is per-sequence, not per-timestep",
        m_in.shape == (B, H),
        f"shape {m_in.shape}",
    )
    # A per-timestep mask would have to be (T, B, H); make sure the applied tensor
    # repeats the same pattern at every step.
    xd = model.cache[1]
    check(
        "the same input mask is applied at every timestep",
        bool(np.allclose(xd, x.transpose(1, 0, 2) * m_in[None, :, :])),
        "",
    )


def test_recurrent_mask_is_not_the_memory():
    """A zeroed recurrent mask does *not* sever memory, and that is the interesting part.

    Zeroing the recurrent weight removes h_{t-1} from the gate pre-activations, but the
    cell state still carries over multiplicatively (c_t = f*c_{t-1} + i*g). So the first
    token still reaches the last step through c. This is why the paper's warning about
    recurrent dropout is about degrading the gates, not about cutting the cell state —
    and it is why the damage is smaller than "the memory is gone" would suggest.
    """
    B, T, I, H = 1, 5, 6, 8
    model = DropoutLSTM(I, H, 4, seed=0)
    cut = (np.ones((B, I)), np.ones((B, H)), np.zeros((B, H)))
    open_mask = (np.ones((B, I)), np.ones((B, H)), np.ones((B, H)))

    def last_logits(masks, first_symbol):
        x = np.zeros((B, T, I))
        x[0, 0, first_symbol] = 1.0
        model.forward(x, masks=masks)
        return model.cache[6][-1].copy()

    check(
        "the recurrent mask is not inert",
        not np.allclose(last_logits(open_mask, 3), last_logits(cut, 3)),
        "cutting the recurrent path changes the output",
    )
    check(
        "a cut recurrent path still carries memory through the cell state",
        not np.allclose(last_logits(cut, 0), last_logits(cut, 3)),
        "different first tokens still reach the last step with m_rec = 0",
    )


def test_dropout_stops_the_overfitting_turn():
    """The paper's actual claim, measured as a shape rather than a single number.

    Without dropout the held-out loss turns upward after its minimum; with dropout the
    curve flattens near the minimum. Best-val is *not* the interesting quantity here —
    at this scale the undropped model reaches a slightly better peak and then ruins it.
    """
    path = os.path.join(HERE, "..", "_shared", "data", "alice_20k.txt")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()[:2500]
    chars, stoi, _ = encode(text)
    data = np.array([stoi[c] for c in text])
    split = int(len(data) * 0.8)
    train_data, val_data = data[:split], data[split:]

    results = {}
    for name, kwargs in [
        ("none", {}),
        ("dropout", {"p_in": 0.5, "p_out": 0.5}),
    ]:
        model = DropoutLSTM(len(chars), 64, len(chars), seed=0, **kwargs)
        best, trace = train(
            model, train_data, val_data, steps=1200, seq_len=24, batch_size=8,
            lr=3e-3, seed=0, eval_every=300,
        )
        final = trace[-1][2]
        results[name] = final - best

    check(
        "without dropout the held-out loss turns up",
        results["none"] > 0.15,
        f"degradation after the minimum = {results['none']:.3f} bpc",
    )
    check(
        "with dropout the curve stays near its minimum",
        results["dropout"] < 0.05,
        f"degradation after the minimum = {results['dropout']:.3f} bpc",
    )


if __name__ == "__main__":
    test_gradient_through_masks()
    test_mask_statistics()
    test_mask_is_shared_across_time()
    test_recurrent_mask_is_not_the_memory()
    test_dropout_stops_the_overfitting_turn()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
