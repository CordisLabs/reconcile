"""Tests for the NTM addressing slice. Run with: python3 test_ntm.py

The addressing weights are a probability distribution by construction, and each operation
has an exact property: sharpening converges to one-hot, a closed interpolating gate keeps
the previous weights bit for bit, and shifting is a circular convolution.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import (  # noqa: E402
    NTM,
    content_addressing,
    cosine_similarity,
    location_addressing,
    read,
    write,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_content_addressing_is_a_distribution():
    rng = np.random.default_rng(0)
    memory = rng.normal(size=(8, 4))
    w = content_addressing(memory, rng.normal(size=4), 1.0)
    check("weights sum to 1", abs(w.sum() - 1.0) < 1e-12, f"{w.sum():.12f}")
    check("weights are non-negative", bool((w >= 0).all()))


def test_sharpening_converges_to_the_nearest_row():
    rng = np.random.default_rng(1)
    memory = rng.normal(size=(8, 4))
    key = rng.normal(size=4)
    w = content_addressing(memory, key, 200.0)
    check(
        "large beta gives a one-hot at the most similar row",
        np.argmax(w) == np.argmax(cosine_similarity(memory, key)) and w.max() > 0.999,
        f"max weight {w.max():.6f}",
    )


def test_gate_zero_keeps_the_previous_weights():
    prev = np.array([0.1, 0.2, 0.3, 0.0, 0.4, 0.0, 0.0, 0.0])
    shift = np.eye(8)[0]
    out = location_addressing(prev, 0.0, shift, 1.0)
    check("gate = 0 with a zero shift reproduces prev exactly",
          np.allclose(out, prev, atol=1e-12), f"max |Δ| = {np.abs(out - prev).max():.2e}")


def test_shift_is_circular():
    prev = np.zeros(8)
    prev[0] = 1.0
    shift = np.zeros(8)
    shift[1] = 1.0
    out = location_addressing(prev, 0.0, shift, 1.0)
    expected = np.zeros(8)
    expected[1] = 1.0
    check("a one-hot shift moves the mass by one slot", np.allclose(out, expected), f"{out}")


def test_read_is_a_convex_combination():
    rng = np.random.default_rng(2)
    memory = rng.normal(size=(6, 3))
    w = np.ones(6) / 6.0
    r = read(memory, w)
    check("uniform weights give the memory mean", np.allclose(r, memory.mean(axis=0)),
          f"{r} vs {memory.mean(axis=0)}")


def test_write_with_zero_erase_and_add_is_a_no_op():
    rng = np.random.default_rng(3)
    memory = rng.normal(size=(6, 3))
    w = np.ones(6) / 6.0
    after = write(memory, w, np.zeros(3), np.zeros(3))
    check("zero erase and zero add leave memory untouched",
          np.allclose(after, memory, atol=1e-15), f"max |Δ| = {np.abs(after - memory).max():.2e}")


def test_erase_then_add_writes_exactly():
    """Writing 0 then 1 to an addressed row must set that row to 1, and only that row."""
    memory = np.zeros((4, 2))
    w = np.array([1.0, 0.0, 0.0, 0.0])
    after = write(memory, w, np.ones(2), np.ones(2))
    check("the addressed row becomes the added value",
          np.allclose(after[0], [1.0, 1.0]) and np.allclose(after[1:], 0.0), f"{after}")


def test_cell_round_trip():
    cell = NTM(slots=8, width=4, seed=0)
    before = cell.memory.copy()
    r, w = cell.read_step(np.ones(4), 5.0)
    check("read returns a width-vector", r.shape == (4,))
    check("read does not change memory", np.allclose(cell.memory, before))


if __name__ == "__main__":
    test_content_addressing_is_a_distribution()
    test_sharpening_converges_to_the_nearest_row()
    test_gate_zero_keeps_the_previous_weights()
    test_shift_is_circular()
    test_read_is_a_convex_combination()
    test_write_with_zero_erase_and_add_is_a_no_op()
    test_erase_then_add_writes_exactly()
    test_cell_round_trip()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
