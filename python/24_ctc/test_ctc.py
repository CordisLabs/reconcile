"""Tests for the CTC slice. Run with: python3 test_ctc.py

The judge is the definition: brute-force enumeration of all alignments. The dynamic
program must agree with it exactly, including on the cases that trip implementations up —
repeated labels, which need a blank between them, and the empty target.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import (  # noqa: E402
    beam_search, brute_force_loss, collapse, ctc_loss, ctc_loss_and_grad,
    greedy_decode, log_softmax,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def setup(T=5, V=3, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(T, V))
    return log_softmax(logits), V, T


def test_dp_matches_brute_force():
    log_probs, V, _ = setup()
    for target in ([1], [1, 2], [2, 1], [1, 1], [1, 2, 1], []):
        dp = ctc_loss(log_probs, target)
        bf = brute_force_loss(log_probs, target, V)
        check(
            f"DP == brute force for target {target}",
            abs(dp - bf) < 1e-9,
            f"DP {dp:.10f} vs brute {bf:.10f}",
        )


def test_repeats_need_a_blank():
    """Two identical labels cannot come from two frames without a blank between them."""
    rng = np.random.default_rng(1)
    log_probs = log_softmax(rng.normal(size=(2, 3)))
    inf_loss = ctc_loss(log_probs, [1, 1])
    check(
        "target [1,1] from T=2 is impossible",
        inf_loss == float("inf") or inf_loss > 1e10,
        f"loss = {inf_loss}",
    )
    # one extra frame makes it possible
    log_probs3 = log_softmax(rng.normal(size=(3, 3)))
    check("target [1,1] from T=3 is possible", ctc_loss(log_probs3, [1, 1]) < 1e10)


def test_probabilities_over_all_outputs_sum_to_one():
    """Every path collapses to exactly one label sequence, so the masses must sum to 1.

    This is the strongest check available: it exercises the collapse rule on all V^T
    paths at once and verifies the DP against each accumulated total.
    """
    from itertools import product

    rng = np.random.default_rng(2)
    T, V = 5, 3
    log_probs = log_softmax(rng.normal(size=(T, V)))
    masses = {}
    for path in product(range(V), repeat=T):
        seq = tuple(collapse(path))
        prob = float(np.exp(sum(log_probs[t, path[t]] for t in range(T))))
        masses[seq] = masses.get(seq, 0.0) + prob
    total = sum(masses.values())
    check("total probability over all outputs is 1", abs(total - 1.0) < 1e-9, f"{total:.12f}")
    worst = 0.0
    for seq, mass in masses.items():
        dp = ctc_loss(log_probs, list(seq))
        worst = max(worst, abs(dp - (-np.log(mass))))
    check(
        "DP matches the enumerated mass for every output sequence",
        worst < 1e-9,
        f"{len(masses)} sequences, max |Δ| = {worst:.2e}",
    )


def test_impossible_target():
    log_probs, _, _ = setup(T=3)
    check(
        "a target longer than the sequence is impossible",
        ctc_loss(log_probs, [1, 2, 1, 2]) > 1e10,
        f"{ctc_loss(log_probs, [1, 2, 1, 2])}",
    )


def test_gradient():
    """Analytic gradient of the CTC loss against finite differences, in all regimes."""
    rng = np.random.default_rng(3)
    worst = 0.0
    for T, V, target in [(4, 3, [1, 2]), (5, 4, [1, 1]), (3, 3, []), (3, 3, [1, 2, 1])]:
        logits = rng.normal(size=(T, V))
        _, grad = ctc_loss_and_grad(logits, target)
        eps = 1e-6
        for t in range(T):
            for k in range(V):
                o = logits[t, k]
                logits[t, k] = o + eps
                lp, _ = ctc_loss_and_grad(logits, target)
                logits[t, k] = o - eps
                lm, _ = ctc_loss_and_grad(logits, target)
                logits[t, k] = o
                worst = max(worst, abs(grad[t, k] - (lp - lm) / (2 * eps)))
    check("CTC gradient vs finite differences", worst < 1e-7, f"max_abs = {worst:.2e}")


def test_decoding():
    """Beam search cannot be worse than greedy: it searches a superset of its path."""
    rng = np.random.default_rng(4)
    log_probs = log_softmax(rng.normal(size=(8, 4)))
    g = greedy_decode(log_probs)
    b = beam_search(log_probs, beam=16)
    check("beam search finds a sequence at least as likely as greedy",
          ctc_loss(log_probs, b) <= ctc_loss(log_probs, g) + 1e-9,
          f"beam {-ctc_loss(log_probs, b):.4f} vs greedy {-ctc_loss(log_probs, g):.4f}")
    check("beam search returns no blanks", 0 not in b, f"{b}")

    # a confident model should be decoded back to the sequence it was built around
    T, V = 6, 3
    logits = np.full((T, V), -8.0)
    logits[:, 0] = 0.0
    logits[1, 1] = 8.0
    logits[2, 2] = 8.0
    lp = log_softmax(logits)
    check("confident model decodes to the intended sequence",
          beam_search(lp, beam=8) == [1, 2], f"{beam_search(lp, beam=8)}")


if __name__ == "__main__":
    test_dp_matches_brute_force()
    test_gradient()
    test_decoding()
    test_repeats_need_a_blank()
    test_probabilities_over_all_outputs_sum_to_one()
    test_impossible_target()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
