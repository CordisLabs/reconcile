"""Tests for the n-gram KL measurement. Run with: python3 test_ngram.py

The judge is arithmetic: KL of a distribution against itself is zero, and the n-gram
probabilities of a string must sum to one.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ngram_kl import kl, ngram_probs  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_kl_of_a_distribution_with_itself():
    text = "the quick brown fox jumps over the lazy dog"
    p = ngram_probs(text, 2)
    check("KL(p || p) = 0", abs(kl(p, p)) < 1e-12, f"{kl(p, p):.2e}")


def test_kl_is_positive_for_different_texts():
    a = ngram_probs("abcabcabcabc", 2)
    b = ngram_probs("xyzxyzxyzxyz", 2)
    check("KL(p || q) > 0 for different texts", kl(a, b) > 1.0, f"{kl(a, b):.3f}")


def test_probabilities_sum_to_one():
    text = "hello world " * 20
    for n in (1, 2, 3):
        total = sum(ngram_probs(text, n).values())
        check(f"{n}-gram probabilities sum to 1", abs(total - 1.0) < 1e-9, f"{total:.12f}")


def test_trained_model_moves_toward_the_corpus():
    """The curve must descend: later checkpoints fit higher-order statistics better."""
    from ngram_kl import main as curve
    rows = curve()
    check("trigram KL falls over training", rows[-1][2] < rows[0][2],
          f"{rows[0][2]:.3f} -> {rows[-1][2]:.3f}")
    check("unigram KL is already near zero at the first checkpoint", rows[0][0] < 0.5,
          f"{rows[0][0]:.3f}")


if __name__ == "__main__":
    test_kl_of_a_distribution_with_itself()
    test_kl_is_positive_for_different_texts()
    test_probabilities_sum_to_one()
    test_trained_model_moves_toward_the_corpus()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
