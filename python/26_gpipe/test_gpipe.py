"""Tests for the GPipe slice. Run with: python3 test_gpipe.py

The equivalence is exact and is checked bitwise. The memory claim is measured by running
the schedule, not estimated from a formula — an earlier closed-form guess in this file was
wrong and predicted a gain that grows without bound with the number of micro-batches.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import (  # noqa: E402
    Pipeline, backward_full, backward_micro, backward_recompute, peak_curve,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def setup():
    rng = np.random.default_rng(0)
    pipeline = Pipeline([16, 32, 32, 24, 8], seed=0)
    x = rng.normal(size=(64, 16))
    return pipeline, x


def test_output_is_bitwise_identical():
    """Pipeline parallel must not change the arithmetic: same output, every bit."""
    pipeline, x = setup()
    full_out, _ = pipeline.forward_full(x)
    for micro in (2, 4, 8, 16):
        out, _ = pipeline.forward_micro(x, micro)
        diff = float(np.abs(out - full_out).max())
        check(
            f"micro-batches = {micro} reproduce the full forward exactly",
            diff == 0.0,
            f"max|Δ| = {diff:.1e}",
        )


def test_peak_activation_memory_drops():
    pipeline, x = setup()
    rows = peak_curve(pipeline, x, (2, 4, 8, 16))
    full = rows[0][3]
    for micro, peak, _, _ in rows:
        ratio = full / peak
        check(
            f"micro = {micro} holds fewer activations than the full batch",
            peak < full,
            f"peak {peak} vs full {full} ({ratio:.2f}x)",
        )
    peaks = [r[1] for r in rows]
    check(
        "peak memory decreases monotonically with more micro-batches",
        all(b <= a for a, b in zip(peaks, peaks[1:])),
        f"{peaks}",
    )
    # the gain saturates near the pipeline depth, not at the micro-batch count
    best_ratio = full / peaks[-1]
    check(
        "the gain saturates rather than growing with micro-batches",
        best_ratio < rows[-1][0],
        f"16 micro-batches gave only {best_ratio:.2f}x, not 16x",
    )


def test_batch_must_divide_evenly():
    pipeline, x = setup()
    ok = False
    try:
        pipeline.forward_micro(x, 7)
    except ValueError:
        ok = True
    check("a non-dividing split is rejected", ok)


def test_backward_matches_numerically():
    """The analytic gradient through the pipeline against finite differences."""
    pipeline, _ = setup()
    rng = np.random.default_rng(7)
    x = rng.normal(size=(16, 16))
    grads = backward_full(pipeline, x)
    eps = 1e-6
    worst = 0.0
    for k in range(len(pipeline.stages)):
        W, _b = pipeline.stages[k]
        for (i, j) in [(0, 0), (2, 3)]:
            orig = W[i, j]
            W[i, j] = orig + eps
            plus = pipeline.forward_full(x)[0].sum()
            W[i, j] = orig - eps
            minus = pipeline.forward_full(x)[0].sum()
            W[i, j] = orig
            worst = max(worst, abs((plus - minus) / (2 * eps) - grads[k][0][i, j]))
    check("pipeline gradient vs finite differences", worst < 1e-7, f"max_abs = {worst:.2e}")


def test_micro_gradients_match_full():
    """Splitting the batch must not change the gradient, only how it is accumulated."""
    pipeline, x = setup()
    g_full = backward_full(pipeline, x)
    for micro in (2, 4, 8):
        g_micro = backward_micro(pipeline, x, micro)
        worst = max(max(float(np.abs(a[0] - b[0]).max()), float(np.abs(a[1] - b[1]).max()))
                    for a, b in zip(g_full, g_micro))
        check(f"micro = {micro} gradient matches the full batch", worst < 1e-12,
              f"max |Δ| = {worst:.2e}")


def test_recomputation_matches_and_saves():
    """Recomputing activations must give the same gradient while storing less."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(64, 32))
    peaks = []
    for K in (4, 8, 16):
        pipeline = Pipeline([32] * (K + 1), seed=0)
        g_full = backward_full(pipeline, x)
        g_rec, peak = backward_recompute(pipeline, x)
        worst = max(max(float(np.abs(a[0] - b[0]).max()), float(np.abs(a[1] - b[1]).max()))
                    for a, b in zip(g_full, g_rec))
        check(f"K={K}: recomputed gradient matches", worst < 1e-12, f"max |Δ| = {worst:.2e}")
        peaks.append(peak)
    check("stored activations do not grow with depth",
          peaks[0] == peaks[-1], f"{peaks}")


if __name__ == "__main__":
    test_output_is_bitwise_identical()
    test_recomputation_matches_and_saves()
    test_backward_matches_numerically()
    test_micro_gradients_match_full()
    test_peak_activation_memory_drops()
    test_batch_must_divide_evenly()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
