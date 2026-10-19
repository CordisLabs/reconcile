"""Tests for the CS231n slice. Run with: python3 test_cs231n.py

Two independent judges, both arithmetic:
  * im2col convolution must equal the naive loop version elementwise;
  * VGG16's parameter count, computed from the architecture table, must equal the
    course's published 138,357,544.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import (  # noqa: E402
    VGG16_CONV,
    conv2d_im2col,
    conv2d_naive,
    conv_output_size,
    maxpool_backward,
    maxpool_forward,
    vgg16_params,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_im2col_equals_naive():
    """The fast path and the obvious path must agree exactly."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 12, 14))
    W = rng.normal(size=(5, 3, 3, 3))
    b = rng.normal(size=5)
    for stride, pad in [(1, 0), (2, 1), (1, 2)]:
        a = conv2d_naive(x, W, b, stride, pad)
        c = conv2d_im2col(x, W, b, stride, pad)
        err = float(np.abs(a - c).max())
        check(
            f"im2col == naive (stride {stride}, pad {pad})",
            err < 1e-12,
            f"max|Δ| = {err:.1e}, shape {a.shape}",
        )


def test_im2col_equals_naive_other_kernels():
    """1x1 and 5x5 kernels: the rearrangement must not assume 3x3."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 9, 11))
    for k in (1, 3, 5):
        W = rng.normal(size=(4, 2, k, k))
        a = conv2d_naive(x, W, None, 1, 0)
        c = conv2d_im2col(x, W, None, 1, 0)
        check(f"im2col == naive for {k}x{k} kernel", float(np.abs(a - c).max()) < 1e-12)


def test_output_size_formula():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 17, 19))
    for k, stride, pad in [(3, 1, 1), (3, 2, 1), (5, 1, 0), (1, 1, 0)]:
        W = rng.normal(size=(2, 1, k, k))
        out = conv2d_im2col(x, W, None, stride, pad)
        h = conv_output_size(17, k, stride, pad)
        w = conv_output_size(19, k, stride, pad)
        check(
            f"output size formula (k={k}, s={stride}, p={pad})",
            out.shape == (2, h, w),
            f"{out.shape} vs {(2, h, w)}",
        )


def test_pooling_forward():
    """A known 4x4 input gives a known 2x2 output; no ambiguity about the max."""
    x = np.arange(16, dtype=float).reshape(1, 4, 4)
    out, argmax = maxpool_forward(x, 2, 2)
    check("max-pool of arange(16) is [5, 7, 13, 15]", np.array_equal(out.ravel(), [5, 7, 13, 15]), f"{out.ravel()}")
    check("argmax points at the bottom-right of each window", np.array_equal(argmax.ravel(), [3, 3, 3, 3]))


def test_pooling_backward_routes_to_argmax():
    """Gradient must go to exactly the winning position, one unit each."""
    x = np.arange(16, dtype=float).reshape(1, 4, 4)
    _, argmax = maxpool_forward(x, 2, 2)
    dx = maxpool_backward(np.ones((1, 2, 2)), argmax, x.shape, 2, 2)
    expected = np.zeros_like(x)
    for c in range(1):
        for i in range(2):
            for j in range(2):
                di, dj = divmod(int(argmax[c, i, j]), 2)
                expected[c, i * 2 + di, j * 2 + dj] += 1.0
    check("pooling backward matches the routed positions", np.array_equal(dx, expected), f"sum {dx.sum():.0f}")


def test_vgg16_parameter_count():
    """138,357,544 is the number in the course notes; recompute it from the table."""
    total = vgg16_params()
    check(
        "VGG16 has 138,357,544 parameters",
        total == 138_357_544,
        f"computed {total:,}",
    )
    n_conv = sum(c for _, c in VGG16_CONV)
    check("VGG16 has 13 convolutional layers", n_conv == 13, f"{n_conv} conv layers")


if __name__ == "__main__":
    test_im2col_equals_naive()
    test_im2col_equals_naive_other_kernels()
    test_output_size_formula()
    test_pooling_forward()
    test_pooling_backward_routes_to_argmax()
    test_vgg16_parameter_count()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
