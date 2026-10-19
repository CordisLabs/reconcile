"""Tests for the dilated-convolution slice. Run with: python3 test_dilated.py

Everything here is arithmetic the paper states, so the judge is exact equality rather
than a tolerance on a benchmark.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import (  # noqa: E402
    conv2d_same,
    context_module,
    dilated_conv2d,
    parameter_count,
    receptive_field,
    spread_kernel,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_dilation_is_a_spread_kernel():
    """A dilated conv is a normal conv with zeros interleaved into the kernel."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(23, 23))
    kernel = rng.normal(size=(3, 3))
    for d in (1, 2, 4, 8):
        a = dilated_conv2d(x, kernel, d)
        b = conv2d_same(x, spread_kernel(kernel, d))
        err = float(np.abs(a - b).max())
        check(f"dilation {d}: dilated conv == spread-kernel conv", err == 0.0, f"max|Δ| = {err}")


def test_receptive_field_matches_figure_one():
    """Figure 1: dilations 1,2,4 give 3x3, 7x7, 15x15."""
    expected = {1: 3, 2: 7, 3: 15}
    for n, want in expected.items():
        dilations = [2 ** i for i in range(n)]
        got = receptive_field(dilations, canvas=161)
        check(
            f"stack {dilations} gives a {want}x{want} receptive field",
            got == (want, want),
            f"measured {got}",
        )


def test_context_module_is_67():
    """The paper's basic context module: 7 layers, dilations 1,1,2,4,8,16,1 -> 67x67.

    The paper then pads by 33 pixels, which is exactly (67-1)/2. This is one of the few
    paper numbers that can be checked end to end.
    """
    dilations = [1, 1, 2, 4, 8, 16, 1]
    got = receptive_field(dilations, canvas=321)
    check("basic context module has a 67x67 receptive field", got == (67, 67), f"measured {got}")
    check("the paper's 33-pixel padding is (67-1)/2", (67 - 1) // 2 == 33)


def test_exponential_growth_formula():
    """Size after n layers of dilations 1,2,4,... is 2^(n+1)-1."""
    for n in (1, 2, 3, 4):
        dilations = [2 ** i for i in range(n)]
        got = receptive_field(dilations, canvas=161)[0]
        want = 2 ** (n + 1) - 1
        check(f"{n} layers -> {want}x{want}", got == want, f"measured {got}")


def test_coverage_has_no_holes():
    """'without loss of coverage': every position inside the field contributes."""
    canvas = 161
    x = np.zeros((canvas, canvas))
    x[canvas // 2, canvas // 2] = 1.0
    y = context_module(x, [1, 2, 4])
    rows = np.nonzero(np.any(y != 0, axis=1))[0]
    cols = np.nonzero(np.any(y != 0, axis=0))[0]
    block = y[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    check(
        "no holes inside the 15x15 field",
        bool((block != 0).all()),
        f"{block.size} interior positions, {int((block != 0).sum())} nonzero",
    )


def test_parameters_grow_linearly():
    """Exponential receptive field, linear parameters."""
    dilations = [1, 1, 2, 4, 8, 16, 1]
    params = parameter_count(dilations)
    dense = 67 * 67
    check(
        "7 layers cost 63 parameters where one 67x67 kernel costs 4489",
        params == 63 and dense == 4489,
        f"{params} vs {dense}",
    )


def test_resolution_is_preserved():
    """No pooling or subsampling: output shape equals input shape."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(19, 23))
    y = context_module(x, [1, 2, 4])
    check("output keeps the input resolution", y.shape == x.shape, f"{x.shape} -> {y.shape}")


if __name__ == "__main__":
    test_dilation_is_a_spread_kernel()
    test_receptive_field_matches_figure_one()
    test_context_module_is_67()
    test_exponential_growth_formula()
    test_coverage_has_no_holes()
    test_parameters_grow_linearly()
    test_resolution_is_preserved()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
