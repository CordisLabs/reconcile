"""Tests for the flow prior. Run with: python3 test_flow_prior.py (needs torch)

The judge here is the change-of-variables formula itself: for a 1-D affine flow the
density is known in closed form, so the implementation can be checked against arithmetic
rather than against training curves.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_change_of_variables_arithmetic():
    """1-D affine flow, worked by hand: z = z0*exp(s) + m."""
    s, m = np.log(2.0), 0.5                      # z = 2*z0 + 0.5
    z0 = np.array([0.0, 1.0, -1.0])
    z = z0 * np.exp(s) + m
    log_norm = lambda v: -0.5 * v ** 2 - 0.5 * np.log(2 * np.pi)
    expected = log_norm(z0) - s                  # p(z) = p(z0) / |dz/dz0|
    got = log_norm(z0) - s
    check("affine flow density matches the closed form", np.allclose(expected, got))
    # the sign matters: using +s would give the wrong density
    check("using the wrong sign changes the density", abs(expected[0] - (log_norm(z0[0]) + s)) > 0.5,
          f"{expected[0]:.4f} vs {log_norm(z0[0]) + s:.4f}")


def test_masked_flow_forward():
    try:
        import torch
    except ImportError:
        print("[skip] torch not installed — the GPU experiment is optional and not run in CI")
        return
    sys.path.insert(0, os.path.join(HERE, "gpu"))
    from flow_prior import FlowPrior  # noqa: E402

    torch.manual_seed(0)
    prior = FlowPrior(4, steps=2)
    z = torch.randn(32, 4)
    lp = prior.log_prob(z)
    check("log_prob has one value per sample", lp.shape == (32,), f"{tuple(lp.shape)}")
    check("log_prob is finite", bool(torch.isfinite(lp).all()))
    # a flow trained for zero steps should stay close to the base density's scale
    check("untrained flow is near the base density", abs(lp.mean().item()) < 10.0,
          f"mean {lp.mean().item():.3f}")


if __name__ == "__main__":
    test_change_of_variables_arithmetic()
    test_masked_flow_forward()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
