"""Tests for the scaling-laws slice. Run with: python3 test_scaling.py

Everything here is arithmetic on curves whose truth is known: the exponents are recovered
from noisy data, the closed-form optimum is checked against an independent search, and the
allocation exponents are compared with their derivations.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import (  # noqa: E402
    allocation_exponent,
    brute_force_optimum,
    closed_form_optimum,
    fit_power_law,
    loss_joint,
    loss_size,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_recover_the_exponent():
    """Fit L = a N^-alpha on a noisy synthetic curve and recover alpha."""
    rng = np.random.default_rng(0)
    alpha_true, a_true = 0.076, 0.5
    ns = np.logspace(2, 7, 25)
    ys = loss_size(ns, a_true, alpha_true) * np.exp(rng.normal(0, 0.01, ns.size))
    a, alpha, r2 = fit_power_law(ns, ys)
    check("recovers alpha to within 1%", abs(alpha - alpha_true) / alpha_true < 0.01,
          f"{alpha:.5f} vs {alpha_true}")
    check("recovers the prefactor to within 2%", abs(a - a_true) / a_true < 0.02, f"{a:.4f} vs {a_true}")
    check("the fit is genuinely good", r2 > 0.99, f"r2 = {r2:.5f}")


def test_closed_form_matches_search():
    """The analytic optimum must agree with an independent numerical search."""
    for compute in (1e15, 1e18, 1e21):
        n1, d1 = closed_form_optimum(compute, 0.5, 0.076, 0.4, 0.095)
        n2, d2 = brute_force_optimum(compute, 0.5, 0.076, 0.4, 0.095)
        check(
            f"C = {compute:.0e}: closed form matches golden-section search",
            abs(n1 - n2) / n1 < 1e-5 and abs(d1 - d2) / d1 < 1e-5,
            f"N {n1:.4e} vs {n2:.4e}, D {d1:.4e} vs {d2:.4e}",
        )


def test_allocation_exponents():
    """N ~ C^(beta/(alpha+beta)) and D ~ C^(alpha/(alpha+beta)), measured from the optima."""
    alpha, beta = 0.076, 0.095
    cs = np.logspace(15, 22, 20)
    n_exp = allocation_exponent(cs, 0.5, alpha, 0.4, beta)
    check(
        "N's exponent in C matches beta/(alpha+beta)",
        abs(n_exp - beta / (alpha + beta)) < 1e-9,
        f"{n_exp:.6f} vs {beta / (alpha + beta):.6f}",
    )
    ns = [closed_form_optimum(c, 0.5, alpha, 0.4, beta)[0] for c in cs]
    ds = [closed_form_optimum(c, 0.5, alpha, 0.4, beta)[1] for c in cs]
    A = np.vstack([np.log(cs), np.ones_like(cs)]).T
    d_exp = float(np.linalg.lstsq(A, np.log(ds), rcond=None)[0][0])
    check(
        "D's exponent matches alpha/(alpha+beta)",
        abs(d_exp - alpha / (alpha + beta)) < 1e-9,
        f"{d_exp:.6f} vs {alpha / (alpha + beta):.6f}",
    )
    check("the two exponents sum to one", abs(n_exp + d_exp - 1.0) < 1e-9)


def test_bigger_models_are_more_sample_efficient():
    """At a fixed amount of data, more parameters means lower loss — the paper's point."""
    losses = [loss_joint(n, 1e6, 0.5, 0.076, 0.4, 0.095, 1.0) for n in (1e3, 1e4, 1e5, 1e6)]
    check(
        "loss falls as parameters grow at fixed data",
        all(b < a for a, b in zip(losses, losses[1:])),
        " ".join(f"{v:.3f}" for v in losses),
    )


if __name__ == "__main__":
    test_recover_the_exponent()
    test_closed_form_matches_search()
    test_allocation_exponents()
    test_bigger_models_are_more_sample_efficient()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
