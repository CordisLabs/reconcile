"""Numerical gradient checking used by every Python-track slice.

Plain NumPy, no test framework. Every slice asserts against a *finite difference*
estimate instead of against "it runs", which is the only way an implementation can
claim correctness without consulting someone else's code.
"""

import numpy as np


def numeric_grad(f, params, eps=1e-6):
    """Central-difference gradient of scalar f() w.r.t. every entry of `params`.

    `f` must recompute its forward pass from `params` on each call, so the
    parameters are perturbed in place and restored.
    """
    grads = {}
    for name, value in params.items():
        g = np.zeros_like(value)
        flat = value.reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            f_plus = f()
            flat[i] = orig - eps
            f_minus = f()
            flat[i] = orig
            g.reshape(-1)[i] = (f_plus - f_minus) / (2.0 * eps)
        grads[name] = g
    return grads


def max_error(analytic, numeric, floor=1e-6):
    """Return (max_abs, max_rel, n_below_floor).

    ``max_rel`` is computed only over entries whose magnitude exceeds ``floor``.
    Entries below it are structural zeros (or numerical zeros); their relative error
    is dominated by the finite-difference roundoff and carries no information, so
    they are counted separately instead of being folded into the ratio.
    """
    max_abs = 0.0
    max_rel = 0.0
    n_below_floor = 0
    for name, a in analytic.items():
        n = numeric[name]
        diff = np.abs(a - n)
        max_abs = max(max_abs, float(diff.max()))
        scale = np.abs(a) + np.abs(n)
        informative = scale > floor
        n_below_floor += int((~informative).sum())
        if informative.any():
            max_rel = max(max_rel, float((diff[informative] / scale[informative]).max()))
    return max_abs, max_rel, n_below_floor


def check_gradients(f, params, analytic, eps=1e-6, atol=1e-7, rtol=1e-4):
    """Compare analytic gradients against finite differences.

    The pass/fail criterion is ``|a - n| <= atol + rtol * |n|``, not a pure relative
    error. Some gradients are *structurally zero* — e.g. the key bias of attention is
    exactly 0 because every row of dS sums to zero (softmax is invariant to shifting a
    row of scores by a constant). Against such an entry the finite difference returns
    pure roundoff (~1e-10) while the analytic value is ~1e-17, so a relative-error
    criterion reports 1.0 and fails a perfectly correct implementation.

    Returns (max_abs, max_rel, ok).
    """
    numeric = numeric_grad(f, params, eps=eps)
    max_abs, max_rel, _ = max_error(analytic, numeric)
    ok = all(
        np.allclose(analytic[name], numeric[name], atol=atol, rtol=rtol)
        for name in analytic
    )
    return max_abs, max_rel, ok
