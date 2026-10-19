"""Scaling Laws for Neural Language Models (Kaplan et al., 2020).

The paper's claims are fits and a constrained optimisation, so they are checkable
without training anything:

  * loss follows a power law in model size, dataset size and compute;
  * given a compute budget C, with C = 6ND, there is a closed-form optimal split between
    parameters N and data D, and it matches brute-force minimisation;
  * the optimal N and D themselves follow power laws in C.

The synthetic curves here use exponents of the same order as the paper's (alpha ~ 0.076
for size, beta ~ 0.095 for data), with multiplicative noise, so recovering them is a real
fit rather than a tautology.
"""

import numpy as np


# --------------------------------------------------------------------------
# the functional forms
# --------------------------------------------------------------------------

def loss_size(n, a, alpha, l_inf=0.0):
    """L(N) = a / N^alpha + L_inf — the paper's form for size-only fits."""
    return a * n ** (-alpha) + l_inf


def loss_data(d, b, beta, l_inf=0.0):
    return b * d ** (-beta) + l_inf


def loss_joint(n, d, a, alpha, b, beta, l_inf):
    """L(N, D) = a/N^alpha + b/D^beta + L_inf, the additive form the paper fits."""
    return a * n ** (-alpha) + b * d ** (-beta) + l_inf


def fit_power_law(xs, ys):
    """Least squares on log y = log a - alpha log x. Returns (a, alpha, r2)."""
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(ys, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ np.array([slope, intercept])
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(np.exp(intercept)), float(-slope), r2


# --------------------------------------------------------------------------
# compute-optimal allocation
# --------------------------------------------------------------------------

def closed_form_optimum(compute, a, alpha, b, beta):
    """N and D minimising a/N^alpha + b/D^beta subject to 6 N D = C.

    Substituting D = C/(6N) and setting the derivative to zero gives

        N^{alpha+beta} = (alpha a) / (beta b) * (C/6)^beta
        N ~ C^{beta/(alpha+beta)},   D ~ C^{alpha/(alpha+beta)}
    """
    k = compute / 6.0
    n = ((alpha * a) / (beta * b)) ** (1.0 / (alpha + beta)) * k ** (beta / (alpha + beta))
    d = k / n
    return n, d


def brute_force_optimum(compute, a, alpha, b, beta, iters=200):
    """Independent check: golden-section search on log N.

    A fixed linear grid is useless here — the optimum moves across ten orders of
    magnitude as compute grows, and an earlier grid-based version of this function simply
    ran into its own boundary and returned the edge. Searching in log space with a wide
    bracket avoids that.
    """
    k = compute / 6.0

    def loss_at(log_n):
        n = 10.0 ** log_n
        return a * n ** (-alpha) + b * (k / n) ** (-beta)

    lo, hi = -5.0, 20.0
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = loss_at(c), loss_at(d)
    for _ in range(iters):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = loss_at(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = loss_at(d)
    n = 10.0 ** (0.5 * (lo + hi))
    return n, k / n


def allocation_exponent(compute_values, a, alpha, b, beta):
    """Slope of log N_opt against log C — the paper's 'N ~ C^x' exponent."""
    ns = [closed_form_optimum(c, a, alpha, b, beta)[0] for c in compute_values]
    cs = np.asarray(compute_values, dtype=float)
    lc = np.log(cs)
    ln = np.log(np.asarray(ns))
    A = np.vstack([lc, np.ones_like(lc)]).T
    (slope, _), *_ = np.linalg.lstsq(A, ln, rcond=None)
    return float(slope)
