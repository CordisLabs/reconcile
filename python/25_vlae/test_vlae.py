"""Tests for the VLAE slice. Run with: python3 test_vlae.py

The paper is motivated by posterior collapse, so that is what is measured: as the KL weight
grows, the latent carries less information (lower KL, fewer active dimensions) and the
reconstruction gets worse. The paper's own fix — a lossy code with an autoregressive
decoder, and an IAF prior — is not implemented.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from implementation import VAE, make_data, train_vae  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def sweep(steps=300):
    data = make_data()
    rows = []
    for beta in (0.01, 0.1, 1.0, 10.0):
        model = VAE(data.shape[1], hidden=32, latent=8, seed=0)
        kl_dim, rec = train_vae(model, data, beta, steps=steps, seed=0)
        rows.append((beta, float(kl_dim.sum()), int((kl_dim > 0.01).sum()), rec))
    return rows


def test_kl_is_non_negative():
    data = make_data()
    model = VAE(data.shape[1], latent=8, seed=0)
    kl_dim, _ = train_vae(model, data, 1.0, steps=50, seed=0)
    check("every KL dimension is non-negative", bool((kl_dim >= 0).all()),
          f"min {kl_dim.min():.2e}")


def test_posterior_collapse_with_beta():
    rows = sweep()
    print("    beta   KL      active   reconstruction")
    for beta, kl, active, rec in rows:
        print(f"    {beta:<5}  {kl:6.3f}  {active}/8     {rec:6.3f}")
    kls = [r[1] for r in rows]
    check("KL decreases monotonically as beta grows",
          all(b < a for a, b in zip(kls, kls[1:])), f"{[round(k, 3) for k in kls]}")
    check("high beta collapses the latent", rows[-1][1] < 0.2 * rows[0][1],
          f"KL {rows[0][1]:.3f} -> {rows[-1][1]:.3f}")
    actives = [r[2] for r in rows]
    check("fewer dimensions stay active", actives[-1] < actives[0], f"{actives}")
    recs = [r[3] for r in rows]
    check("reconstruction gets worse as the latent is squeezed",
          recs[-1] > recs[0] * 1.1, f"{recs[0]:.3f} -> {recs[-1]:.3f}")


if __name__ == "__main__":
    test_kl_is_non_negative()
    test_posterior_collapse_with_beta()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
