"""Tests for the TSP extension. Run with: python3 test_tsp.py

The judge is brute force: the optimal tour of a small point set is computable exactly, so
both the tour length and the symmetry handling can be checked without training.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tsp import canonical, optimal_tour, tour_length  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_optimal_tour_on_a_square():
    """Four points on a square: the optimal tour is the perimeter, 4 units."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    order, length = optimal_tour(pts)
    check("optimal tour of the unit square is the perimeter",
          abs(length - 4.0) < 1e-9, f"{length:.6f}")
    check("the tour visits every point once", sorted(order) == [0, 1, 2, 3], f"{order}")


def test_optimal_tour_beats_a_crossing_tour():
    """A crossing tour is never optimal: swapping the segments shortens it."""
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    _order, best = optimal_tour(pts)
    crossing = [0, 1, 2, 3]                       # 0-1 and 2-3 cross
    check("optimal is at least as short as a known crossing tour",
          best <= tour_length(pts, crossing) + 1e-12,
          f"{best:.4f} <= {tour_length(pts, crossing):.4f}")


def test_canonical_ignores_rotation_and_direction():
    base = [0, 1, 2, 3]
    rotated = [2, 3, 0, 1]
    reversed_ = [0, 3, 2, 1]
    check("rotation is removed", canonical(rotated) == canonical(base), f"{canonical(rotated)}")
    check("direction is removed", canonical(reversed_) == canonical(base), f"{canonical(reversed_)}")
    check("a different tour stays different", canonical([0, 2, 1, 3]) != canonical(base))


if __name__ == "__main__":
    test_optimal_tour_on_a_square()
    test_optimal_tour_beats_a_crossing_tour()
    test_canonical_ignores_rotation_and_direction()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all tests passed")
