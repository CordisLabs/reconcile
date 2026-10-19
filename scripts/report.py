#!/usr/bin/env python3
"""Print the measured numbers from every slice.

Each slice's test file already prints its results as "[ok  ] what  (number)". This runs
them and collects those lines, so "what did this repository actually measure" is one
command instead of twenty-three.

Run: python3 scripts/report.py [--slice 16]
"""

import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINE = re.compile(r"^\[ok\s*\]\s*(.+)$")


def python_slices(only=None):
    for test in sorted(glob.glob(os.path.join(ROOT, "python", "*", "test_*.py"))):
        name = os.path.basename(os.path.dirname(test))
        if only and only not in name:
            continue
        yield name, test


def run(test):
    try:
        out = subprocess.run(
            [sys.executable, test], capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(test),
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default=None, help="substring of the slice directory name")
    args = ap.parse_args()

    total = 0
    for name, test in python_slices(args.slice):
        out, err = run(test)
        if err:
            print(f"\n## {name}  ({err})")
            continue
        lines = [m.group(1) for m in (LINE.match(l) for l in out.stdout.splitlines()) if m]
        print(f"\n## {name}  ({len(lines)} checked)")
        for line in lines:
            print(f"  {line}")
        total += len(lines)
        if out.returncode != 0:
            print("  !! test suite failed")
    print(f"\n{total} checked lines from {len(list(python_slices(args.slice)))} slices")


if __name__ == "__main__":
    main()
