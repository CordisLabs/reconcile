#!/usr/bin/env python3
"""Check that the slice directories and SYLLABUS.md agree.

Two ways they can drift, both of which have happened in this repository:

  * a slice exists and passes its tests but the syllabus row was never marked done;
  * a syllabus row is marked done but the slice directory is missing.

The mapping is by number: `python/07_char_rnn` is row 07, `rust/m01_mdl` is row 01.

Run: python3 scripts/check_consistency.py    (also part of `make test`)
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def slice_numbers():
    found = {}
    for track, pattern in (("python", "python"), ("rust", "rust")):
        base = os.path.join(ROOT, track)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if not os.path.isdir(path) or name in ("_shared", "target"):
                continue
            m = re.match(r"^m?(\d\d)_", name)
            if m:
                found[m.group(1)] = f"{track}/{name}"
    return found


def syllabus_rows():
    path = os.path.join(ROOT, "SYLLABUS.md")
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"^\| (\d\d) \|", line)
            if m:
                rows[m.group(1)] = {"done": "✅" in line, "line": line.rstrip()}
    return rows


def main():
    slices = slice_numbers()
    rows = syllabus_rows()
    problems = []

    for num, path in slices.items():
        row = rows.get(num)
        if row is None:
            problems.append(f"slice {path}: no syllabus row {num}")
        elif not row["done"]:
            problems.append(f"slice {path}: syllabus row {num} is not marked done")

    for num, row in rows.items():
        if row["done"] and num not in slices:
            problems.append(f"syllabus row {num} is marked done but no slice directory exists")

    total = len(rows)
    print(f"slices: {len(slices)}, syllabus rows: {total}, marked done: {sum(r['done'] for r in rows.values())}")
    if problems:
        for p in problems:
            print(f"  MISMATCH: {p}")
        return 1
    print("directories and SYLLABUS agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
