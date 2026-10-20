"""The reading of section 9, checked against every measured two-arm shape.

Reads ../logs/3070/grid13.txt and grid_rest.txt (NPP on every pair of arm heights 1-5, RTX 3070). The rule:
a shape with left arm l and right arm r splits exactly when REACH <= r <= l, where REACH is how far up the right
arm NPP re-points, one number per NPP version and shape (six in all, fitted by hand to these logs).

    python rule.py
"""
import re
from pathlib import Path

REACH = {"V": {"11": 3, "12": 2, "13": 2}, "U": {"11": 4, "12": 4, "13": 3}}
logs = Path(__file__).resolve().parent.parent / "logs" / "3070"
seen, agree = {}, 0
for f in ("grid13.txt", "grid_rest.txt"):
    for line in (logs / f).read_text().splitlines():
        m = re.fullmatch(r"(\d+) ([VU])_l(\d)_r(\d) broken links (\d+), false merges (\d+)", line.strip())
        if m:
            v, shape, l, r, broken, merges = m.groups()
            seen[v, shape, int(l), int(r)] = int(broken) + int(merges) > 0
for (v, shape, l, r), split in sorted(seen.items()):
    predicted = REACH[shape][v] <= r <= l
    agree += predicted == split
    if predicted != split:
        print(f"NPP {v} {shape} l={l} r={r}: measured {'split' if split else 'correct'}, rule says otherwise")
assert len(seen) == 150, len(seen)
print(f"rule agrees with {agree} of {len(seen)} measured shapes")
