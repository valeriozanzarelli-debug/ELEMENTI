#!/usr/bin/env python3
"""
Long-horizon diagnostics: how close empirical frequencies are to uniform
within time windows (decade × wheel). Useful to explore non-stationarity /
'regime' shifts — not for claiming predictability.

Reads data/draws_long.csv (run parse_draws.py first).
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

# 90 numbers on the wheel in modern archive
N_BINS = 90
MAX_ENTROPY = math.log2(N_BINS)


def chi2_uniform(counts: list[int], total: int) -> float:
    """Pearson chi-square vs discrete uniform on 1..90."""
    if total <= 0:
        return float("nan")
    e = total / N_BINS
    return sum((c - e) ** 2 / e for c in counts)


def shannon_entropy(counts: list[int], total: int) -> float:
    """Bits; upper bound log2(90) if mass spread evenly across all bins."""
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def load_events(long_csv: Path) -> list[tuple[int, str, tuple[int, int, int, int, int]]]:
    """(year, wheel, five numbers)."""
    out: list[tuple[int, str, tuple[int, int, int, int, int]]] = []
    with long_csv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            y = int(row["year"])
            w = row["wheel"]
            nums = []
            for k in ("n1", "n2", "n3", "n4", "n5"):
                v = row[k]
                if v == "" or v is None:
                    nums = []
                    break
                nums.append(int(v))
            if len(nums) != 5:
                continue
            out.append((y, w, tuple(nums)))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    long_csv = root / "data" / "draws_long.csv"
    if not long_csv.is_file():
        print("Missing data/draws_long.csv — run parse_draws.py first.", file=sys.stderr)
        return 1

    events = load_events(long_csv)
    # (decade_start, wheel) -> list of 90 counts
    buckets: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0] * N_BINS)

    for year, wheel, nums in events:
        decade = (year // 10) * 10
        key = (decade, wheel)
        for n in nums:
            if 1 <= n <= N_BINS:
                buckets[key][n - 1] += 1

    analysis_dir = root / "data" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / "uniformity_by_decade_wheel.csv"

    rows_out = []
    for (decade, wheel) in sorted(buckets.keys()):
        c = buckets[(decade, wheel)]
        total = sum(c)
        if total < N_BINS:
            # too sparse for meaningful chi2 vs 90 bins
            continue
        n_draws = total // 5
        chi2 = chi2_uniform(c, total)
        h = shannon_entropy(c, total)
        rows_out.append(
            {
                "decade_start": decade,
                "decade_label": f"{decade}s",
                "wheel": wheel,
                "n_draws": n_draws,
                "n_balls": total,
                "chi2_uniform": round(chi2, 4),
                "entropy_bits": round(h, 6),
                "max_entropy_bits": round(MAX_ENTROPY, 6),
                "entropy_ratio": round(h / MAX_ENTROPY, 6),
            }
        )

    fields = [
        "decade_start",
        "decade_label",
        "wheel",
        "n_draws",
        "n_balls",
        "chi2_uniform",
        "entropy_bits",
        "max_entropy_bits",
        "entropy_ratio",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} window rows -> {out_path}")
    print(
        "Interpret: high chi2 vs uniform = stronger departure in that decade+wheel; "
        "entropy_ratio near 1 = empirical mass more spread across 90 numbers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
