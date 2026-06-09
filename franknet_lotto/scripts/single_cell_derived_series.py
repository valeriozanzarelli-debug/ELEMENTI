#!/usr/bin/env python3
"""
Build derived sequences for ONE wheel + position (a single "casella"),
from a minimum year onward — for exploring temporal structure in:
  - signed / absolute / circular differences between consecutive draws
  - sum of decimal digits (once) and digital root
  - complement to 90 (90 - n), plus digit_sum / digital_root of that complement

Also writes n-gram counts (on discretized diff) and lag autocorrelations.

Uses only the standard library. Run from repo root or any cwd (script locates data/).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

# Keep in sync with parse_draws.py
WHEEL_ORDER = [
    "BARI",
    "CAGLIARI",
    "FIRENZE",
    "GENOVA",
    "MILANO",
    "NAPOLI",
    "PALERMO",
    "ROMA",
    "TORINO",
    "VENEZIA",
    "NAZIONALE",
]


def digit_sum_once(n: int) -> int:
    return sum(int(c) for c in str(n))


def digital_root(n: int) -> int:
    while n > 9:
        n = digit_sum_once(n)
    return n


def circ_dist(a: int, b: int) -> int:
    """Shortest arc length on labels 1..90 treated as a cycle (90 adjacent to 1)."""
    d = abs(a - b)
    return min(d, 90 - d)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / math.sqrt(sxx * syy)


def load_column(
    long_csv: Path, wheel: str, position: int, from_year: int
) -> list[dict]:
    """Rows sorted by time: one value per draw for that wheel."""
    if wheel not in WHEEL_ORDER:
        raise SystemExit(f"Unknown wheel {wheel}. Choose one of {WHEEL_ORDER}")
    if not 1 <= position <= 5:
        raise SystemExit("position must be 1..5 (n1..n5)")
    col = f"n{position}"
    rows: list[dict] = []
    with long_csv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["wheel"] != wheel:
                continue
            if int(row["year"]) < from_year:
                continue
            v = row[col]
            if v == "":
                continue
            n = int(v)
            rows.append(
                {
                    "year": int(row["year"]),
                    "date": row["date"],
                    "draw_index_year": int(row["draw_index_year"]),
                    "n": n,
                }
            )
    rows.sort(key=lambda x: (x["date"], x["draw_index_year"]))
    return rows


def build_series(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    n_prev: int | None = None
    for i, r in enumerate(rows):
        n = r["n"]
        ds = digit_sum_once(n)
        dr = digital_root(n)
        comp = 90 - n
        cds = digit_sum_once(comp)
        cdr = digital_root(comp)
        diff_s: int | None = None
        diff_a: int | None = None
        cd: int | None = None
        if n_prev is not None:
            diff_s = n - n_prev
            diff_a = abs(diff_s)
            cd = circ_dist(n, n_prev)
        rec = {
            "idx": i,
            "year": r["year"],
            "date": r["date"],
            "draw_index_year": r["draw_index_year"],
            "n": n,
            "diff_signed": diff_s if diff_s is not None else "",
            "diff_abs": diff_a if diff_a is not None else "",
            "circ_dist": cd if cd is not None else "",
            "digit_sum": ds,
            "digital_root": dr,
            "complement_90": comp,
            "complement_digit_sum": cds,
            "complement_digital_root": cdr,
        }
        out.append(rec)
        n_prev = n
    return out


def add_lag_columns(series: list[dict], lag: int, keys: list[str]) -> None:
    for i, rec in enumerate(series):
        for k in keys:
            j = i - lag
            rec[f"{k}_lag{lag}"] = series[j][k] if j >= 0 else ""


def ngrams_int(seq: list[int], k: int) -> Counter[tuple[int, ...]]:
    c: Counter[tuple[int, ...]] = Counter()
    for i in range(len(seq) - k + 1):
        c[tuple(seq[i : i + k])] += 1
    return c


def write_top_ngrams(
    path: Path, counter: Counter, title: str, top: int = 80
) -> None:
    lines = [f"# {title}", "# rank,count,tuple"]
    for rank, (tup, cnt) in enumerate(counter.most_common(top), 1):
        lines.append(f"{rank},{cnt},\"{list(tup)}\"")
    path.write_text("\n".join(lines), encoding="utf-8")


def autocorr_lags(values: list[float | None], max_lag: int) -> list[tuple[int, float | None]]:
    """Pearson corr between v[t] and v[t+lag] (lag >= 1)."""
    # strip to list with indices where value is float
    clean = [(i, float(v)) for i, v in enumerate(values) if v is not None]
    if len(clean) < 10:
        return [(k, None) for k in range(1, max_lag + 1)]
    results: list[tuple[int, float | None]] = []
    for lag in range(1, max_lag + 1):
        xs: list[float] = []
        ys: list[float] = []
        idx_map = {i: val for i, val in clean}
        for i, val in clean:
            j = i + lag
            if j in idx_map:
                xs.append(val)
                ys.append(idx_map[j])
        results.append((lag, pearson(xs, ys)))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Derived series for one wheel position.")
    ap.add_argument("--wheel", default="MILANO", help="Wheel name, e.g. MILANO")
    ap.add_argument("--position", type=int, default=1, help="1..5 = n1..n5")
    ap.add_argument("--from-year", type=int, default=1900)
    ap.add_argument("--lag", type=int, default=5, help="Extra lag columns (e.g. 5 draws)")
    ap.add_argument("--max-autocorr-lag", type=int, default=40)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    long_csv = root / "data" / "draws_long.csv"
    if not long_csv.is_file():
        print("Missing data/draws_long.csv — run parse_draws.py first.", file=sys.stderr)
        return 1

    rows = load_column(long_csv, args.wheel, args.position, args.from_year)
    if len(rows) < 2:
        print("Not enough rows for this filter.", file=sys.stderr)
        return 1

    series = build_series(rows)
    lag = max(1, args.lag)
    add_lag_columns(
        series,
        lag,
        [
            "n",
            "diff_signed",
            "digit_sum",
            "complement_90",
            "complement_digit_sum",
        ],
    )

    out_dir = root / "data" / "analysis" / "single_cell"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.wheel}_pos{args.position}_y{args.from_year}"
    csv_path = out_dir / f"{tag}_series.csv"

    fieldnames = list(series[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(series)

    # Integer diff stream for n-grams (skip first empty)
    diffs = [r["diff_signed"] for r in series if r["diff_signed"] != ""]
    diffs_i = [int(x) for x in diffs]

    for k, name in [(2, "bigram"), (3, "trigram"), (4, "4gram")]:
        ctr = ngrams_int(diffs_i, k)
        write_top_ngrams(
            out_dir / f"{tag}_diffsigned_{name}s.csv",
            ctr,
            f"{name} counts on signed diff (wheel={args.wheel} pos={args.position})",
        )

    for k, name in [(2, "bigram"), (3, "trigram")]:
        comp_seq = [r["complement_90"] for r in series]
        ctr = ngrams_int(comp_seq, k)
        write_top_ngrams(
            out_dir / f"{tag}_complement90_{name}s.csv",
            ctr,
            f"{name} counts on complement_90",
        )

    for k, name in [(2, "bigram"), (3, "trigram")]:
        cds_seq = [r["complement_digit_sum"] for r in series]
        ctr = ngrams_int(cds_seq, k)
        write_top_ngrams(
            out_dir / f"{tag}_complement_digitsum_{name}s.csv",
            ctr,
            f"{name} counts on digit_sum(90-n)",
        )

    ds_seq = [r["digit_sum"] for r in series]
    for k, name in [(2, "bigram"), (3, "trigram")]:
        ctr = ngrams_int(ds_seq, k)
        write_top_ngrams(
            out_dir / f"{tag}_digitsum_{name}s.csv",
            ctr,
            f"{name} counts on digit_sum (single pass)",
        )

    # Autocorrelation: raw n, diff_signed, digit_sum, complement
    def col_to_floats(key: str) -> list[float | None]:
        outv: list[float | None] = []
        for r in series:
            v = r[key]
            outv.append(float(v) if v != "" else None)
        return outv

    autocorr_report: dict = {"wheel": args.wheel, "position": args.position, "from_year": args.from_year, "n_rows": len(series), "series_csv": csv_path.name}
    for key in (
        "n",
        "diff_signed",
        "digit_sum",
        "complement_90",
        "digital_root",
        "complement_digit_sum",
    ):
        vals = col_to_floats(key)
        pairs = autocorr_lags(vals, args.max_autocorr_lag)
        autocorr_report[f"autocorr_{key}"] = [
            {"lag": L, "pearson_r": (round(r, 6) if r is not None else None)}
            for L, r in pairs
        ]

    json_path = out_dir / f"{tag}_autocorrelation.json"
    json_path.write_text(json.dumps(autocorr_report, indent=2), encoding="utf-8")

    print(f"Series rows: {len(series)} -> {csv_path}")
    print(f"Top n-gram tables and {json_path.name} in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
