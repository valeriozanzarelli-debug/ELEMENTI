#!/usr/bin/env python3
"""
Search for exploitable structure inside a long derived series CSV
(produced by single_cell_derived_series.py).

Methods (all on chronological train/test split to limit overfitting):
  - Markov chains (orders 0–3) on discrete streams: digital_root, digit_sum,
    binned diff_signed, raw n mod k
  - AR(1) linear autoregression on raw n
  - Longest / strongest repeated contiguous motifs (tuples)
  - Null: shuffled digital_root — how often does random order match Markov "gain"?

Writes JSON + Markdown report under data/analysis/single_cell/.

Stdlib only.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class MarkovResult:
    name: str
    order: int
    alphabet_size: int
    train_n: int
    test_n: int
    accuracy: float
    baseline_majority_accuracy: float
    description: str


def read_series_csv(path: Path) -> dict[str, list]:
    cols: dict[str, list] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for k, v in row.items():
                if v == "":
                    cols[k].append(None)
                else:
                    if k in ("idx", "year", "draw_index_year"):
                        cols[k].append(int(v))
                    elif k == "date":
                        cols[k].append(v)
                    else:
                        try:
                            cols[k].append(int(v))
                        except ValueError:
                            cols[k].append(None)
    return dict(cols)


def markov_eval(
    seq: list[int],
    order: int,
    train_ratio: float,
    name: str,
) -> MarkovResult | None:
    """Chronological split; predict next symbol with MLE Markov; fallback to global majority."""
    s = [x for x in seq if x is not None]
    if len(s) < order + 50:
        return None
    split = max(order + 1, int(len(s) * train_ratio))
    train = s[:split]
    if len(s) - split < 20:
        return None

    glob = Counter(train)
    maj_glob = glob.most_common(1)[0][0]

    if order == 0:
        total = 0
        correct = 0
        for i in range(split, len(s)):
            total += 1
            if s[i] == maj_glob:
                correct += 1
        acc = correct / total
        return MarkovResult(
            name=name,
            order=0,
            alphabet_size=len(set(s)),
            train_n=len(train),
            test_n=total,
            accuracy=acc,
            baseline_majority_accuracy=acc,
            description=f"Markov order 0 (marginal mode) on {name}",
        )

    trans: dict[tuple, Counter] = defaultdict(Counter)
    for i in range(order, len(train)):
        ctx = tuple(train[i - order : i])
        trans[ctx][train[i]] += 1

    correct = 0
    base = 0
    total = 0
    for i in range(split, len(s)):
        ctx = tuple(s[i - order : i])
        true = s[i]
        if ctx in trans and trans[ctx]:
            pred = trans[ctx].most_common(1)[0][0]
        else:
            pred = maj_glob
        if pred == true:
            correct += 1
        if maj_glob == true:
            base += 1
        total += 1

    alphabet_size = len(set(s))
    return MarkovResult(
        name=name,
        order=order,
        alphabet_size=alphabet_size,
        train_n=len(train),
        test_n=total,
        accuracy=correct / total,
        baseline_majority_accuracy=base / total,
        description=f"Markov order {order} on {name}",
    )


def bin_quantiles(train: list[int], full: list[int], n_bins: int) -> list[int]:
    """Map each value to 0..n_bins-1 using train quantile edges."""
    t = sorted(train)
    if len(t) < n_bins * 5:
        return []
    edges: list[int] = []
    for b in range(1, n_bins):
        idx = min(len(t) - 1, int(b / n_bins * len(t)))
        edges.append(t[idx])
    edges = sorted(set(edges))
    if not edges:
        return []

    def code(x: int) -> int:
        return bisect.bisect_right(edges, x)

    return [code(x) for x in full]


def ols_ar1(y: list[int], train_ratio: float) -> dict | None:
    """y[t+1] = a*y[t] + b; fit on train prefix; eval MAE on test."""
    if len(y) < 30:
        return None
    split = int(len(y) * train_ratio)
    if split < 10 or len(y) - split < 10:
        return None
    x = y[: split - 1]
    tgt = y[1:split]
    n = len(x)
    mx = sum(x) / n
    my = sum(tgt) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx <= 1e-12:
        return None
    sxy = sum((x[i] - mx) * (tgt[i] - my) for i in range(n))
    a = sxy / sxx
    b = my - a * mx

    # train R^2
    ss_tot = sum((tgt[i] - my) ** 2 for i in range(n))
    pred_tr = [a * x[i] + b for i in range(n)]
    ss_res = sum((tgt[i] - pred_tr[i]) ** 2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mean_train = my
    mae_ar = 0.0
    mae_mean = 0.0
    test_n = 0
    for i in range(split - 1, len(y) - 1):
        x_i = y[i]
        y_next = y[i + 1]
        p_ar = a * x_i + b
        mae_ar += abs(y_next - p_ar)
        mae_mean += abs(y_next - mean_train)
        test_n += 1
    if test_n == 0:
        return None
    return {
        "a": round(a, 6),
        "b": round(b, 6),
        "train_r2": round(r2, 6),
        "test_mae_ar1": round(mae_ar / test_n, 4),
        "test_mae_predict_mean": round(mae_mean / test_n, 4),
        "test_n": test_n,
    }


def repeated_motifs(
    seq: list[int],
    min_len: int,
    max_len: int,
    min_count: int,
    top: int,
) -> list[dict]:
    """Contiguous tuples that appear at least min_count times."""
    found: list[tuple[tuple, int, int]] = []
    for L in range(max_len, min_len - 1, -1):
        pos: dict[tuple, list[int]] = defaultdict(list)
        for i in range(len(seq) - L + 1):
            tup = tuple(seq[i : i + L])
            pos[tup].append(i)
        for tup, starts in pos.items():
            if len(starts) >= min_count:
                found.append((tup, len(starts), L))
        if len(found) >= top:
            break
    found.sort(key=lambda x: (-x[1], -x[2], x[0]))
    out = []
    for tup, cnt, L in found[:top]:
        out.append(
            {
                "length": L,
                "count": cnt,
                "pattern": list(tup),
            }
        )
    return out


def shuffle_markov_gain(
    seq: list[int], order: int, train_ratio: float, n_shuffles: int, seed: int
) -> dict:
    """Compare real test accuracy gain vs shuffled sequences (same alphabet)."""
    rng = random.Random(seed)
    real_m = markov_eval(seq, order, train_ratio, "digital_root")
    if real_m is None:
        return {}
    real_gain = real_m.accuracy - real_m.baseline_majority_accuracy
    gains: list[float] = []
    s = list(seq)
    for _ in range(n_shuffles):
        rng.shuffle(s)
        m = markov_eval(s, order, train_ratio, "shuffle")
        if m:
            gains.append(m.accuracy - m.baseline_majority_accuracy)
    gains.sort()
    if not gains:
        return {}
    below = sum(1 for g in gains if g < real_gain)
    return {
        "real_gain_over_majority": round(real_gain, 6),
        "shuffle_gains_min_median_max": [
            round(gains[0], 6),
            round(gains[len(gains) // 2], 6),
            round(gains[-1], 6),
        ],
        "real_gain_exceeds_shuffle_fraction": round(below / len(gains), 4),
        "n_shuffles": n_shuffles,
        "note": "If real_gain is not above most shuffles, Markov signal is weak.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--series-csv",
        type=Path,
        default=None,
        help="Path to *_series.csv (default: MILANO pos1 y1900 under single_cell)",
    )
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--shuffle-reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    default_csv = (
        root
        / "data"
        / "analysis"
        / "single_cell"
        / "MILANO_pos1_y1900_series.csv"
    )
    series_path = args.series_csv or default_csv
    if not series_path.is_file():
        print(
            f"Missing {series_path}. Run single_cell_derived_series.py first.",
            file=sys.stderr,
        )
        return 1

    C = read_series_csv(series_path)
    tag = series_path.stem.replace("_series", "")

    digital_root = [x for x in C.get("digital_root", []) if x is not None]
    digit_sum = [x for x in C.get("digit_sum", []) if x is not None]
    n_raw = [x for x in C.get("n", []) if x is not None]
    diffs = [x for x in C.get("diff_signed", []) if x is not None]

    markov_rows: list[MarkovResult] = []
    for order in (0, 1, 2, 3):
        m = markov_eval(digital_root, order, args.train_ratio, "digital_root")
        if m:
            markov_rows.append(m)
    for order in (1, 2):
        m = markov_eval(digit_sum, order, args.train_ratio, "digit_sum")
        if m:
            markov_rows.append(m)

    # binned diff: quantiles from train portion only
    binned: list[int] = []
    if len(diffs) > 100:
        sp = int(len(diffs) * args.train_ratio)
        train_d, full_d = diffs[:sp], diffs
        bq = bin_quantiles(train_d, full_d, 12)
        if bq:
            binned = bq
            for order in (1, 2):
                m = markov_eval(binned, order, args.train_ratio, "diff_signed_q12")
                if m:
                    markov_rows.append(m)

    # n mod 10 (residue)
    mod10 = [x % 10 for x in n_raw]
    for order in (1, 2, 3):
        m = markov_eval(mod10, order, args.train_ratio, "n_mod10")
        if m:
            markov_rows.append(m)

    ar1 = ols_ar1(n_raw, args.train_ratio)

    # Signed diffs: empirically almost no 3-grams repeat >=3 times (sequence is "rich");
    # use short windows. Digital root: only short motifs reach count>=3.
    motifs_diff = repeated_motifs(diffs, 2, 4, 3, 25)
    motifs_dr = repeated_motifs(digital_root, 3, 5, 3, 25)

    shuffle_report_dr = shuffle_markov_gain(
        digital_root, 1, args.train_ratio, args.shuffle_reps, args.seed
    )
    shuffle_report_diff = (
        shuffle_markov_gain(
            binned, 2, args.train_ratio, args.shuffle_reps, args.seed + 1
        )
        if len(binned) > 50
        else {}
    )

    report = {
        "input_csv": str(series_path.name),
        "train_ratio": args.train_ratio,
        "lengths": {
            "n": len(n_raw),
            "diff_signed": len(diffs),
            "digital_root": len(digital_root),
        },
        "markov": [asdict(x) for x in markov_rows],
        "ar1_on_raw_n": ar1,
        "motifs_diff_signed_len2to4_min3": motifs_diff,
        "motifs_digital_root_len3to5_min3": motifs_dr,
        "shuffle_null_markov1_digital_root": shuffle_report_dr,
        "shuffle_null_markov2_diff_q12": shuffle_report_diff,
        "interpretation": (
            "Strong structure would show: large real_gain over majority with "
            "shuffle_null showing real_gain above most shuffles; AR(1) test MAE "
            "clearly below predict-mean; motifs may still appear by chance on long strings."
        ),
    }

    out_dir = root / "data" / "analysis" / "single_cell"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{tag}_structure_report.json"
    md_path = out_dir / f"{tag}_structure_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Structure report: `{series_path.name}`",
        "",
        "## Sizes",
        "",
        f"- n: {len(n_raw)}, diffs: {len(diffs)}, digital_root: {len(digital_root)}",
        "",
        "## Markov (chronological train/test)",
        "",
        "| stream | order | test acc | baseline (majority) |",
        "|--------|-------|----------|---------------------|",
    ]
    for m in markov_rows:
        lines.append(
            f"| {m.name} | {m.order} | {m.accuracy:.4f} | {m.baseline_majority_accuracy:.4f} |"
        )
    lines.append("")
    lines.append("## AR(1) on raw n")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(ar1, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Shuffle null — Markov order 1, digital_root")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(shuffle_report_dr, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Shuffle null — Markov order 2, diff quantile-12 bins")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(shuffle_report_diff, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Top repeated motifs (diff_signed, length 2–4)")
    lines.append("")
    for m in motifs_diff[:10]:
        lines.append(f"- len {m['length']} ×{m['count']}: `{m['pattern']}`")
    lines.append("")
    lines.append("## Top repeated motifs (digital_root, length 3–5)")
    lines.append("")
    for m in motifs_dr[:10]:
        lines.append(f"- len {m['length']} ×{m['count']}: `{m['pattern']}`")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {json_path.name} and {md_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
