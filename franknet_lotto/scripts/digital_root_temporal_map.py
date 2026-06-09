#!/usr/bin/env python3
"""
Digital-root temporal maps and walk-forward checks (no future leakage).

Reads draws_wide.csv, builds per-wheel sequences of digital roots for each
of the 5 positions, then exports:

  - Rolling transition slices (9×9) for heatmaps / Sankey-style analysis
  - Marginal distribution per slice (entropy, mode)
  - Walk-forward Markov order-1 and order-2: incremental prediction accuracy
  - Decade × root contingency + chi² (homogeneity probe)
  - Optional: top repeated 5-tuples (dr1..dr5) per draw for one wheel

This does NOT establish reliable prediction of future combinations; it gives
measurable baselines and temporal “maps” for serious follow-up (plots, ML).

Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def digital_root(n: int) -> int:
    while n > 9:
        n = sum(int(c) for c in str(n))
    return n


def shannon_bits(counts: list[int]) -> float:
    t = sum(counts)
    if t <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / t
        h -= p * math.log2(p)
    return h


def chi2_independence(rows: list[list[int]]) -> tuple[float, int]:
    """Pearson chi-square test of independence; rows = categories A, cols = B."""
    r = len(rows)
    if r == 0:
        return 0.0, 0
    c = len(rows[0])
    n_tot = sum(sum(row) for row in rows)
    if n_tot == 0:
        return 0.0, 0
    row_sums = [sum(row) for row in rows]
    col_sums = [sum(rows[i][j] for i in range(r)) for j in range(c)]
    chi2 = 0.0
    for i in range(r):
        for j in range(c):
            e = row_sums[i] * col_sums[j] / n_tot
            if e <= 0:
                continue
            o = rows[i][j]
            chi2 += (o - e) ** 2 / e
    df = (r - 1) * (c - 1)
    return chi2, df


def load_wheel_series(
    wide_csv: Path, wheel: str, from_year: int
) -> tuple[list[str], list[list[int]]]:
    """Dates aligned with list of 5 ints per draw (skip row if any position empty)."""
    dates: list[str] = []
    values: list[list[int]] = []
    cols = [f"{wheel}_{k}" for k in range(1, 6)]
    with wide_csv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["year"]) < from_year:
                continue
            nums: list[int] = []
            ok = True
            for c in cols:
                v = row[c]
                if v == "":
                    ok = False
                    break
                nums.append(int(v))
            if not ok:
                continue
            dates.append(row["date"])
            values.append(nums)
    return dates, values


def column_dr(values: list[list[int]], pos: int) -> list[int]:
    return [row[pos] for row in values]


def rolling_transitions(
    dr: list[int],
    dates: list[str],
    window: int,
    step: int,
) -> tuple[list[dict], list[dict]]:
    """Slices of transition counts and marginals."""
    trans_rows: list[dict] = []
    marg_rows: list[dict] = []
    n = len(dr)
    if n < window + 1:
        return trans_rows, marg_rows
    slice_idx = 0
    for start in range(0, n - window, step):
        end = start + window
        sub = dr[start : end + 1]
        d0, d1 = dates[start], dates[end]
        trans = [[0] * 9 for _ in range(9)]
        marg = [0] * 9
        for x in sub:
            if 1 <= x <= 9:
                marg[x - 1] += 1
        for i in range(len(sub) - 1):
            a, b = sub[i], sub[i + 1]
            if 1 <= a <= 9 and 1 <= b <= 9:
                trans[a - 1][b - 1] += 1
        row_sums = [sum(trans[i]) for i in range(9)]
        for fr in range(9):
            rs = row_sums[fr]
            for to in range(9):
                cnt = trans[fr][to]
                p = cnt / rs if rs > 0 else 0.0
                trans_rows.append(
                    {
                        "slice_idx": slice_idx,
                        "window_start_date": d0,
                        "window_end_date": d1,
                        "from_dr": fr + 1,
                        "to_dr": to + 1,
                        "count": cnt,
                        "p_to_given_from": round(p, 6) if rs > 0 else "",
                    }
                )
        tot_m = sum(marg)
        ent = shannon_bits(marg)
        mode_i = max(range(9), key=lambda i: marg[i])
        marg_rows.append(
            {
                "slice_idx": slice_idx,
                "window_start_date": d0,
                "window_end_date": d1,
                "n_draws": window + 1,
                "entropy_bits": round(ent, 6),
                "max_entropy_bits": round(math.log2(9), 6),
                "mode_dr": mode_i + 1,
                "mode_freq": marg[mode_i],
                "mode_share": round(marg[mode_i] / tot_m, 6) if tot_m else 0,
            }
        )
        slice_idx += 1
    return trans_rows, marg_rows


def walkforward_markov1(dr: list[int], dates: list[str], min_train: int) -> list[dict]:
    trans: Counter[tuple[int, int]] = Counter()
    out: list[dict] = []
    n = len(dr)
    correct = 0
    total = 0
    for t in range(1, n):
        if t >= min_train:
            ctx = dr[t - 1]
            nxt_counts: Counter = Counter()
            for (a, b), c in trans.items():
                if a == ctx:
                    nxt_counts[b] += c
            if nxt_counts:
                pred = nxt_counts.most_common(1)[0][0]
            else:
                hist = Counter(dr[:t])
                pred = hist.most_common(1)[0][0]
            hit = int(pred == dr[t])
            total += 1
            correct += hit
            out.append(
                {
                    "t_index": t,
                    "date": dates[t],
                    "true_dr": dr[t],
                    "pred_dr": pred,
                    "hit": hit,
                    "rolling_accuracy": round(correct / total, 6),
                }
            )
        trans[(dr[t - 1], dr[t])] += 1
    return out


def walkforward_markov2(dr: list[int], dates: list[str], min_train: int) -> list[dict]:
    """Context (dr[t-2], dr[t-1]) -> dr[t]; train only on past completed transitions."""
    ctx_nxt: dict[tuple[int, int], Counter] = defaultdict(Counter)
    out: list[dict] = []
    n = len(dr)
    correct = 0
    total = 0
    for t in range(2, n):
        if t >= min_train:
            ctx = (dr[t - 2], dr[t - 1])
            hist = ctx_nxt[ctx]
            if hist:
                pred = hist.most_common(1)[0][0]
            else:
                pred = Counter(dr[:t]).most_common(1)[0][0]
            hit = int(pred == dr[t])
            total += 1
            correct += hit
            out.append(
                {
                    "t_index": t,
                    "date": dates[t],
                    "true_dr": dr[t],
                    "pred_dr": pred,
                    "hit": hit,
                    "rolling_accuracy": round(correct / total, 6),
                }
            )
        ctx_nxt[(dr[t - 2], dr[t - 1])][dr[t]] += 1
    return out


def walkforward_baseline_mode(dr: list[int], min_train: int) -> float:
    """Fraction correct if we always predict train-mode-so-far (no Markov)."""
    correct = 0
    total = 0
    for t in range(min_train, len(dr)):
        hist = Counter(dr[:t])
        pred = hist.most_common(1)[0][0]
        if pred == dr[t]:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def decade_contingency(dr: list[int], dates: list[str]) -> tuple[list[list[int]], list[str]]:
    """9 columns (roots 1..9), rows = decade labels."""
    buckets: dict[str, list[int]] = defaultdict(lambda: [0] * 9)
    for d, x in zip(dates, dr):
        if not (1 <= x <= 9):
            continue
        year = int(d[:4])
        dec = f"{(year // 10) * 10}s"
        buckets[dec][x - 1] += 1
    decades = sorted(buckets.keys())
    table = [buckets[d] for d in decades]
    return table, decades


def top_tuple_patterns(values: list[list[int]], top: int) -> list[dict]:
    """5-tuples of digital roots per draw."""
    ctr: Counter[tuple[int, ...]] = Counter()
    for row in values:
        tup = tuple(digital_root(x) for x in row)
        ctr[tup] += 1
    out = []
    for tup, c in ctr.most_common(top):
        out.append({"pattern": list(tup), "count": c})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", default="MILANO")
    ap.add_argument("--from-year", type=int, default=1900)
    ap.add_argument("--window", type=int, default=400, help="Draws per rolling window")
    ap.add_argument("--step", type=int, default=100, help="Rolling step")
    ap.add_argument("--min-train", type=int, default=600, help="Min index before walk-forward preds")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    wide = root / "data" / "draws_wide.csv"
    if not wide.is_file():
        print("Missing data/draws_wide.csv — run parse_draws.py first.", file=sys.stderr)
        return 1

    dates, raw_vals = load_wheel_series(wide, args.wheel, args.from_year)
    if len(raw_vals) < args.min_train + 50:
        print("Not enough draws for this wheel/year filter.", file=sys.stderr)
        return 1

    out_dir = root / "data" / "analysis" / "digital_root" / args.wheel.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "wheel": args.wheel,
        "from_year": args.from_year,
        "n_draws": len(dates),
        "positions": {},
    }

    for pos in range(5):
        dr = [digital_root(raw_vals[i][pos]) for i in range(len(raw_vals))]
        tag = f"pos{pos + 1}"
        sub = out_dir / tag
        sub.mkdir(parents=True, exist_ok=True)

        trans_rows, marg_rows = rolling_transitions(
            dr, dates, args.window, args.step
        )
        if trans_rows:
            last_slice = max(r["slice_idx"] for r in trans_rows)
            mat = [[0.0] * 9 for _ in range(9)]
            for r in trans_rows:
                if r["slice_idx"] != last_slice:
                    continue
                i, j = int(r["from_dr"]) - 1, int(r["to_dr"]) - 1
                p = r["p_to_given_from"]
                mat[i][j] = float(p) if p != "" else 0.0
            meta = {
                "slice_idx": last_slice,
                "window_start_date": next(
                    x["window_start_date"]
                    for x in marg_rows
                    if x["slice_idx"] == last_slice
                ),
                "window_end_date": next(
                    x["window_end_date"]
                    for x in marg_rows
                    if x["slice_idx"] == last_slice
                ),
                "matrix_9x9_rows_from1_to9": mat,
            }
            with (sub / "last_window_transition_matrix.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(meta, f, indent=2)

        with (sub / "rolling_transitions_long.csv").open("w", newline="", encoding="utf-8") as f:
            if trans_rows:
                w = csv.DictWriter(f, fieldnames=list(trans_rows[0].keys()))
                w.writeheader()
                w.writerows(trans_rows)
        with (sub / "rolling_marginals.csv").open("w", newline="", encoding="utf-8") as f:
            if marg_rows:
                w = csv.DictWriter(f, fieldnames=list(marg_rows[0].keys()))
                w.writeheader()
                w.writerows(marg_rows)

        wf1 = walkforward_markov1(dr, dates, args.min_train)
        with (sub / "walkforward_markov1.csv").open("w", newline="", encoding="utf-8") as f:
            if wf1:
                w = csv.DictWriter(f, fieldnames=list(wf1[0].keys()))
                w.writeheader()
                w.writerows(wf1)

        wf2 = walkforward_markov2(dr, dates, max(args.min_train, 3))
        with (sub / "walkforward_markov2.csv").open("w", newline="", encoding="utf-8") as f:
            if wf2:
                w = csv.DictWriter(f, fieldnames=list(wf2[0].keys()))
                w.writeheader()
                w.writerows(wf2)

        acc1 = wf1[-1]["rolling_accuracy"] if wf1 else None
        acc2 = wf2[-1]["rolling_accuracy"] if wf2 else None
        base = walkforward_baseline_mode(dr, args.min_train)

        tab, dec_labels = decade_contingency(dr, dates)
        chi2, df = chi2_independence(tab)

        summary["positions"][tag] = {
            "walkforward_final_acc_markov1": acc1,
            "walkforward_final_acc_markov2": acc2,
            "walkforward_baseline_global_mode": round(base, 6),
            "uniform_baseline_acc": round(1 / 9, 6),
            "decade_chi2": round(chi2, 4),
            "decade_chi2_df": df,
            "decade_labels": dec_labels,
        }

    # Joint 5-tuple patterns (same wheel, all positions)
    patterns = top_tuple_patterns(
        [[digital_root(x) for x in row] for row in raw_vals], 40
    )
    with (out_dir / "joint_5tuple_top40.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "count", "dr1", "dr2", "dr3", "dr4", "dr5"])
        for i, p in enumerate(patterns, 1):
            w.writerow([i, p["count"], *p["pattern"]])

    readme = out_dir / "README_MAP.md"
    readme.write_text(
        f"""# Mappa temporale — radici digitali ({args.wheel}, da {args.from_year})

## Cartelle `pos1` … `pos5`

- **`rolling_transitions_long.csv`**: per ogni finestra scorrevole, tutte le coppie (from_dr → to_dr) con conteggi e **P(to | from)**. Usala per heatmap 9×9 animate nel tempo (`slice_idx` / date).
- **`rolling_marginals.csv`**: entropia e moda della distribuzione marginale delle radici nella finestra (quanto è “piatta” 1…9).
- **`walkforward_markov1.csv` / `walkforward_markov2.csv`**: previsione **solo dal passato** (nessun look-ahead): ad ogni estrazione si aggiorna la catena e si predice la radice successiva. Colonna `rolling_accuracy` = accuratezza cumulativa dal primo step di test.
- **Baseline** in `summary.json`: stesso schema ma predici sempre la **moda globale** del passato; **uniforme** = 1/9 ≈ 0.111.

## File comuni alla ruota

- **`joint_5tuple_top40.csv`**: combinazioni di 5 radici (una riga estrazione) più frequenti nello storico — spazio 9⁵, atteso molto sparso sotto indipendenza.

## Uso per “previsione”

Qualsiasi modello va **rivalidato** con walk-forward o cross-validation temporale. Le combinazioni complete hanno cardinalità enorme: la probabilità sotto ipotesi i.i.d. uniforme sui numeri del lotto **non** coincide con uniforme sulle radici, ma **nessuna analisi storica** giustifica certezza sulle uscite future.

Chi² decenni (in `summary.json`): valori alti ⇒ marginale delle radici **non omogenea** tra decenni (regole, archivio, non-stazionarietà); non implica prevedibilità.

Generato da: `scripts/digital_root_temporal_map.py`
""",
        encoding="utf-8",
    )

    summary["joint_5tuple_note"] = (
        "Top counts are descriptive; expected repetition under simple nulls is low."
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote digital root map under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
