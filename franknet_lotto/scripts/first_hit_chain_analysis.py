#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ordine di prima comparsa (1..90) dentro ogni ciclo di copertura, nell'ordine in cui
i numeri compaiono nel CSV (stesso ordine di colonne di draws_wide).

Trasformazione a catena (come da richiesta esemplificata: primo 3 -> 4):
  chained[0] = order[0] + 1
  chained[i] = chained[i-1] + order[i]  per i >= 1

Analisi:
  - doppioni nella stessa estrazione (stesso numero piu volte tra le ruote)
  - entropia empirica del valore alla posizione k nel ciclo (su cicli completi)
  - transizioni tra decili di ord[i] -> ord[i+1]
  - correlazione posizione nel ciclo vs valore (media per step)

Output: data/analysis/first_hit_chain/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

META_COLS = {"year", "date", "draw_index_year", "num_wheels"}


def row_numbers_ordered(row: dict, keys: list[str]) -> list[int]:
    out: list[int] = []
    for k in keys:
        v = row.get(k, "")
        if v == "" or v is None:
            continue
        out.append(int(v))
    return out


def chain_from_order(order: list[int]) -> list[int]:
    if not order:
        return []
    c = [order[0] + 1]
    for i in range(1, len(order)):
        c.append(c[-1] + order[i])
    return c


def shannon_entropy(counts: Counter, n_cat: int) -> float:
    tot = sum(counts.values())
    if tot <= 0:
        return 0.0
    h = 0.0
    for i in range(1, n_cat + 1):
        x = counts.get(i, 0)
        if x <= 0:
            continue
        p = x / tot
        h -= p * math.log2(p)
    return h


def decile(x: int) -> int:
    return min(9, (x - 1) * 10 // 90)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=1871)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    wide = root / "data" / "draws_wide.csv"
    if not wide.is_file():
        print("Manca data/draws_wide.csv", file=sys.stderr)
        return 1

    out_dir = root / "data" / "analysis" / "first_hit_chain"
    out_dir.mkdir(parents=True, exist_ok=True)

    with wide.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        keys = [k for k in r.fieldnames or [] if k not in META_COLS]

        seen: set[int] = set()
        first_hit_order: list[int] = []
        cycle_id = 0
        completed_cycles: list[dict] = []
        dup_rows: list[dict] = []

        for row in r:
            if int(row["year"]) < args.from_year:
                continue
            date = row["date"]
            didx = row["draw_index_year"]
            nums = row_numbers_ordered(row, keys)
            if not nums:
                continue

            cnt = Counter(nums)
            excess = sum(c - 1 for c in cnt.values() if c > 1)
            dup_rows.append(
                {
                    "cycle_id": cycle_id,
                    "date": date,
                    "draw_index_year": didx,
                    "n_balls": len(nums),
                    "n_unique": len(cnt),
                    "duplicate_excess": excess,
                }
            )

            for x in nums:
                if not (1 <= x <= 90):
                    continue
                if x in seen:
                    continue
                seen.add(x)
                first_hit_order.append(x)

            if len(seen) == 90:
                chain = chain_from_order(first_hit_order)
                completed_cycles.append(
                    {
                        "cycle_id": cycle_id,
                        "order": list(first_hit_order),
                        "chained": chain,
                        "chained_mod90": [((v - 1) % 90) + 1 for v in chain],
                    }
                )
                cycle_id += 1
                seen = set()
                first_hit_order = []

    # Long CSV: ogni passo di ogni ciclo completo
    step_path = out_dir / "first_hit_steps.csv"
    with step_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "cycle_id",
                "step",
                "first_hit_value",
                "chained",
                "chained_mod90",
            ]
        )
        for c in completed_cycles:
            cid = c["cycle_id"]
            for i, (v, ch, m) in enumerate(
                zip(c["order"], c["chained"], c["chained_mod90"])
            ):
                w.writerow([cid, i + 1, v, ch, m])

    # Entropia per posizione k (1..90) sul valore first_hit
    ent_rows = []
    n_c = len(completed_cycles)
    for k in range(90):
        ctr: Counter = Counter()
        for c in completed_cycles:
            ctr[c["order"][k]] += 1
        h = shannon_entropy(ctr, 90)
        h_max = math.log2(90)
        ent_rows.append(
            {
                "step": k + 1,
                "n_cycles": n_c,
                "entropy_bits": round(h, 6),
                "max_entropy_bits": round(h_max, 6),
                "ratio": round(h / h_max, 6) if h_max > 0 else 0,
            }
        )

    with (out_dir / "entropy_by_step.csv").open("w", newline="", encoding="utf-8") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(ent_rows[0].keys()))
        wr.writeheader()
        wr.writerows(ent_rows)

    # Step con entropia minima (piu "concentrata" empiricamente — non implica prevedibilita futura)
    ent_sorted = sorted(ent_rows, key=lambda x: x["entropy_bits"])
    lowest = ent_sorted[:5]
    highest = ent_sorted[-5:]

    # Transizioni decile -> decile tra passi consecutivi (pool tutti i cicli)
    trans: dict[tuple[int, int], int] = defaultdict(int)
    for c in completed_cycles:
        o = c["order"]
        for i in range(len(o) - 1):
            trans[(decile(o[i]), decile(o[i + 1]))] += 1

    with (out_dir / "transition_decile_next.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["from_decile_0to9", "to_decile_0to9", "count"])
        for (a, b), cnt in sorted(trans.items()):
            w.writerow([a, b, cnt])

    # Media e dev std del valore a ogni step
    mean_by_step = []
    for k in range(90):
        vals = [c["order"][k] for c in completed_cycles]
        m = sum(vals) / len(vals)
        var = sum((x - m) ** 2 for x in vals) / len(vals)
        mean_by_step.append(
            {
                "step": k + 1,
                "mean_value": round(m, 4),
                "std_value": round(math.sqrt(var), 4),
            }
        )

    with (out_dir / "mean_std_by_step.csv").open("w", newline="", encoding="utf-8") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(mean_by_step[0].keys()))
        wr.writeheader()
        wr.writerows(mean_by_step)

    # Doppioni: riepilogo
    dup_total = sum(d["duplicate_excess"] for d in dup_rows)
    draws_with_dup = sum(1 for d in dup_rows if d["duplicate_excess"] > 0)
    summary = {
        "n_completed_cycles": n_c,
        "chain_rule": "chained[0]=order[0]+1; chained[i]=chained[i-1]+order[i]",
        "first_hit_order": "per ciclo, ordine di prima comparsa 1..90; ordine lettura = ordine colonne CSV",
        "duplicates_within_same_draw": {
            "total_excess_ball_duplicates": dup_total,
            "draw_rows_total": len(dup_rows),
            "draw_rows_with_any_duplicate": draws_with_dup,
            "fraction_draws_with_duplicate": round(
                draws_with_dup / len(dup_rows), 6
            )
            if dup_rows
            else 0,
            "note": "Con ~55 numeri per estrazione su 90 etichette, lo stesso valore compare quasi sempre su piu ruote: non e anomalia.",
        },
        "entropy_by_step": {
            "lowest_entropy_steps_most_concentrated_marginal": lowest,
            "highest_entropy_steps": highest,
            "note": "Bassa entropia = distribuzione empirica piu stretta sui cicli osservati; non e prova di modello causale.",
        },
        "files": {
            "steps": "first_hit_steps.csv",
            "entropy_by_step": "entropy_by_step.csv",
            "transition_decile": "transition_decile_next.csv",
            "mean_std_by_step": "mean_std_by_step.csv",
            "duplicates_per_draw": "duplicates_per_draw.csv",
        },
    }

    with (out_dir / "duplicates_per_draw.csv").open("w", newline="", encoding="utf-8") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(dup_rows[0].keys()))
        wr.writeheader()
        wr.writerows(dup_rows)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nScritto in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
