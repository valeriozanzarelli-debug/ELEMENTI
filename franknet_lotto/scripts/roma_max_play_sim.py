#!/usr/bin/env python3
"""
ROMA: simulazione AMBI 2025+2026 con copertura massima da cross_opt_v1.
Confronta strategie (top-N, pool intero, posta alta).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_ambi_2025 import (
    count_winning_ambi,
    load_draws_by_date,
    net_from_gross,
    score_cross_numbers,
    top_n_from_pool,
)
from vertibile_rule_test import pool_cross_opt_v1

WHEEL = "ROMA"
AMBO_MULT = 250.0
TAX = 0.08


@dataclass
class Strategy:
    name: str
    mode: Literal["top_n", "full_pool"]
    top_n: int | None
    stake_eur: float


def picks_for_strategy(pool: set[int], scores: dict[int, int], strat: Strategy) -> list[int]:
    if strat.mode == "full_pool":
        ranked = sorted(pool, key=lambda n: (-scores.get(n, 0), n))
        return ranked
    assert strat.top_n is not None
    return top_n_from_pool(pool, scores, strat.top_n)


def simulate_roma_year(
    dates_all: list[str],
    by_date: dict,
    year: int,
    strat: Strategy,
) -> dict:
    prev: dict[str, list[int]] = {}
    for d in dates_all:
        if d >= f"{year}-01-01":
            break
        if WHEEL in by_date.get(d, {}):
            prev[WHEEL] = by_date[d][WHEEL]

    daily = []
    total_spent = 0.0
    total_won = 0.0
    total_ambi = 0
    pool_sizes = []

    for d in sorted(x for x in dates_all if x.startswith(f"{year}-")):
        if WHEEL not in by_date.get(d, {}):
            continue
        if WHEEL not in prev:
            prev[WHEEL] = by_date[d][WHEEL]
            continue

        pool = pool_cross_opt_v1(prev[WHEEL])
        scores = score_cross_numbers(prev[WHEEL])
        picks = picks_for_strategy(pool, scores, strat)
        if len(picks) < 2:
            prev[WHEEL] = by_date[d][WHEEL]
            continue

        pairs = math.comb(len(picks), 2)
        stake_ambo = strat.stake_eur / pairs
        drawn = set(by_date[d][WHEEL])
        hits = count_winning_ambi(picks, drawn)
        won = hits * stake_ambo * AMBO_MULT

        total_spent += strat.stake_eur
        total_won += won
        total_ambi += hits
        pool_sizes.append(len(picks))

        daily.append(
            {
                "date": d,
                "spent": strat.stake_eur,
                "won_gross": round(won, 2),
                "net": round(won - strat.stake_eur, 2),
                "ambi": hits,
                "picks_n": len(picks),
                "pairs": pairs,
            }
        )
        prev[WHEEL] = by_date[d][WHEEL]

    n = len(daily)
    won_net = net_from_gross(total_won, TAX)
    return {
        "strategy": strat.name,
        "year": year,
        "draws": n,
        "avg_picks": round(sum(pool_sizes) / n, 1) if n else 0,
        "avg_pairs": round(sum(d["pairs"] for d in daily) / n, 0) if n else 0,
        "total_spent_eur": round(total_spent, 2),
        "total_won_gross_eur": round(total_won, 2),
        "total_won_net_eur": round(won_net, 2),
        "net_gross_eur": round(total_won - total_spent, 2),
        "net_net_eur": round(won_net - total_spent, 2),
        "ambi_hits": total_ambi,
        "ambi_per_draw": round(total_ambi / n, 4) if n else 0,
        "roi_gross_pct": round(100 * (total_won - total_spent) / total_spent, 2) if total_spent else 0,
        "daily": daily,
    }


def build_strategies() -> list[Strategy]:
    s = []
    for n in (8, 12, 15, 20, 25, 30):
        s.append(Strategy(f"top{n}_1eur", "top_n", n, 1.0))
    s.append(Strategy("full_pool_1eur", "full_pool", None, 1.0))
    s.append(Strategy("full_pool_5eur", "full_pool", None, 5.0))
    s.append(Strategy("full_pool_10eur", "full_pool", None, 10.0))
    s.append(Strategy("top20_5eur", "top_n", 20, 5.0))
    s.append(Strategy("top25_5eur", "top_n", 25, 5.0))
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2025,2026")
    ap.add_argument("--out", default="data/analysis/roma_max_play_2025_2026.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    dates, by_date = load_draws_by_date(root / "data" / "draws_wide.csv")
    years = [int(y.strip()) for y in args.years.split(",")]
    strategies = build_strategies()

    results = []
    for year in years:
        for strat in strategies:
            r = simulate_roma_year(dates, by_date, year, strat)
            # daily troppo lungo per JSON
            r_summary = {k: v for k, v in r.items() if k != "daily"}
            results.append(r_summary)

    # migliore per anno (netto lordo)
    best_by_year = {}
    for year in years:
        yr = [r for r in results if r["year"] == year]
        best = max(yr, key=lambda x: x["net_gross_eur"])
        profitable = [r for r in yr if r["net_gross_eur"] > 0]
        best_by_year[str(year)] = {
            "best_strategy": best["strategy"],
            "best_net_gross": best["net_gross_eur"],
            "profitable_count": len(profitable),
            "profitable_strategies": [
                {"name": r["strategy"], "net": r["net_gross_eur"], "spent": r["total_spent_eur"]}
                for r in sorted(profitable, key=lambda x: -x["net_gross_eur"])
            ],
        }

    # totale 2025+2026 per strategie chiave
    combined = defaultdict(lambda: {"spent": 0.0, "won": 0.0, "ambi": 0, "draws": 0})
    for r in results:
        c = combined[r["strategy"]]
        c["spent"] += r["total_spent_eur"]
        c["won"] += r["total_won_gross_eur"]
        c["ambi"] += r["ambi_hits"]
        c["draws"] += r["draws"]

    combined_rows = []
    for name, c in combined.items():
        combined_rows.append(
            {
                "strategy": name,
                "draws": c["draws"],
                "spent": round(c["spent"], 2),
                "won_gross": round(c["won"], 2),
                "net_gross": round(c["won"] - c["spent"], 2),
                "ambi": c["ambi"],
            }
        )
    combined_rows.sort(key=lambda x: -x["net_gross"])

    report = {
        "wheel": WHEEL,
        "method": "cross_opt_v1",
        "years": years,
        "ambo_multiplier": AMBO_MULT,
        "tax_rate": TAX,
        "results_by_year_strategy": results,
        "best_by_year": best_by_year,
        "combined_2025_2026": combined_rows,
    }

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"ROMA | cross_opt_v1 | AMBI | anni {years}")
    print()
    for year in years:
        print(f"--- {year} ---")
        yr = sorted([r for r in results if r["year"] == year], key=lambda x: -x["net_gross_eur"])
        print(f"{'Strategia':<22} {'Draw':>5} {'Speso':>9} {'Vinto':>9} {'Netto':>9} {'Ambi':>5}")
        for r in yr:
            flag = " *" if r["net_gross_eur"] > 0 else ""
            print(
                f"{r['strategy']:<22} {r['draws']:>5} {r['total_spent_eur']:>9.2f} "
                f"{r['total_won_gross_eur']:>9.2f} {r['net_gross_eur']:>+9.2f} {r['ambi_hits']:>5}{flag}"
            )
        b = best_by_year[str(year)]
        print(f"Migliore: {b['best_strategy']} ({b['best_net_gross']:+.2f} EUR)")
        if b["profitable_strategies"]:
            print(f"In profitto: {len(b['profitable_strategies'])} strategie")
        else:
            print("Nessuna strategia in profitto")
        print()

    print("--- COMBINATO 2025+2026 (top 5) ---")
    for row in combined_rows[:5]:
        print(
            f"{row['strategy']:<22} speso {row['spent']:.2f} | vinto {row['won_gross']:.2f} | "
            f"netto {row['net_gross']:+.2f} | ambi {row['ambi']}"
        )
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
