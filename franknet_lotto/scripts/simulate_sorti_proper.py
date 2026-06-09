#!/usr/bin/env python3
"""
Simulazione CORRETTA sorti Lotto (ADM) su singola ruota.
1 euro INTERO sulla combinazione secca (top-K dal metodo cross_opt_v1).

Moltiplicatori ufficiali singola ruota, 1€ sulla sorte:
  estratto 11.23 | ambo 250 | terno 4500 | quaterna 120000 | cinquina 6_000_000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_ambi_2025 import load_draws_by_date, score_cross_numbers, top_n_from_pool
from vertibile_rule_test import pool_cross_opt_v1

WHEELS = [
    "BARI", "CAGLIARI", "FIRENZE", "GENOVA", "MILANO", "NAPOLI",
    "PALERMO", "ROMA", "TORINO", "VENEZIA", "NAZIONALE",
]

# ADM coefficienti (1€ intero sulla sorte, singola ruota)
PAYOUT = {
    "estratto": 11.23,
    "ambo": 250.0,
    "terno": 4500.0,
    "quaterna": 120_000.0,
    "cinquina": 6_000_000.0,
}
K_FOR = {"estratto": 1, "ambo": 2, "terno": 3, "quaterna": 4, "cinquina": 5}
TAX = 0.08


@dataclass(frozen=True)
class BetType:
    name: str
    k: int
    mult: float


BETS = [BetType(n, K_FOR[n], PAYOUT[n]) for n in K_FOR]


def check_win(picks: list[int], drawn: set[int], bet: BetType) -> bool:
    if bet.name == "estratto":
        return picks[0] in drawn
    if bet.name == "ambo":
        return picks[0] in drawn and picks[1] in drawn
    if bet.name == "terno":
        return all(p in drawn for p in picks)
    if bet.name == "quaterna":
        return all(p in drawn for p in picks)
    if bet.name == "cinquina":
        return set(picks) == drawn
    return False


def simulate_wheel_year(
    dates_all: list[str],
    by_date: dict,
    wheel: str,
    year: int,
    bet: BetType,
    stake: float = 1.0,
) -> dict:
    prev: dict[str, list[int]] = {}
    for d in dates_all:
        if d >= f"{year}-01-01":
            break
        if wheel in by_date.get(d, {}):
            prev[wheel] = by_date[d][wheel]

    spent = 0.0
    won = 0.0
    hits = 0
    draws = 0

    for d in sorted(x for x in dates_all if x.startswith(f"{year}-")):
        if wheel not in by_date.get(d, {}):
            continue
        if wheel not in prev:
            prev[wheel] = by_date[d][wheel]
            continue

        pool = pool_cross_opt_v1(prev[wheel])
        scores = score_cross_numbers(prev[wheel])
        picks = top_n_from_pool(pool, scores, bet.k)
        if len(picks) < bet.k:
            prev[wheel] = by_date[d][wheel]
            continue

        drawn = set(by_date[d][wheel])
        spent += stake
        draws += 1
        if check_win(picks, drawn, bet):
            hits += 1
            won += stake * bet.mult
        prev[wheel] = by_date[d][wheel]

    won_net = won * (1 - TAX) if won > 0 else 0.0
    return {
        "wheel": wheel,
        "year": year,
        "bet": bet.name,
        "stake_eur": stake,
        "multiplier": bet.mult,
        "draws": draws,
        "hits": hits,
        "hit_rate": round(hits / draws, 4) if draws else 0,
        "spent_eur": round(spent, 2),
        "won_gross_eur": round(won, 2),
        "won_net_eur": round(won_net, 2),
        "net_gross_eur": round(won - spent, 2),
        "net_net_eur": round(won_net - spent, 2),
        "roi_gross_pct": round(100 * (won - spent) / spent, 1) if spent else 0,
    }


def random_hit_rate(bet: BetType) -> float:
    """Probabilita teorica caso (numeri specifici scelti a priori)."""
    if bet.name == "estratto":
        return 5 / 90
    if bet.name == "ambo":
        return math.comb(5, 2) / math.comb(90, 2)
    if bet.name == "terno":
        return math.comb(5, 3) / math.comb(90, 3)
    if bet.name == "quaterna":
        return math.comb(5, 4) / math.comb(90, 4)
    if bet.name == "cinquina":
        return 1 / math.comb(90, 5)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2023,2024,2025,2026")
    ap.add_argument("--out", default="data/analysis/sorti_proper_sim.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    dates, by_date = load_draws_by_date(root / "data" / "draws_wide.csv")
    years = [int(y) for y in args.years.split(",")]

    all_rows = []
    for year in years:
        for wheel in WHEELS:
            for bet in BETS:
                all_rows.append(simulate_wheel_year(dates, by_date, wheel, year, bet))

    profitable = [r for r in all_rows if r["net_gross_eur"] > 0 and r["draws"] >= 20]
    profitable.sort(key=lambda x: -x["net_gross_eur"])

    # aggregato per wheel+bet+years combined
    combo = defaultdict(lambda: {"spent": 0, "won": 0, "hits": 0, "draws": 0})
    for r in all_rows:
        key = (r["wheel"], r["bet"])
        c = combo[key]
        c["spent"] += r["spent_eur"]
        c["won"] += r["won_gross_eur"]
        c["hits"] += r["hits"]
        c["draws"] += r["draws"]

    combo_rows = []
    for (wheel, bet), c in combo.items():
        combo_rows.append({
            "wheel": wheel,
            "bet": bet,
            "years": years,
            "draws": c["draws"],
            "hits": c["hits"],
            "hit_rate": round(c["hits"] / c["draws"], 4) if c["draws"] else 0,
            "spent": round(c["spent"], 2),
            "won_gross": round(c["won"], 2),
            "net_gross": round(c["won"] - c["spent"], 2),
        })
    combo_rows.sort(key=lambda x: -x["net_gross"])

    report = {
        "method": "cross_opt_v1",
        "rule": "1 EUR intero sulla sorte secca (top-K per score cross nel pool)",
        "payouts_adm_single_wheel": PAYOUT,
        "tax_rate": TAX,
        "random_hit_rates": {b.name: round(random_hit_rate(b), 6) for b in BETS},
        "random_ev_per_1eur": {
            b.name: round(random_hit_rate(b) * PAYOUT[b.name] - 1, 4) for b in BETS
        },
        "all_results": all_rows,
        "profitable_cases": profitable,
        "combined_by_wheel_bet": combo_rows,
    }

    out = root / args.out
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("SIMULAZIONE CORRETTA | 1 euro INTERO per sorte | cross_opt_v1")
    print("Quote ADM singola ruota:", PAYOUT)
    print()
    print("EV teorico caso (1 euro/sorta):")
    for b in BETS:
        print(f"  {b.name:10} hit~{random_hit_rate(b)*100:.3f}%  EV={random_hit_rate(b)*b.mult - 1:+.2f} EUR")
    print()
    print(f"Combinazioni IN PROFITTO (netto lordo > 0, >=20 estrazioni): {len(profitable)}")
    print(f"{'Ruota':<12} {'Anno':>5} {'Sorte':<10} {'Draw':>5} {'Hit':>4} {'Hit%':>6} {'Speso':>8} {'Vinto':>10} {'NETTO':>10}")
    for r in profitable[:25]:
        print(
            f"{r['wheel']:<12} {r['year']:>5} {r['bet']:<10} {r['draws']:>5} {r['hits']:>4} "
            f"{r['hit_rate']*100:>5.1f}% {r['spent_eur']:>8.0f} {r['won_gross_eur']:>10.2f} "
            f"{r['net_gross_eur']:>+10.2f}"
        )
    if not profitable:
        print("  (nessuna)")
    print()
    print("TOP 15 combinazioni wheel+bet su TUTTI gli anni:")
    print(f"{'Ruota':<12} {'Sorte':<10} {'Draw':>5} {'Hit%':>6} {'Speso':>8} {'Vinto':>10} {'NETTO':>10}")
    for r in combo_rows[:15]:
        print(
            f"{r['wheel']:<12} {r['bet']:<10} {r['draws']:>5} {r['hit_rate']*100:>5.1f}% "
            f"{r['spent']:>8.0f} {r['won_gross']:>10.2f} {r['net_gross']:>+10.2f}"
        )
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
