#!/usr/bin/env python3
"""
Report giornaliero completo: cross_opt_v1 su TORINO, ROMA, BARI.
Tutte le sorti ADM (1 euro ciascuna per estrazione) + riepilogo 2025/2026.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_ambi_2025 import load_draws_by_date, score_cross_numbers, top_n_from_pool
from simulate_sorti_proper import BETS, BetType, PAYOUT, TAX, check_win
from vertibile_rule_test import pool_cross_opt_v1

WHEELS = ["TORINO", "ROMA", "BARI"]
SORTI_ORDER = ["estratto", "ambo", "terno", "quaterna", "cinquina"]


@dataclass
class DayResult:
    date: str
    pool_size: int
    top5: list[int]
    spent: float
    won_gross: float
    hits: dict[str, bool]
    drawn: list[int]


def simulate_wheel_year(wheel: str, year: int, dates_all: list, by_date: dict) -> dict:
    prev: dict[str, list[int]] = {}
    for d in dates_all:
        if d >= f"{year}-01-01":
            break
        if wheel in by_date.get(d, {}):
            prev[wheel] = by_date[d][wheel]

    daily: list[DayResult] = []
    by_bet = {b.name: {"spent": 0.0, "won": 0.0, "hits": 0, "draws": 0} for b in BETS}

    for d in sorted(x for x in dates_all if x.startswith(f"{year}-")):
        if wheel not in by_date.get(d, {}):
            continue
        if wheel not in prev:
            prev[wheel] = by_date[d][wheel]
            continue

        pool = pool_cross_opt_v1(prev[wheel])
        scores = score_cross_numbers(prev[wheel])
        top5 = top_n_from_pool(pool, scores, 5)
        if len(top5) < 5:
            prev[wheel] = by_date[d][wheel]
            continue

        drawn = by_date[d][wheel]
        drawn_set = set(drawn)
        day_spent = 0.0
        day_won = 0.0
        hits_today: dict[str, bool] = {}

        for bet in BETS:
            picks = top5[: bet.k]
            stake = 1.0
            day_spent += stake
            by_bet[bet.name]["spent"] += stake
            by_bet[bet.name]["draws"] += 1
            won = check_win(picks, drawn_set, bet)
            hits_today[bet.name] = won
            if won:
                win_amt = stake * bet.mult
                day_won += win_amt
                by_bet[bet.name]["won"] += win_amt
                by_bet[bet.name]["hits"] += 1

        daily.append(
            DayResult(
                date=d,
                pool_size=len(pool),
                top5=top5,
                spent=day_spent,
                won_gross=day_won,
                hits=hits_today,
                drawn=drawn,
            )
        )
        prev[wheel] = drawn

    n = len(daily)
    total_spent = sum(x.spent for x in daily)
    total_won = sum(x.won_gross for x in daily)
    won_net = total_won * (1 - TAX) if total_won > 0 else 0.0

    daily_rows = []
    for x in daily:
        daily_rows.append(
            {
                "date": x.date,
                "pool_size": x.pool_size,
                "top5_predicted": x.top5,
                "drawn": x.drawn,
                "pool_hits": sum(1 for t in x.top5 if t in set(x.drawn)),
                "spent_eur": x.spent,
                "won_gross_eur": round(x.won_gross, 2),
                "net_gross_eur": round(x.won_gross - x.spent, 2),
                "hits": x.hits,
            }
        )

    bet_summary = []
    for name in SORTI_ORDER:
        b = by_bet[name]
        bet_summary.append(
            {
                "bet": name,
                "multiplier": PAYOUT[name],
                "numbers_played": {"estratto": 1, "ambo": 2, "terno": 3, "quaterna": 4, "cinquina": 5}[name],
                "draws": b["draws"],
                "hits": b["hits"],
                "hit_rate_pct": round(100 * b["hits"] / b["draws"], 2) if b["draws"] else 0,
                "spent_eur": round(b["spent"], 2),
                "won_gross_eur": round(b["won"], 2),
                "net_gross_eur": round(b["won"] - b["spent"], 2),
            }
        )

    return {
        "wheel": wheel,
        "year": year,
        "method": "cross_opt_v1",
        "play_plan_per_draw_day": {
            "description": "Ogni giorno con estrazione: 1 EUR per sorte (5 sorti) = 5 EUR totali",
            "numbers_predicted_top5": "top-5 per score cross nel pool (~59 numeri)",
            "per_sorte": {
                name: {
                    "numeri": k,
                    "numeri_sono": f"top-{k} della previsione",
                    "costo_eur": 1.0,
                    "vincita_se_ok_lordo": PAYOUT[name],
                }
                for name, k in zip(SORTI_ORDER, [1, 2, 3, 4, 5])
            },
        },
        "summary": {
            "draw_days": n,
            "avg_pool_size": round(statistics.mean([x.pool_size for x in daily]), 1) if daily else 0,
            "avg_pool_hits_in_top5": round(
                statistics.mean([sum(1 for t in x.top5 if t in set(x.drawn)) for x in daily]), 2
            )
            if daily
            else 0,
            "total_spent_eur": round(total_spent, 2),
            "total_won_gross_eur": round(total_won, 2),
            "total_won_net_eur": round(won_net, 2),
            "net_gross_eur": round(total_won - total_spent, 2),
            "net_net_eur": round(won_net - total_spent, 2),
            "avg_daily_spent_eur": round(total_spent / n, 2) if n else 0,
            "avg_daily_won_gross_eur": round(total_won / n, 2) if n else 0,
            "avg_daily_net_gross_eur": round((total_won - total_spent) / n, 2) if n else 0,
            "roi_gross_pct": round(100 * (total_won - total_spent) / total_spent, 1) if total_spent else 0,
        },
        "by_bet": bet_summary,
        "daily": daily_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2025,2026")
    ap.add_argument("--out", default="data/analysis/wheel_daily_torino_roma_bari.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    dates, by_date = load_draws_by_date(root / "data" / "draws_wide.csv")
    years = [int(y) for y in args.years.split(",")]

    reports = []
    for wheel in WHEELS:
        for year in years:
            reports.append(simulate_wheel_year(wheel, year, dates, by_date))

    # stampa
    print("METODO: cross_opt_v1")
    print("OGNI GIORNO DI ESTRAZIONE:")
    print("  - Pool ~59 numeri dalla quintina del giorno prima")
    print("  - Previsione: TOP-5 numeri (score ricette cross)")
    print("  - Giochi 5 sorti x 1 EUR = 5 EUR/giorno")
    print("    estratto(top1) | ambo(top2) | terno(top3) | quaterna(top4) | cinquina(top5)")
    print()

    for rep in reports:
        s = rep["summary"]
        print("=" * 60)
        print(f"{rep['wheel']} | {rep['year']} | {s['draw_days']} giorni estrazione")
        print(f"  Pool medio: {s['avg_pool_size']} num | top5 indovina {s['avg_pool_hits_in_top5']}/5 in media")
        print(f"  SPESA totale: {s['total_spent_eur']:.2f} EUR  ({s['avg_daily_spent_eur']:.2f} EUR/giorno)")
        print(f"  VINTO lordo:  {s['total_won_gross_eur']:.2f} EUR  ({s['avg_daily_won_gross_eur']:.2f} EUR/giorno)")
        print(f"  NETTO lordo:  {s['net_gross_eur']:+.2f} EUR  ({s['avg_daily_net_gross_eur']:+.2f} EUR/giorno)")
        print(f"  NETTO netto:  {s['net_net_eur']:+.2f} EUR (dopo tasse 8%)")
        print("  Per sorte:")
        for b in rep["by_bet"]:
            flag = " <<<" if b["net_gross_eur"] > 0 else ""
            print(
                f"    {b['bet']:<10} {b['numbers_played']} num | "
                f"hit {b['hits']}/{b['draws']} ({b['hit_rate_pct']:.1f}%) | "
                f"speso {b['spent_eur']:.0f} | vinto {b['won_gross_eur']:.2f} | "
                f"netto {b['net_gross_eur']:+.2f}{flag}"
            )
        print()

    out = root / args.out
    out.write_text(json.dumps({"wheels": WHEELS, "years": years, "reports": reports}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report JSON (con dettaglio giorno per giorno): {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
