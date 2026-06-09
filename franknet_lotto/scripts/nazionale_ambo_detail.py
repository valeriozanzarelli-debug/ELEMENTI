#!/usr/bin/env python3
"""Dettaglio NAZIONALE ambo — unico caso stabile statisticamente."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_ambi_2025 import load_draws_by_date, score_cross_numbers, top_n_from_pool
from simulate_sorti_proper import check_win, BetType, TAX
from vertibile_rule_test import pool_cross_opt_v1

bet = BetType("ambo", 2, 250.0)
dates, by_date = load_draws_by_date(Path(__file__).parents[1] / "data" / "draws_wide.csv")
prev = {}
years: dict[int, dict] = {}
hits_detail = []

for d in dates:
    if d[:4] < "2015":
        if "NAZIONALE" in by_date.get(d, {}):
            prev["NAZIONALE"] = by_date[d]["NAZIONALE"]
        continue
    if "NAZIONALE" not in by_date.get(d, {}):
        continue
    if "NAZIONALE" not in prev:
        prev["NAZIONALE"] = by_date[d]["NAZIONALE"]
        continue
    y = int(d[:4])
    pool = pool_cross_opt_v1(prev["NAZIONALE"])
    picks = top_n_from_pool(pool, score_cross_numbers(prev["NAZIONALE"]), 2)
    drawn = by_date[d]["NAZIONALE"]
    hit = check_win(picks, set(drawn), bet)
    years.setdefault(y, {"s": 0, "h": 0, "w": 0})
    years[y]["s"] += 1
    if hit:
        years[y]["h"] += 1
        years[y]["w"] += 250
        hits_detail.append({"date": d, "picks": picks, "drawn": drawn})
    prev["NAZIONALE"] = drawn

print("NAZIONALE | AMBO | cross_opt_v1 | top-2 | 1 EUR/estrazione")
print()
for y in sorted(years):
    s = years[y]
    print(f"  {y}: {s['h']}/{s['s']} ambo | speso {s['s']} | vinto {s['w']} | net {s['w']-s['s']:+d}")
tot_s = sum(v["s"] for v in years.values())
tot_w = sum(v["w"] for v in years.values())
print(f"\nTOTALE 2015-26: net lordo {tot_w-tot_s:+d} | netto dopo tasse {tot_w*(1-TAX)-tot_s:+.0f}")
print(f"\nHit registrati ({len(hits_detail)}):")
for h in hits_detail:
    print(f"  {h['date']} giocati {h['picks']} estratti {h['drawn']}")
