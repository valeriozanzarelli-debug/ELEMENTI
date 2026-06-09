#!/usr/bin/env python3
"""Report ROMA: p95 cross_opt_v1 + simulazione AMBI 2025 solo ROMA."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_ambi_2025 import (
    Assumptions,
    count_winning_ambi,
    load_draws_by_date,
    score_cross_numbers,
    top_n_from_pool,
)
from vertibile_rule_test import (
    eval_rule,
    load_wheel,
    mc_mean_analytic,
    monte_carlo_baseline,
    pool_cross_opt_v1,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    wide = root / "data" / "draws_wide.csv"
    rows = load_wheel(wide, "ROMA")

    pairs200 = [(rows[-(201 + i)]["nums"], rows[-(200 + i)]["nums"]) for i in range(200)]
    er = eval_rule(pairs200, pool_cross_opt_v1, "v1")
    mc = mc_mean_analytic(er.pool_sizes)
    _, p95, _ = monte_carlo_baseline(pairs200, er.pool_sizes, reps=3000, seed=42)

    print("=== ROMA | cross_opt_v1 | ultime 200 transizioni ===")
    print(f"Periodo: {rows[-201]['date']} -> {rows[-1]['date']}")
    print(f"Hits: {er.hits}/1000")
    print(f"MC media: {mc:.1f}")
    print(f"p95 Monte Carlo: {p95}")
    print(f"BATTE p95: {'SI' if er.hits > p95 else 'NO'} (margine {er.hits - p95:+.0f})")
    print(f"Media hit/estrazione: {sum(er.per_draw)/200:.3f}/5")

    rows2025 = [r for r in rows if r["date"].startswith("2025-")]
    if len(rows2025) > 1:
        pairs25 = [(rows2025[i]["nums"], rows2025[i + 1]["nums"]) for i in range(len(rows2025) - 1)]
        er25 = eval_rule(pairs25, pool_cross_opt_v1, "v1")
        mc25 = mc_mean_analytic(er25.pool_sizes)
        _, p95_25, _ = monte_carlo_baseline(pairs25, er25.pool_sizes, reps=3000, seed=42)
        print()
        print(f"=== ROMA | cross_opt_v1 | solo anno 2025 ({len(pairs25)} trans) ===")
        print(f"Hits: {er25.hits}/{len(pairs25)*5}")
        print(f"MC media: {mc25:.1f} | p95: {p95_25} | BATTE p95: {'SI' if er25.hits > p95_25 else 'NO'}")

    dates, by_date = load_draws_by_date(wide)
    prev: dict[str, list[int]] = {}
    for d in dates:
        if d >= "2025-01-01":
            break
        if "ROMA" in by_date.get(d, {}):
            prev["ROMA"] = by_date[d]["ROMA"]

    daily = []
    for d in sorted(x for x in dates if x.startswith("2025-")):
        if "ROMA" not in by_date[d]:
            continue
        if "ROMA" not in prev:
            prev["ROMA"] = by_date[d]["ROMA"]
            continue
        pool = pool_cross_opt_v1(prev["ROMA"])
        scores = score_cross_numbers(prev["ROMA"])
        picks = top_n_from_pool(pool, scores, 8)
        drawn = set(by_date[d]["ROMA"])
        hits = count_winning_ambi(picks, drawn)
        win = hits * (1.0 / 28) * 250
        daily.append({"date": d, "spent": 1.0, "won": win, "ambi": hits, "drawn": by_date[d]["ROMA"]})
        prev["ROMA"] = by_date[d]["ROMA"]

    spent = sum(x["spent"] for x in daily)
    won = sum(x["won"] for x in daily)
    ambi = sum(x["ambi"] for x in daily)

    print()
    print("=== ROMA | AMBI 2025 | cross_opt_v1 | 1 euro/estrazione | top-8 (28 coppie) ===")
    print(f"Giorni giocati: {len(daily)}")
    print(f"Spesa totale: {spent:.2f} euro")
    print(f"Vinto lordo: {won:.2f} euro")
    print(f"Netto lordo: {won - spent:+.2f} euro")
    print(f"Netto dopo 8% tasse: {won * 0.92 - spent:+.2f} euro")
    print(f"Ambi vinti: {ambi} ({ambi/len(daily):.3f}/giorno)")
    print(f"Per vincere in pari servirebbero ~{spent / 8.93:.1f} ambi (8.93 euro ciascuno)")

    by_m: dict[str, dict] = defaultdict(lambda: {"s": 0.0, "w": 0.0, "a": 0})
    for x in daily:
        m = x["date"][:7]
        by_m[m]["s"] += x["spent"]
        by_m[m]["w"] += x["won"]
        by_m[m]["a"] += x["ambi"]
    print("\nPer mese:")
    for m in sorted(by_m):
        net = by_m[m]["w"] - by_m[m]["s"]
        print(f"  {m}: speso {by_m[m]['s']:.0f} | vinto {by_m[m]['w']:.2f} | netto {net:+.2f} | ambi {by_m[m]['a']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
