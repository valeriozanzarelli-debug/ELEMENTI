#!/usr/bin/env python3
"""Cerca guadagno reale — cache picks per transizione."""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_ambi_2025 import load_draws_by_date, score_cross_numbers, top_n_from_pool
from simulate_sorti_proper import BETS, PAYOUT, TAX, check_win, random_hit_rate
from vertibile_rule_test import RULES

WHEELS = [
    "BARI", "CAGLIARI", "FIRENZE", "GENOVA", "MILANO", "NAPOLI",
    "PALERMO", "ROMA", "TORINO", "VENEZIA", "NAZIONALE",
]
METHODS = ["cross_opt_v1", "cross_twin11", "cross_only", "hybrid", "scored_top25"]
CONFIGS = [
    ("top_k", "estratto", 1),
    ("top_k", "ambo", 2),
    ("best_pair_top4", "ambo", 2),
    ("best_pair_top6", "ambo", 2),
    ("best_pair_top8", "ambo", 2),
    ("top_k", "terno", 3),
]
ALL_YEARS = list(range(2015, 2027))
TRAIN = set(range(2015, 2022))
TEST = set(range(2022, 2027))
BET_MAP = {b.name: b for b in BETS}


def best_pair(scores: dict[int, int], nums: list[int]) -> tuple[int, int]:
    ba, bb, bs = nums[0], nums[1], -1
    for a, b in combinations(nums, 2):
        s = scores.get(a, 0) + scores.get(b, 0)
        if s > bs:
            ba, bb, bs = a, b, s
    return ba, bb


def precompute_picks(prev: list[int], method: str) -> dict[str, list[int]]:
    pool_fn = RULES[method]
    pool = pool_fn(prev)
    scores = score_cross_numbers(prev)
    top8 = top_n_from_pool(pool, scores, 8)
    p4, p6 = best_pair(scores, top8[:4]), best_pair(scores, top8[:6])
    p8 = best_pair(scores, top8[:8])
    return {
        "estratto_top1": top8[:1],
        "ambo_top2": top8[:2],
        "ambo_pair4": list(p4),
        "ambo_pair6": list(p6),
        "ambo_pair8": list(p8),
        "terno_top3": top8[:3],
    }


PICK_KEY = {
    ("top_k", "estratto"): "estratto_top1",
    ("top_k", "ambo"): "ambo_top2",
    ("best_pair_top4", "ambo"): "ambo_pair4",
    ("best_pair_top6", "ambo"): "ambo_pair6",
    ("best_pair_top8", "ambo"): "ambo_pair8",
    ("top_k", "terno"): "terno_top3",
}


def build_wheel_data(wide: Path) -> dict[str, list[dict]]:
    dates, by_date = load_draws_by_date(wide)
    data: dict[str, list[dict]] = {w: [] for w in WHEELS}
    prev: dict[str, list[int]] = {}
    for d in dates:
        y = int(d[:4])
        for w in WHEELS:
            if w not in by_date.get(d, {}):
                continue
            drawn = by_date[d][w]
            if w in prev:
                picks_by_method = {m: precompute_picks(prev[w], m) for m in METHODS}
                data[w].append({"year": y, "drawn": set(drawn), "picks": picks_by_method})
            prev[w] = drawn
    return data


def sim(trans_list: list[dict], years: set[int] | None, method: str, strat: str, bet: str) -> dict:
    pk = PICK_KEY[(strat, bet)]
    b = BET_MAP[bet]
    spent = hits = 0
    won = 0.0
    for t in trans_list:
        if years and t["year"] not in years:
            continue
        picks = t["picks"][method].get(pk)
        if not picks or len(picks) < b.k:
            continue
        spent += 1
        if check_win(picks, t["drawn"], b):
            hits += 1
            won += b.mult
    p0 = random_hit_rate(b)
    hr = hits / spent if spent else 0
    return {
        "draws": spent, "hits": hits, "hit_rate": round(hr, 5),
        "random_hit_rate": round(p0, 5), "hit_vs_random_x": round(hr / p0, 2) if p0 else 0,
        "spent_eur": spent, "won_gross_eur": round(won, 2),
        "net_gross_eur": round(won - spent, 2),
        "p_value": round(1 - sum(math.exp(-spent*p0)*(spent*p0)**i/math.factorial(i) for i in range(hits)), 4) if spent and hits else 1.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/analysis/profit_zones.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    print("Precomputo transizioni...", flush=True)
    wheel_data = build_wheel_data(root / "data" / "draws_wide.csv")
    print("Simulo...", flush=True)

    full, stable, by_year = [], [], {}
    best_wheel: dict[str, dict] = {}

    for wheel in WHEELS:
        trans = [t for t in wheel_data[wheel] if t["year"] in ALL_YEARS]
        for method in METHODS:
            for strat, bet, _ in CONFIGS:
                r = sim(trans, set(ALL_YEARS), method, strat, bet)
                row = {"wheel": wheel, "method": method, "strategy": strat, "bet": bet, "period": "2015-2026", **r}
                full.append(row)
                if r["draws"] >= 100 and r["net_gross_eur"] > best_wheel.get(wheel, {}).get("net_gross_eur", -1e9):
                    best_wheel[wheel] = row

                tr = sim(trans, TRAIN, method, strat, bet)
                te = sim(trans, TEST, method, strat, bet)
                if tr["draws"] >= 100 and te["draws"] >= 50:
                    if tr["net_gross_eur"] > 0 and te["net_gross_eur"] > 0:
                        if tr["hit_rate"] > tr["random_hit_rate"] and te["hit_rate"] > te["random_hit_rate"]:
                            stable.append({
                                "wheel": wheel, "method": method, "strategy": strat, "bet": bet,
                                "train_net": tr["net_gross_eur"], "test_net": te["net_gross_eur"],
                                "combined_net": tr["net_gross_eur"] + te["net_gross_eur"],
                                "test_hits": te["hits"], "test_draws": te["draws"],
                                "test_hit_rate": te["hit_rate"],
                            })

                if bet == "ambo" and method == "cross_opt_v1" and strat == "top_k":
                    prof = [y for y in ALL_YEARS if sim(trans, {y}, method, strat, bet)["net_gross_eur"] > 0]
                    by_year[wheel] = prof

    profitable = sorted(
        [r for r in full if r["draws"] >= 100 and r["net_gross_eur"] > 0 and r["hit_rate"] > r["random_hit_rate"]],
        key=lambda x: -x["net_gross_eur"],
    )
    stable.sort(key=lambda x: -x["combined_net"])

    report = {
        "random_ev": {b.name: round(random_hit_rate(b) * PAYOUT[b.name] - 1, 4) for b in BETS},
        "top_profitable": profitable[:30],
        "stable_oos": stable[:20],
        "best_per_wheel": [best_wheel[w] for w in WHEELS],
        "ambo_profitable_years": {w: by_year.get(w, []) for w in ["TORINO", "ROMA", "BARI", "PALERMO", "NAPOLI"]},
    }
    out = root / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nEV random:", report["random_ev"])
    print("\nTOP 20 profitto 2015-26:")
    for r in profitable[:20]:
        print(f"  {r['wheel']:<10} {r['bet']:<8} {r['method']:<14} {r['strategy']:<16} "
              f"{r['hits']}/{r['draws']} {r['hit_rate']*100:.2f}% net {r['net_gross_eur']:+.0f} p={r['p_value']:.3f}")

    print("\nSTABILI (train+test):")
    for r in stable[:12] or [{"msg": "nessuno"}]:
        if "msg" in r:
            print("  nessun caso profittevole in entrambi i periodi")
        else:
            print(f"  {r['wheel']} {r['bet']} {r['strategy']}: train {r['train_net']:+.0f} test {r['test_net']:+.0f}")

    print("\nMIGLIOR per RUOTA:")
    for w in WHEELS:
        r = best_wheel[w]
        m = " ***" if r["net_gross_eur"] > 0 else ""
        print(f"  {w:<10} {r['bet']:<8} {r['method']:<14} net {r['net_gross_eur']:+.0f}{m}")

    print("\nAMBO cross_opt_v1 anni in +:")
    for w, ys in report["ambo_profitable_years"].items():
        tot = next((x for x in full if x["wheel"]==w and x["bet"]=="ambo" and x["method"]=="cross_opt_v1" and x["strategy"]=="top_k"), None)
        if tot:
            print(f"  {w}: {ys} | totale net {tot['net_gross_eur']:+.0f}")
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
