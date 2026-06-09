#!/usr/bin/env python3
"""
Report dettagliato metodi cross_only / cross_twin11 + varianti ottimizzate.
Mostra calcoli esatti e validazione su N transizioni BARI.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from vertibile_rule_test import (
    RULES,
    adj_diffs,
    comp,
    eval_rule,
    is_special_no_comp,
    is_twin,
    load_wheel,
    mc_mean_analytic,
    monte_carlo_baseline,
    neighbors,
    pool_cross_only,
    pool_cross_opt_v1,
    pool_scored_topk,
    vertibile,
)


def add(pool: set[int], n: int | None) -> None:
    if n is not None and 1 <= n <= 90:
        pool.add(n)


def pool_cross_traced(nums: list[int], *, twin11: bool) -> tuple[set[int], dict[int, list[str]]]:
    """Come cross_only ma con ricetta per ogni numero nel pool."""
    pool: set[int] = set()
    recipes: dict[int, list[str]] = defaultdict(list)
    diffs = adj_diffs(nums)

    def put(n: int | None, recipe: str) -> None:
        if n is not None and 1 <= n <= 90:
            pool.add(n)
            if recipe not in recipes[n]:
                recipes[n].append(recipe)

    diff_labels = []
    for i, d in enumerate(diffs):
        a, b = nums[i], nums[i + 1]
        diff_labels.append(f"d{i+1}=|{a}-{b}|={d}")

    for idx, n in enumerate(nums, 1):
        cn = comp(n)
        cn_txt = "N/A" if cn is None else str(cn)
        if is_special_no_comp(n):
            cn_txt += f" ({n} speciale)"
        for j, d in enumerate(diffs, 1):
            put(n + d, f"orig[{idx}]({n}) + d{j}({d}) = {n+d}")
            put(abs(n - d), f"|orig[{idx}]({n}) - d{j}({d})| = {abs(n-d)}")
            if cn is not None:
                put(cn + d, f"comp(orig[{idx}])={cn} + d{j}({d}) = {cn+d}")
                put(abs(cn - d), f"|comp(orig[{idx}])={cn} - d{j}({d})| = {abs(cn-d)}")
        for base_name, base in [
            (f"orig[{idx}]={n}", n),
            (f"comp(orig[{idx}])={cn_txt}", cn),
            (f"vert(orig[{idx}])", vertibile(n)),
        ]:
            if base is None:
                continue
            step = 11 if (twin11 and is_twin(base)) else 1
            for sign, delta in [("-", -step), ("+", step)]:
                x = base + delta
                if 1 <= x <= 90:
                    put(x, f"{base_name} {sign} {step} = {x}")

    return pool, dict(recipes)


def pool_cross_opt_v2(nums: list[int]) -> set[int]:
    """Solo incroci comp+diff (peso maggiore nell'esempio utente) + vicini gemello."""
    pool: set[int] = set()
    diffs = adj_diffs(nums)
    for n in nums:
        cn = comp(n)
        if cn is not None:
            for d in diffs:
                add(pool, cn + d)
                add(pool, abs(cn - d))
        for base in filter(None, [n, cn, vertibile(n)]):
            for x in neighbors(base, True):
                add(pool, x)
    return pool


def pool_cross_opt_v3(nums: list[int]) -> set[int]:
    """Incroci + se 45/90 presenti espansione speciale."""
    pool = pool_cross_only(nums, twin11=True)
    if 45 in nums:
        for n in nums:
            add(pool, 45)
            add(pool, abs(n - 45))
    if 90 in nums:
        add(pool, 9)
        for n in nums:
            add(pool, abs(n - 9))
    return pool


def pool_cross_scored(nums: list[int], k: int) -> set[int]:
    """Punteggio solo su ricette cross (no orig diretto)."""
    scores: defaultdict[int, int] = defaultdict(int)

    def bump(n: int | None, w: int) -> None:
        if n is not None and 1 <= n <= 90:
            scores[n] += w

    diffs = adj_diffs(nums)
    for idx, n in enumerate(nums, 1):
        cn = comp(n)
        for j, d in enumerate(diffs, 1):
            bump(n + d, 2)
            bump(abs(n - d), 1)
            if cn is not None:
                bump(cn + d, 4)  # comp+diff pesa di piu
                bump(abs(cn - d), 2)
        for base in filter(None, [n, cn, vertibile(n)]):
            for x in neighbors(base, True):
                bump(x, 2 if is_twin(base) else 1)
    if 45 in nums:
        bump(45, 3)
    if 90 in nums:
        bump(9, 2)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return {n for n, _ in ranked[:k]}


OPT_RULES: dict[str, object] = {
    "cross_only": RULES["cross_only"],
    "cross_twin11": RULES["cross_twin11"],
    "cross_opt_v1": RULES["cross_opt_v1"],
    "cross_opt_v2": pool_cross_opt_v2,
    "cross_opt_v3": pool_cross_opt_v3,
    "cross_scored_42": lambda n: pool_cross_scored(n, 42),
    "cross_scored_48": lambda n: pool_cross_scored(n, 48),
    "cross_scored_52": lambda n: pool_cross_scored(n, 52),
    "scored_top19": RULES["scored_top19"],
}


def evaluate_all(
    pairs: list[tuple[list[int], list[int]]],
    dates: list[str],
    mc_reps: int,
    seed: int,
) -> list[dict]:
    out = []
    for name, fn in OPT_RULES.items():
        er = eval_rule(pairs, fn, name)
        mc_mean = mc_mean_analytic(er.pool_sizes)
        _, p95, _ = monte_carlo_baseline(pairs, er.pool_sizes, reps=mc_reps, seed=seed)
        out.append(
            {
                "name": name,
                "hits": er.hits,
                "total": er.total,
                "avg_pool": er.avg_pool,
                "mc_mean": mc_mean,
                "mc_p95": p95,
                "margin_mean": er.hits - mc_mean,
                "beats_p95": er.hits > p95,
                "per_draw": er.per_draw,
                "pool_sizes": er.pool_sizes,
            }
        )
    return sorted(out, key=lambda x: (-int(x["beats_p95"]), -x["hits"], -x["margin_mean"]))


def per_draw_table(
    pairs: list[tuple[list[int], list[int]]],
    dates: list[str],
    fn,
    name: str,
) -> list[dict]:
    rows = []
    for i, (prev, nxt) in enumerate(pairs):
        pool, recipes = pool_cross_traced(prev, twin11=(name == "cross_twin11"))
        if name not in ("cross_only", "cross_twin11"):
            pool = fn(prev)
            recipes = {}
        hits = []
        misses = []
        for pos, t in enumerate(nxt, 1):
            item = {"pos": pos, "num": t}
            if t in pool:
                item["ok"] = True
                item["recipe"] = (recipes.get(t) or ["nel pool"])[0]
                hits.append(item)
            else:
                item["ok"] = False
                misses.append(item)
        rows.append(
            {
                "date_from": dates[i],
                "date_to": dates[i + 1],
                "prev": prev,
                "next": nxt,
                "diffs": adj_diffs(prev),
                "pool_size": len(pool),
                "hits": len(hits),
                "hit_detail": hits,
                "miss_detail": misses,
            }
        )
    return rows


def example_calculation(nums: list[int], next_nums: list[int], twin11: bool) -> dict:
    pool, recipes = pool_cross_traced(nums, twin11=twin11)
    return {
        "input": nums,
        "diffs": adj_diffs(nums),
        "diff_formulas": [
            f"|{nums[i]}-{nums[i+1]}| = {adj_diffs(nums)[i]}" for i in range(4)
        ],
        "pool_size": len(pool),
        "pool_sorted": sorted(pool),
        "next": next_nums,
        "matches": [
            {
                "pos": i + 1,
                "num": t,
                "hit": t in pool,
                "recipes": recipes.get(t, []),
            }
            for i, t in enumerate(next_nums)
        ],
    }


ALL_WHEELS = [
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


def run_wheel_report(
    wide: Path,
    wheel: str,
    transitions: int,
    mc_reps: int,
    seed: int,
) -> dict:
    rows = load_wheel(wide, wheel)
    need = transitions + 1
    if len(rows) < need:
        return {
            "wheel": wheel,
            "error": f"solo {len(rows)} estrazioni complete, servono {need}",
        }
    chunk = rows[-need:]
    pairs = [(chunk[i]["nums"], chunk[i + 1]["nums"]) for i in range(transitions)]
    dates = [r["date"] for r in chunk]
    summary = {}
    for m in ("cross_only", "cross_twin11", "cross_opt_v1"):
        fn = OPT_RULES[m]
        er = eval_rule(pairs, fn, m)
        mc_mean = mc_mean_analytic(er.pool_sizes)
        _, p95, _ = monte_carlo_baseline(pairs, er.pool_sizes, reps=mc_reps, seed=seed)
        summary[m] = {
            "hits": er.hits,
            "total": er.total,
            "mc_mean": mc_mean,
            "mc_p95": p95,
            "beats_mc_mean": er.hits > mc_mean,
            "beats_p95": er.hits > p95,
            "margin_vs_mean": er.hits - mc_mean,
            "margin_vs_p95": er.hits - p95,
            "avg_pool": er.avg_pool,
            "avg_per_draw": statistics.mean(er.per_draw),
        }

    best_m = max(summary, key=lambda k: (int(summary[k]["beats_p95"]), summary[k]["hits"]))
    return {
        "wheel": wheel,
        "transitions": transitions,
        "period": [dates[0], dates[-1]],
        "n_draws_available": len(rows),
        "best_method": best_m,
        "methods": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", default="BARI")
    ap.add_argument("--all-wheels", action="store_true")
    ap.add_argument("--transitions", type=int, default=200)
    ap.add_argument("--mc-reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/analysis/cross_method_report_200.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    wide = root / "data" / "draws_wide.csv"

    if args.all_wheels:
        reports = []
        print(f"Tutte le ruote | {args.transitions} transizioni | MC reps={args.mc_reps}")
        print()
        print(
            f"{'Ruota':<12} {'cross_only':>12} {'cross_twin11':>12} "
            f"{'cross_opt_v1':>12} {'best P95':>10}"
        )
        print("-" * 62)
        for wheel in ALL_WHEELS:
            rep = run_wheel_report(wide, wheel, args.transitions, args.mc_reps, args.seed)
            reports.append(rep)
            if "error" in rep:
                print(f"{wheel:<12} ERRORE: {rep['error']}")
                continue
            co = rep["methods"]["cross_only"]
            ct = rep["methods"]["cross_twin11"]
            ov = rep["methods"]["cross_opt_v1"]
            p95_flag = "SI" if ov["beats_p95"] else ("~" if co["beats_p95"] or ct["beats_p95"] else "no")
            print(
                f"{wheel:<12} "
                f"{co['hits']:>4}/{co['mc_p95']:<4} "
                f"{ct['hits']:>4}/{ct['mc_p95']:<4} "
                f"{ov['hits']:>4}/{ov['mc_p95']:<4} "
                f"{p95_flag:>10}"
            )

        # totale aggregato tutte le ruote
        ok_methods = ["cross_only", "cross_twin11", "cross_opt_v1"]
        totals = {m: {"hits": 0, "mc_mean": 0.0, "mc_p95": 0, "beats_p95": 0} for m in ok_methods}
        valid = [r for r in reports if "error" not in r]
        for r in valid:
            for m in ok_methods:
                totals[m]["hits"] += r["methods"][m]["hits"]
                totals[m]["mc_mean"] += r["methods"][m]["mc_mean"]
                totals[m]["mc_p95"] += r["methods"][m]["mc_p95"]
                if r["methods"][m]["beats_p95"]:
                    totals[m]["beats_p95"] += 1

        print()
        print("TOTALE 11 ruote (somma hit / somma p95):")
        for m in ok_methods:
            t = totals[m]
            print(
                f"  {m:<16} hits={t['hits']}  MCmean={t['mc_mean']:.0f}  "
                f"p95={t['mc_p95']:.0f}  margine_p95={t['hits']-t['mc_p95']:+.0f}  "
                f"ruote_batte_p95={t['beats_p95']}/{len(valid)}"
            )

        out_path = root / "data" / "analysis" / "cross_method_all_wheels_200.json"
        out_path.write_text(
            json.dumps({"transitions": args.transitions, "wheels": reports, "totals": totals}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nReport JSON: {out_path}")
        return 0

    rows = load_wheel(wide, args.wheel)
    need = args.transitions + 1
    chunk = rows[-need:]
    pairs = [(chunk[i]["nums"], chunk[i + 1]["nums"]) for i in range(args.transitions)]
    dates = [r["date"] for r in chunk]

    results = evaluate_all(pairs, dates, args.mc_reps, args.seed)

    print(f"BARI | {args.transitions} transizioni | {dates[0]} -> {dates[-1]}")
    print()
    print(f"{'Metodo':<18} {'Hits':>5} {'Pool':>6} {'MCavg':>7} {'MCp95':>6} {'Marg':>6} {'P95':>4}")
    print("-" * 58)
    for r in results:
        flag = "SI" if r["beats_p95"] else "no"
        star = "*" if r["beats_p95"] else " "
        print(
            f"{star}{r['name']:<17} {r['hits']:>5} {r['avg_pool']:>6.1f} "
            f"{r['mc_mean']:>7.1f} {r['mc_p95']:>6.0f} {r['margin_mean']:>+6.1f} {flag:>4}"
        )

    best = results[0]
    best_fn = OPT_RULES[best["name"]]
    dist = statistics.mean(best["per_draw"])
    print()
    print(
        f"Migliore: {best['name']} -> {best['hits']}/{best['total']} "
        f"({100*best['hits']/best['total']:.1f}%), media {dist:.2f}/5 per estrazione"
    )
    print(f"  MC media={best['mc_mean']:.1f}, p95={best['mc_p95']:.0f}, margine vs media={best['margin_mean']:+.1f}")

    # esempio didattico: ultima transizione con cross_twin11
    prev, nxt = pairs[-1]
    ex = example_calculation(prev, nxt, twin11=True)
    ex_user = example_calculation([11, 36, 33, 25, 57], [79, 44, 34, 76, 86], twin11=True)

    # tabella riassuntiva 200 per cross_only e cross_twin11
    summary_200 = {}
    for m in ("cross_only", "cross_twin11", results[0]["name"]):
        if m not in OPT_RULES:
            continue
        fn = OPT_RULES[m]
        er = eval_rule(pairs, fn, m)
        _, p95, _ = monte_carlo_baseline(pairs, er.pool_sizes, reps=args.mc_reps, seed=args.seed)
        draws_5 = sum(1 for h in er.per_draw if h == 5)
        draws_0 = sum(1 for h in er.per_draw if h == 0)
        summary_200[m] = {
            "hits": er.hits,
            "mc_mean": mc_mean_analytic(er.pool_sizes),
            "mc_p95": p95,
            "avg_per_draw": statistics.mean(er.per_draw),
            "draws_with_5": draws_5,
            "draws_with_0": draws_0,
            "per_draw_histogram": {str(k): er.per_draw.count(k) for k in range(6)},
        }

    report = {
        "wheel": args.wheel,
        "transitions": args.transitions,
        "period": [dates[0], dates[-1]],
        "results": results,
        "summary_200": summary_200,
        "example_user": ex_user,
        "example_last_draw": ex,
    }

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
