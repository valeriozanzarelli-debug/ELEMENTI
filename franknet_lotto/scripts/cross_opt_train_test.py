#!/usr/bin/env python3
"""
Train/test cross_opt_v1 su 500+500 transizioni per ruota.
- 10 ruote: train 500 (blocco piu vecchio) -> ottimizza -> test 500 (blocco piu recente)
- 1 ruota hold-out (MILANO): mai usata in training, validazione finale
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from vertibile_rule_test import (
    adj_diffs,
    add,
    comp,
    eval_rule,
    is_twin,
    load_wheel,
    mc_mean_analytic,
    monte_carlo_baseline,
    neighbors,
    pool_cross_opt_v1,
    vertibile,
)

TRAIN_WHEELS = [
    "BARI",
    "CAGLIARI",
    "FIRENZE",
    "GENOVA",
    "NAPOLI",
    "PALERMO",
    "ROMA",
    "TORINO",
    "VENEZIA",
    "NAZIONALE",
]
HOLDOUT_WHEEL = "MILANO"
N_TRANS = 500


@dataclass(frozen=True)
class CrossOptConfig:
    """Parametri del metodo cross ottimizzabile."""

    twin11: bool = True
    twin_also_pm1: bool = False  # gemelli: sia ±11 che ±1
    add_comp_pure: bool = True
    add_vert_pure: bool = True
    add_vert_diff_neighbors: bool = True
    comp_cross_only: bool = False  # solo comp+diff, no orig+diff
    add_special_4590: bool = False
    twin_shift_quintina: bool = False  # se gemello in quintina: m±11 per tutti


def pool_cross_configurable(nums: list[int], cfg: CrossOptConfig) -> set[int]:
    pool: set[int] = set()
    diffs = adj_diffs(nums)

    def nb(base: int) -> None:
        if cfg.twin_also_pm1 and is_twin(base):
            for step in (1, 11):
                for x in (base - step, base + step):
                    if 1 <= x <= 90:
                        add(pool, x)
        else:
            for x in neighbors(base, cfg.twin11):
                add(pool, x)

    for n in nums:
        cn = comp(n)
        vn = vertibile(n)
        for d in diffs:
            if not cfg.comp_cross_only:
                add(pool, n + d)
                add(pool, abs(n - d))
            if cn is not None:
                add(pool, cn + d)
                add(pool, abs(cn - d))
        for base in filter(None, [n, cn, vn]):
            nb(base)

    if cfg.add_comp_pure:
        for n in nums:
            add(pool, comp(n))
    if cfg.add_vert_pure:
        for n in nums:
            add(pool, vertibile(n))

    if cfg.add_vert_diff_neighbors:
        for d in diffs:
            vd = vertibile(d)
            if vd:
                nb(vd)

    if cfg.add_special_4590:
        if 45 in nums:
            add(pool, 45)
            for n in nums:
                add(pool, abs(n - 45))
        if 90 in nums:
            add(pool, 9)
            for n in nums:
                add(pool, abs(n - 9))

    if cfg.twin_shift_quintina and any(is_twin(n) for n in nums):
        for m in nums:
            for step in (11, 22):
                add(pool, m + step)
                add(pool, m - step)

    return pool


def make_fn(cfg: CrossOptConfig):
    return lambda nums: pool_cross_configurable(nums, cfg)


def slice_pairs(rows: list[dict], offset_end: int, n_trans: int) -> list[tuple[list[int], list[int]]]:
    """offset_end=501 -> ultimi 500 trans; offset_end=1001 -> 500 prima di quelle."""
    end = len(rows) - offset_end + 1  # +1 per avere n_trans coppie
    start = end - n_trans
    if start < 0:
        raise ValueError(f"Non bastano estrazioni: start={start}")
    chunk = rows[start:end + 1]
    return [(chunk[i]["nums"], chunk[i + 1]["nums"]) for i in range(n_trans)]


def slice_period(rows: list[dict], offset_end: int, n_trans: int) -> list[str]:
    end = len(rows) - offset_end + 1
    start = end - n_trans
    chunk = rows[start : end + 1]
    return [chunk[0]["date"], chunk[-1]["date"]]


def eval_method_on_wheels(
    wide: Path,
    wheels: list[str],
    rule_fn,
    offset_end: int,
    n_trans: int,
    mc_reps: int,
    seed: int,
    label: str = "",
) -> dict:
    results = {}
    tot_hits = 0
    tot_mc = 0.0
    tot_p95 = 0
    for i, wheel in enumerate(wheels):
        rows = load_wheel(wide, wheel)
        try:
            pairs = slice_pairs(rows, offset_end, n_trans)
            period = slice_period(rows, offset_end, n_trans)
        except ValueError as e:
            results[wheel] = {"error": str(e), "n_rows": len(rows)}
            continue
        er = eval_rule(pairs, rule_fn, wheel)
        mc_mean = mc_mean_analytic(er.pool_sizes)
        _, p95, _ = monte_carlo_baseline(
            pairs, er.pool_sizes, reps=mc_reps, seed=seed + hash(wheel) % 10000
        )
        tot_hits += er.hits
        tot_mc += mc_mean
        tot_p95 += p95
        results[wheel] = {
            "period": period,
            "hits": er.hits,
            "total": er.total,
            "mc_mean": mc_mean,
            "mc_p95": p95,
            "beats_p95": er.hits > p95,
            "margin_mean": er.hits - mc_mean,
            "margin_p95": er.hits - p95,
            "avg_pool": er.avg_pool,
            "avg_per_draw": sum(er.per_draw) / len(er.per_draw),
        }
        if label:
            print(f"    [{label}] {i+1}/{len(wheels)} {wheel} hits={er.hits} p95={p95}", flush=True)
    n_ok = sum(1 for r in results.values() if "error" not in r)
    return {
        "wheels": results,
        "aggregate": {
            "wheels_ok": n_ok,
            "hits": tot_hits,
            "mc_mean": tot_mc,
            "mc_p95": tot_p95,
            "margin_mean": tot_hits - tot_mc,
            "margin_p95": tot_hits - tot_p95,
            "beats_p95_wheels": sum(
                1 for r in results.values() if r.get("beats_p95")
            ),
        },
    }


def grid_configs() -> list[tuple[str, CrossOptConfig]]:
    """Griglia ridotta ma esaustiva sulle leve principali."""
    configs: list[tuple[str, CrossOptConfig]] = [
        ("cross_opt_v1_baseline", CrossOptConfig()),
    ]
    toggles = {
        "twin_pm1": {"twin_also_pm1": True},
        "special": {"add_special_4590": True},
        "comp_only": {"comp_cross_only": True},
        "no_vdiff": {"add_vert_diff_neighbors": False},
        "twin_shift": {"twin_shift_quintina": True},
    }
    # singole aggiunte rispetto baseline
    for name, kw in toggles.items():
        configs.append((f"v1+{name}", CrossOptConfig(**kw)))
    # combo promettenti
    combos = [
        ("v2_twin_pm1+special", {"twin_also_pm1": True, "add_special_4590": True}),
        ("v2_twin_pm1+shift", {"twin_also_pm1": True, "twin_shift_quintina": True}),
        (
            "v2_comp_only+pm1+special",
            {
                "comp_cross_only": True,
                "twin_also_pm1": True,
                "add_special_4590": True,
            },
        ),
        (
            "v2_full_tuned",
            {
                "twin_also_pm1": True,
                "add_special_4590": True,
                "twin_shift_quintina": True,
            },
        ),
        (
            "v2_pm1_no_orig_cross",
            {
                "twin_also_pm1": True,
                "comp_cross_only": True,
                "add_vert_diff_neighbors": True,
            },
        ),
    ]
    for name, kw in combos:
        base = asdict(CrossOptConfig())
        base.update(kw)
        configs.append((name, CrossOptConfig(**base)))
    return configs


def eval_fast(
    wide: Path,
    wheels: list[str],
    rule_fn,
    offset_end: int,
    n_trans: int,
) -> dict:
    """Solo hits e MC analitico (niente Monte Carlo) — per grid search."""
    results = {}
    tot_hits = 0
    tot_mc = 0.0
    for wheel in wheels:
        rows = load_wheel(wide, wheel)
        try:
            pairs = slice_pairs(rows, offset_end, n_trans)
            period = slice_period(rows, offset_end, n_trans)
        except ValueError as e:
            results[wheel] = {"error": str(e)}
            continue
        er = eval_rule(pairs, rule_fn, wheel)
        mc_mean = mc_mean_analytic(er.pool_sizes)
        tot_hits += er.hits
        tot_mc += mc_mean
        results[wheel] = {
            "period": period,
            "hits": er.hits,
            "mc_mean": mc_mean,
            "margin_mean": er.hits - mc_mean,
            "avg_pool": er.avg_pool,
        }
    return {
        "wheels": results,
        "aggregate": {
            "hits": tot_hits,
            "mc_mean": tot_mc,
            "margin_mean": tot_hits - tot_mc,
        },
    }


def pick_best_on_train(
    wide: Path,
    wheels: list[str],
    offset_train: int,
    n_trans: int,
) -> tuple[str, CrossOptConfig, list[dict]]:
    leaderboard = []
    configs = grid_configs()
    for i, (name, cfg) in enumerate(configs):
        fn = make_fn(cfg)
        rep = eval_fast(wide, wheels, fn, offset_train, n_trans)
        agg = rep["aggregate"]
        leaderboard.append(
            {
                "name": name,
                "config": asdict(cfg),
                "hits": agg["hits"],
                "margin_mean": agg["margin_mean"],
            }
        )
        print(f"  grid [{i+1}/{len(configs)}] {name}: hits={agg['hits']} marg={agg['margin_mean']:+.0f}", flush=True)
    leaderboard.sort(key=lambda x: (x["margin_mean"], x["hits"]), reverse=True)
    best_name = leaderboard[0]["name"]
    best_cfg = CrossOptConfig(**leaderboard[0]["config"])
    return best_name, best_cfg, leaderboard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-reps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", default=HOLDOUT_WHEEL)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    wide = root / "data" / "draws_wide.csv"

    # Blocchi temporali (ultime 1000 transizioni per ruota):
    #   TEST  = ultime 500   (offset_end=501)
    #   TRAIN = 500 precedenti (offset_end=1001)
    offset_test = 501
    offset_train = 1001

    print("=" * 70)
    print("FASE 0: cross_opt_v1 baseline")
    print(f"  Train wheels (10): {', '.join(TRAIN_WHEELS)}")
    print(f"  Hold-out: {args.holdout}")
    print(f"  Train = 500 transizioni piu vecchie nel blocco -1000")
    print(f"  Test  = 500 transizioni piu recenti")
    print("=" * 70)

    baseline_fn = pool_cross_opt_v1

    print("\n--- cross_opt_v1 su TRAIN (500 x 10 ruote) ---")
    train_v1 = eval_method_on_wheels(
        wide, TRAIN_WHEELS, baseline_fn, offset_train, N_TRANS, args.mc_reps, args.seed, "v1-train"
    )
    agg = train_v1["aggregate"]
    print(
        f"  hits={agg['hits']}  MCmean={agg['mc_mean']:.0f}  "
        f"p95={agg['mc_p95']:.0f}  margine_p95={agg['margin_p95']:+.0f}  "
        f"ruote>P95={agg['beats_p95_wheels']}/10"
    )

    print("\n--- cross_opt_v1 su TEST OOS (500 x 10 ruote) ---")
    test_v1 = eval_method_on_wheels(
        wide, TRAIN_WHEELS, baseline_fn, offset_test, N_TRANS, args.mc_reps, args.seed, "v1-test"
    )
    agg_t = test_v1["aggregate"]
    print(
        f"  hits={agg_t['hits']}  MCmean={agg_t['mc_mean']:.0f}  "
        f"p95={agg_t['mc_p95']:.0f}  margine_p95={agg_t['margin_p95']:+.0f}  "
        f"ruote>P95={agg_t['beats_p95_wheels']}/10"
    )

    print("\n" + "=" * 70)
    print("FASE 1: grid search su TRAIN (solo 10 ruote, MILANO esclusa)")
    print("=" * 70)
    best_name, best_cfg, leaderboard = pick_best_on_train(
        wide, TRAIN_WHEELS, offset_train, N_TRANS
    )
    best_fn = make_fn(best_cfg)
    print("\nTop 5 configurazioni su TRAIN (selezione su margine vs MC analitico):")
    print(f"  {'Nome':<28} {'Hits':>6} {'MargMean':>9}")
    for row in leaderboard[:5]:
        print(f"  {row['name']:<28} {row['hits']:>6} {row['margin_mean']:>+9.0f}")
    print(f"\nMigliore: {best_name}")
    print(f"  Config: {asdict(best_cfg)}")

    print("\n--- Metodo ottimizzato su TRAIN (500 x 10 ruote, conferma p95) ---")
    train_opt = eval_method_on_wheels(
        wide, TRAIN_WHEELS, best_fn, offset_train, N_TRANS, args.mc_reps, args.seed, "opt-train"
    )
    agg_ot = train_opt["aggregate"]
    print(
        f"  hits={agg_ot['hits']}  MCmean={agg_ot['mc_mean']:.0f}  "
        f"p95={agg_ot['mc_p95']:.0f}  margine_p95={agg_ot['margin_p95']:+.0f}  "
        f"ruote>P95={agg_ot['beats_p95_wheels']}/10"
    )

    print("\n--- Metodo ottimizzato su TEST OOS (500 x 10 ruote) ---")
    test_opt = eval_method_on_wheels(
        wide, TRAIN_WHEELS, best_fn, offset_test, N_TRANS, args.mc_reps, args.seed, "opt-test"
    )
    agg_o = test_opt["aggregate"]
    print(
        f"  hits={agg_o['hits']}  MCmean={agg_o['mc_mean']:.0f}  "
        f"p95={agg_o['mc_p95']:.0f}  margine_p95={agg_o['margin_p95']:+.0f}  "
        f"ruote>P95={agg_o['beats_p95_wheels']}/10"
    )

    print("\n" + "=" * 70)
    print(f"FASE 2: VALIDAZIONE FINALE su {args.holdout} (mai vista in training)")
    print("=" * 70)
    holdout_train = eval_method_on_wheels(
        wide, [args.holdout], baseline_fn, offset_train, N_TRANS, args.mc_reps, args.seed, "hold-v1-tr"
    )
    holdout_test_v1 = eval_method_on_wheels(
        wide, [args.holdout], baseline_fn, offset_test, N_TRANS, args.mc_reps, args.seed, "hold-v1-te"
    )
    holdout_test_opt = eval_method_on_wheels(
        wide, [args.holdout], best_fn, offset_test, N_TRANS, args.mc_reps, args.seed, "hold-opt-te"
    )
    holdout_train_opt = eval_method_on_wheels(
        wide, [args.holdout], best_fn, offset_train, N_TRANS, args.mc_reps, args.seed, "hold-opt-tr"
    )

    for label, rep in [
        ("v1 train-500", holdout_train),
        ("v1 test-500", holdout_test_v1),
        ("opt train-500", holdout_train_opt),
        ("opt test-500", holdout_test_opt),
    ]:
        w = rep["wheels"][args.holdout]
        print(
            f"  {label}: hits={w['hits']} p95={w['mc_p95']:.0f} "
            f"margine_p95={w['margin_p95']:+.0f} batte_p95={w['beats_p95']}"
        )

    # tabella per ruota su TEST
    print("\n--- Dettaglio TEST 500 per ruota (v1 vs ottimizzato) ---")
    print(f"  {'Ruota':<12} {'v1 hit':>7} {'v1/p95':>8} {'opt hit':>8} {'opt/p95':>8}")
    for wheel in TRAIN_WHEELS:
        a = test_v1["wheels"].get(wheel, {})
        b = test_opt["wheels"].get(wheel, {})
        if "error" in a:
            print(f"  {wheel:<12} ERRORE")
            continue
        print(
            f"  {wheel:<12} {a['hits']:>7} {a['hits']}/{a['mc_p95']:.0f} "
            f"{b['hits']:>8} {b['hits']}/{b['mc_p95']:.0f}"
        )

    report = {
        "holdout_wheel": args.holdout,
        "train_wheels": TRAIN_WHEELS,
        "n_transitions": N_TRANS,
        "offsets": {"train": offset_train, "test": offset_test},
        "cross_opt_v1": {
            "train_10w": train_v1,
            "test_10w": test_v1,
        },
        "optimized": {
            "name": best_name,
            "config": asdict(best_cfg),
            "leaderboard_train_top10": leaderboard[:10],
            "train_10w": train_opt,
            "test_10w": test_opt,
            "vs_v1_test_delta_hits": test_opt["aggregate"]["hits"] - test_v1["aggregate"]["hits"],
            "vs_v1_test_delta_margin_p95": (
                test_opt["aggregate"]["margin_p95"] - test_v1["aggregate"]["margin_p95"]
            ),
        },
        "holdout_final": {
            "v1_train": holdout_train,
            "v1_test": holdout_test_v1,
            "opt_train": holdout_train_opt,
            "opt_test": holdout_test_opt,
        },
    }
    out = root / "data" / "analysis" / "cross_opt_train_test_500.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
