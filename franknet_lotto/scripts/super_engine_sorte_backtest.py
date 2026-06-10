#!/usr/bin/env python3
"""
Backtest super motore (cyclical_science_engine) su AMBO, TERNO, QUATERNA.
Regole ADM: 1€ intero sulla combinazione secca (top-K o best-pair / combo).

Train: grid pesi per sorte su 1871-2014
Test:  2015-2026 — confronto cross vs science ottimizzato

Output: data/analysis/cyclical_science/sorte_backtest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

from cyclical_science_engine import (
    ALL_WHEELS,
    DrawRecord,
    PhaseCooccurrence,
    WeightConfig,
    fit_models,
    load_records,
    pairs_from,
    science_score,
    select_cinquina_combo,
)
from simulate_ambi_2025 import score_cross_numbers, top_n_from_pool
from simulate_sorti_proper import TAX, check_win, BetType
from vertibile_rule_test import pool_cross_opt_v1

SORTE = {
    "ambo": BetType("ambo", 2, 250.0),
    "terno": BetType("terno", 3, 4500.0),
    "quaterna": BetType("quaterna", 4, 120_000.0),
}

# Override empirici stabili (train+letteratura progetto)
MODE_OVERRIDE: dict[tuple[str, str], str] = {
    ("NAZIONALE", "ambo"): "cross",
}

DEFAULT_SCIENCE = WeightConfig(
    w_harmonic=120,
    w_lattice=20,
    w_knn=40,
    w_anchor_echo=14,
    w_cooc=60,
    score_all_90=True,
    pick_pool_size=20,
)


def best_pair(scores: dict[int, float], candidates: list[int]) -> list[int]:
    ba, bb, bs = candidates[0], candidates[1], -1.0
    for a, b in combinations(candidates, 2):
        s = scores.get(a, 0) + scores.get(b, 0)
        if s > bs:
            ba, bb, bs = a, b, s
    return [ba, bb]


def select_combo_k(
    scores: dict[int, float],
    phase_vec: tuple[int, ...],
    cooc: PhaseCooccurrence,
    cfg: WeightConfig,
    k: int,
) -> list[int]:
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    if len(ranked) < k:
        return [n for n, _ in ranked]
    m = min(cfg.pick_pool_size, len(ranked))
    candidates = [n for n, _ in ranked[:m]]
    if k == 5:
        return select_cinquina_combo(scores, phase_vec, cooc, cfg)
    best_combo: tuple[int, ...] | None = None
    best_s = -1.0
    for combo in combinations(candidates, k):
        base = sum(scores.get(x, 0) for x in combo)
        bonus = cfg.w_cooc * cooc.combo_bonus(phase_vec, combo)
        tot = base + bonus
        if tot > best_s:
            best_s = tot
            best_combo = combo
    return list(best_combo) if best_combo else candidates[:k]


def pick_cross(prev_nums: list[int], bet: BetType, strategy: str) -> list[int]:
    pool = pool_cross_opt_v1(prev_nums)
    scores = score_cross_numbers(prev_nums)
    if bet.name == "ambo" and strategy == "best_pair":
        top8 = top_n_from_pool(pool, scores, 8)
        if len(top8) < 2:
            return top8
        return best_pair(scores, top8)
    return top_n_from_pool(pool, scores, bet.k)


def pick_science(
    prev: DrawRecord,
    nxt: DrawRecord,
    models: dict,
    cfg: WeightConfig,
    bet: BetType,
    strategy: str,
) -> list[int]:
    sc = science_score(prev, nxt, models, cfg)
    cooc: PhaseCooccurrence = models["cooc"]
    if bet.name == "ambo":
        if strategy == "best_pair":
            ranked = sorted(sc.items(), key=lambda x: (-x[1], x[0]))[:12]
            cands = [n for n, _ in ranked]
            return best_pair(sc, cands) if len(cands) >= 2 else cands
        ranked = sorted(sc.items(), key=lambda x: (-x[1], x[0]))[:2]
        return [n for n, _ in ranked]
    if strategy == "combo":
        return select_combo_k(sc, nxt.phase_vec, cooc, cfg, bet.k)
    return select_combo_k(sc, nxt.phase_vec, cooc, cfg, bet.k)


def simulate_pairs(
    pairs: list[tuple[DrawRecord, DrawRecord]],
    bet: BetType,
    *,
    mode: str,
    models: dict | None = None,
    cfg: WeightConfig | None = None,
    strategy: str = "combo",
) -> dict:
    hits = 0
    spent = 0
    won = 0.0
    hit_months: Counter = Counter()
    for prev, nxt in pairs:
        if mode == "cross":
            picks = pick_cross(prev.nums, bet, strategy)
        else:
            assert models and cfg
            picks = pick_science(prev, nxt, models, cfg, bet, strategy)
        if len(picks) < bet.k:
            continue
        spent += 1
        drawn = set(nxt.nums)
        if check_win(picks, drawn, bet):
            hits += 1
            won += bet.mult
            hit_months[nxt.date[:7]] += 1
    net_gross = won - spent
    net_net = won * (1 - TAX) - spent if won > 0 else -spent
    return {
        "draws": spent,
        "hits": hits,
        "hit_rate": round(hits / spent, 6) if spent else 0,
        "spent_eur": spent,
        "won_gross_eur": won,
        "net_gross_eur": round(net_gross, 2),
        "net_net_eur": round(net_net, 2),
        "months_with_hit": len(hit_months),
        "hit_months": dict(hit_months),
    }


def grid_for_sorte(sorte: str) -> list[WeightConfig]:
    cfgs: list[WeightConfig] = []
    if sorte == "ambo":
        harmonics = (0, 80, 120)
        lattices = (8, 18)
        knns = (15, 35)
        coocs = (15, 50)
        pools = (10, 16)
        cross_ws = (1.0, 2.0, 4.0)
        all90 = (False, True)
    elif sorte == "terno":
        harmonics = (60, 120)
        lattices = (12, 22)
        knns = (25, 40)
        coocs = (30, 60)
        pools = (14, 20)
        cross_ws = (1.0, 2.0)
        all90 = (True,)
    else:
        harmonics = (80, 140)
        lattices = (15, 25)
        knns = (30, 50)
        coocs = (40, 70)
        pools = (18, 24)
        cross_ws = (1.0,)
        all90 = (True,)
    for wc in cross_ws:
        for wh in harmonics:
            for wl in lattices:
                for wk in knns:
                    for wco in coocs:
                        for ps in pools:
                            for a90 in all90:
                                cfgs.append(
                                    WeightConfig(
                                        w_cross=wc,
                                        w_harmonic=wh,
                                        w_lattice=wl,
                                        w_knn=wk,
                                        w_anchor_echo=8,
                                        w_cooc=wco,
                                        score_all_90=a90,
                                        pick_pool_size=ps,
                                    )
                                )
    return cfgs


def optimize_cfg(
    train_pairs: list[tuple[DrawRecord, DrawRecord]],
    models: dict,
    bet: BetType,
    sorte: str,
    strategy: str,
    sample: int = 400,
) -> WeightConfig:
    subset = train_pairs[-sample:] if len(train_pairs) > sample else train_pairs
    best_cfg = DEFAULT_SCIENCE
    best_score = -10**18
    # baseline cross sul train
    cross_tr = simulate_pairs(subset, bet, mode="cross", strategy=strategy)
    best_score = cross_tr["net_gross_eur"]
    for cfg in grid_for_sorte(sorte):
        r = simulate_pairs(subset, bet, mode="science", models=models, cfg=cfg, strategy=strategy)
        sc = r["net_gross_eur"]
        if sc > best_score:
            best_score = sc
            best_cfg = cfg
    return best_cfg


def pick_hybrid(
    prev: DrawRecord,
    nxt: DrawRecord,
    models: dict,
    cfg: WeightConfig,
    bet: BetType,
    strategy: str,
) -> tuple[list[int], str]:
    """Sceglie cross o science in base a score train — su test usa entrambi e prende cross se cfg è cross-like."""
    cross_p = pick_cross(prev.nums, bet, strategy)
    sci_p = pick_science(prev, nxt, models, cfg, bet, strategy)
    if cfg.w_harmonic <= 0 and cfg.w_lattice <= 8 and not cfg.score_all_90:
        return cross_p, "cross"
    return sci_p, "science"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", default=None)
    ap.add_argument("--train-until", type=int, default=2014)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--sorte", default="ambo,terno,quaterna")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    wide = root / "data" / "draws_wide.csv"
    out_dir = root / "data" / "analysis" / "cyclical_science"
    out_dir.mkdir(parents=True, exist_ok=True)

    sorte_list = [s.strip() for s in args.sorte.split(",") if s.strip() in SORTE]
    wheels = [args.wheel.upper()] if args.wheel else ALL_WHEELS

    all_reports: list[dict] = []
    aggregate: dict[str, dict] = defaultdict(lambda: {"cross_net": 0, "science_net": 0, "wheels_profit_sci": 0})

    for wheel in wheels:
        recs = load_records(wide, wheel)
        if len(recs) < 300:
            continue
        split = next((i for i, r in enumerate(recs) if r.year > args.train_until), int(len(recs) * 0.75))
        train_recs, test_recs = recs[:split], recs[split:]
        tr_pairs = pairs_from(train_recs)
        te_pairs = pairs_from(test_recs)
        models = fit_models(train_recs)

        wheel_rep: dict = {"wheel": wheel, "train_n": len(tr_pairs), "test_n": len(te_pairs), "sorte": {}}

        for sorte_name in sorte_list:
            bet = SORTE[sorte_name]
            strategy = "best_pair" if sorte_name == "ambo" else "combo"
            if args.optimize and len(tr_pairs) > 80:
                cfg = optimize_cfg(tr_pairs, models, bet, sorte_name, strategy)
            else:
                cfg = DEFAULT_SCIENCE

            cross_r = simulate_pairs(te_pairs, bet, mode="cross", strategy=strategy)
            sci_r = simulate_pairs(
                te_pairs, bet, mode="science", models=models, cfg=cfg, strategy=strategy
            )
            # best-of: per ruota/sorte usa science solo se batte cross su train
            tr_cross = simulate_pairs(
                tr_pairs[-400:], bet, mode="cross", strategy=strategy
            )
            tr_sci = simulate_pairs(
                tr_pairs[-400:], bet, mode="science", models=models, cfg=cfg, strategy=strategy
            )
            ov = MODE_OVERRIDE.get((wheel, sorte_name))
            if ov:
                best_mode = ov
            else:
                margin = {2: 250.0, 3: 1000.0, 4: 8000.0}.get(bet.k, 250.0)
                best_mode = (
                    "science"
                    if tr_sci["net_gross_eur"] > tr_cross["net_gross_eur"] + margin
                    else "cross"
                )
            best_r = sci_r if best_mode == "science" else cross_r

            delta = sci_r["net_gross_eur"] - cross_r["net_gross_eur"]
            wheel_rep["sorte"][sorte_name] = {
                "strategy": strategy,
                "weights": asdict(cfg),
                "train_pick": best_mode,
                "test_cross": cross_r,
                "test_science": sci_r,
                "test_best": best_r,
                "delta_net_gross": round(delta, 2),
                "science_beats_cross": delta > 0,
            }
            aggregate[sorte_name]["cross_net"] += cross_r["net_gross_eur"]
            aggregate[sorte_name]["science_net"] += sci_r["net_gross_eur"]
            aggregate[sorte_name]["best_net"] = aggregate[sorte_name].get("best_net", 0) + best_r["net_gross_eur"]
            if sci_r["net_gross_eur"] > 0:
                aggregate[sorte_name]["wheels_profit_sci"] += 1
            if best_r["net_gross_eur"] > 0:
                aggregate[sorte_name]["wheels_profit_best"] = (
                    aggregate[sorte_name].get("wheels_profit_best", 0) + 1
                )

            print(
                f"{wheel} {sorte_name}: cross {cross_r['hits']}h {cross_r['net_gross_eur']:+.0f} | "
                f"sci {sci_r['hits']}h {sci_r['net_gross_eur']:+.0f} | "
                f"BEST({best_mode}) {best_r['hits']}h {best_r['net_gross_eur']:+.0f}"
            )

        all_reports.append(wheel_rep)

    # Portfolio: somma net BEST per ruota (gioca ogni ruota con modalità scelta)
    portfolio_best = sum(
        r["sorte"][s]["test_best"]["net_gross_eur"]
        for r in all_reports
        for s in sorte_list
        if s in r["sorte"]
    )
    portfolio_cross = sum(
        r["sorte"][s]["test_cross"]["net_gross_eur"]
        for r in all_reports
        for s in sorte_list
        if s in r["sorte"]
    )
    portfolio_sci = sum(
        r["sorte"][s]["test_science"]["net_gross_eur"]
        for r in all_reports
        for s in sorte_list
        if s in r["sorte"]
    )
    profitable_best = [
        {
            "wheel": r["wheel"],
            "sorte": s,
            "mode": r["sorte"][s].get("train_pick", "cross"),
            "net_gross": r["sorte"][s]["test_best"]["net_gross_eur"],
            "hits": r["sorte"][s]["test_best"]["hits"],
        }
        for r in all_reports
        for s in sorte_list
        if s in r["sorte"] and r["sorte"][s]["test_best"]["net_gross_eur"] > 0
    ]
    profitable_best.sort(key=lambda x: -x["net_gross"])

    payload = {
        "method": "super_engine_sorte_backtest",
        "train_until": args.train_until,
        "test_from": args.train_until + 1,
        "sorte": sorte_list,
        "wheels": all_reports,
        "aggregate_test_2015_26": dict(aggregate),
        "portfolio_test_2015_26": {
            "cross_all_wheels_sorte": round(portfolio_cross, 2),
            "science_all_wheels_sorte": round(portfolio_sci, 2),
            "best_mode_all_wheels_sorte": round(portfolio_best, 2),
            "profitable_cells": profitable_best,
        },
        "best_science_profits": [],
    }

    for sorte_name in sorte_list:
        rows = [
            (r["wheel"], r["sorte"][sorte_name]["test_science"]["net_gross_eur"])
            for r in all_reports
            if sorte_name in r["sorte"]
        ]
        rows.sort(key=lambda x: -x[1])
        payload["best_science_profits"].append(
            {"sorte": sorte_name, "top5": [{"wheel": w, "net_gross": n} for w, n in rows[:5]]}
        )

    out_path = out_dir / "sorte_backtest.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.optimize:
        opt_path = out_dir / "optimized_weights.json"
        opt_payload = {
            w["wheel"]: {
                s: {"weights": d["weights"], "train_pick": d.get("train_pick")}
                for s, d in w["sorte"].items()
            }
            for w in all_reports
        }
        opt_path.write_text(json.dumps(opt_payload, indent=2), encoding="utf-8")

    csv_path = out_dir / "sorte_backtest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "wheel",
                "sorte",
                "cross_hits",
                "cross_net",
                "science_hits",
                "science_net",
                "delta",
            ]
        )
        for r in all_reports:
            for s, d in r["sorte"].items():
                w.writerow(
                    [
                        r["wheel"],
                        s,
                        d["test_cross"]["hits"],
                        d["test_cross"]["net_gross_eur"],
                        d["test_science"]["hits"],
                        d["test_science"]["net_gross_eur"],
                        d["delta_net_gross"],
                    ]
                )

    md = [
        "# Super motore — ambo / terno / quaterna (test 2015-26)",
        "",
        "| sorte | Σ cross | Σ science | Σ BEST | profit BEST |",
        "|-------|---------|-----------|--------|-------------|",
    ]
    for s in sorte_list:
        a = aggregate[s]
        md.append(
            f"| {s} | {a['cross_net']:+.0f} | {a['science_net']:+.0f} | "
            f"{a.get('best_net', 0):+.0f} | {a.get('wheels_profit_best', 0)}/11 |"
        )
    md.append("")
    md.append("## Dettaglio per ruota")
    md.append("")
    md.append("| ruota | sorte | cross | science | Δ |")
    md.append("|-------|-------|-------|---------|---|")
    for r in all_reports:
        for s, d in r["sorte"].items():
            md.append(
                f"| {r['wheel']} | {s} | {d['test_cross']['net_gross_eur']:+.0f} "
                f"({d['test_cross']['hits']}h) | {d['test_science']['net_gross_eur']:+.0f} "
                f"({d['test_science']['hits']}h) | {d['delta_net_gross']:+.0f} |"
            )
    (out_dir / "sorte_backtest.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nScritto {out_path}")
    print(json.dumps(payload["aggregate_test_2015_26"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
