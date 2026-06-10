#!/usr/bin/env python3
"""Foglio gioco: ambo/terno/quaterna/cinquina super motore per ruota."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cyclical_science_engine import (
    ALL_WHEELS,
    WeightConfig,
    fit_models,
    load_records,
    science_score,
    select_cinquina_combo,
    score_confidence,
)
from super_engine_sorte_backtest import (
    MODE_OVERRIDE,
    SORTE,
    pick_cross,
    pick_science,
    select_combo_k,
)
from vertibile_rule_test import pool_cross_opt_v1

OPT_PATH_NAME = "optimized_weights.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", default=None)
    ap.add_argument("--train-years", type=int, default=30, help="Anni train rolling")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    wide = root / "data" / "draws_wide.csv"
    opt_path = root / "data" / "analysis" / "cyclical_science" / OPT_PATH_NAME
    opt_weights = {}
    if opt_path.is_file():
        opt_weights = json.loads(opt_path.read_text(encoding="utf-8"))

    wheels = [args.wheel.upper()] if args.wheel else ALL_WHEELS
    out: dict = {"wheels": {}}

    for w in wheels:
        recs = load_records(wide, w)
        if len(recs) < 100:
            continue
        cut = max(0, len(recs) - args.train_years * 50)
        train, last = recs[cut:-1], recs[-1]
        models = fit_models(train)
        prev = recs[-2]
        wopt = opt_weights.get(w, {})
        picks_by_sorte: dict = {}
        for sorte_name, bet in SORTE.items():
            mode = MODE_OVERRIDE.get((w, sorte_name), "science")
            ow = wopt.get(sorte_name, {}).get("weights")
            cfg = WeightConfig(**ow) if ow else WeightConfig()
            strat = "best_pair" if sorte_name == "ambo" else "combo"
            if mode == "cross":
                picks = pick_cross(prev.nums, bet, strat)
            else:
                picks = pick_science(prev, last, models, cfg, bet, strat)
            picks_by_sorte[sorte_name] = {"mode": mode, "picks": picks}
        sc = science_score(
            prev, last, models,
            WeightConfig(**wopt.get("ambo", {}).get("weights", {}))
            if wopt.get("ambo", {}).get("weights")
            else WeightConfig(),
        )
        cinq = select_cinquina_combo(
            sc, last.phase_vec, models["cooc"],
            WeightConfig(pick_pool_size=22, w_cooc=60),
        )
        conf = round(score_confidence(sc, cinq), 4)
        out["wheels"][w] = {
            "last_date": last.date,
            "prev_quintina": prev.nums,
            "anchor_pos1": prev.anchor,
            "phase_key": last.phase_vec,
            "sorte": picks_by_sorte,
            "cinquina_picks": cinq,
            "confidence": conf,
        }
        ambo = picks_by_sorte["ambo"]["picks"]
        print(f"{w} | {last.date} | seme {prev.anchor} | ambo {ambo} | cinquina {cinq}")

    out_path = root / "data" / "analysis" / "cyclical_science" / "play_sheet.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nScritto {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
