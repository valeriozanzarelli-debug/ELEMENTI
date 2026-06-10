#!/usr/bin/env python3
"""Foglio gioco: cinquina science per ruota (ultima estrazione → prossima)."""
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

DEFAULT_CFG = WeightConfig(
    w_harmonic=120,
    w_lattice=20,
    w_knn=40,
    w_anchor_echo=14,
    w_cooc=60,
    score_all_90=True,
    pick_pool_size=22,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", default=None)
    ap.add_argument("--train-years", type=int, default=30, help="Anni train rolling")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    wide = root / "data" / "draws_wide.csv"
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
        sc = science_score(prev, last, models, DEFAULT_CFG)
        picks = select_cinquina_combo(sc, last.phase_vec, models["cooc"], DEFAULT_CFG)
        conf = round(score_confidence(sc, picks), 4)
        out["wheels"][w] = {
            "last_date": last.date,
            "prev_quintina": prev.nums,
            "anchor_pos1": prev.anchor,
            "phase_key": last.phase_vec,
            "cinquina_picks": picks,
            "confidence": conf,
            "play_cinquina": conf > 0.5,
        }
        print(
            f"{w} | {last.date} | seme {prev.anchor} | "
            f"cinquina {picks} | conf {conf}"
        )

    out_path = root / "data" / "analysis" / "cyclical_science" / "play_sheet.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nScritto {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
