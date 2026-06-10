#!/usr/bin/env python3
"""API predizione super motore da ultima cinquina (per web / CLI)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from cyclical_science_engine import (
    ALL_WHEELS,
    CouponTracker,
    DrawRecord,
    WeightConfig,
    build_phase_vector,
    fit_models,
    load_records,
    science_score,
    select_cinquina_combo,
    score_confidence,
    signed_diffs,
)
from super_engine_sorte_backtest import (
    MODE_OVERRIDE,
    SORTE,
    best_pair,
    pick_cross,
    pick_science,
    select_combo_k,
)
from simulate_ambi_2025 import score_cross_numbers, top_n_from_pool
from vertibile_rule_test import pool_cross_opt_v1

ROOT = Path(__file__).resolve().parent.parent
WIDE = ROOT / "data" / "draws_wide.csv"
OPT_PATH = ROOT / "data" / "analysis" / "cyclical_science" / "optimized_weights.json"

PAYOUT = {
    "ambo": 250,
    "terno": 4500,
    "quaterna": 120_000,
    "cinquina": 6_000_000,
}


def _load_opt_weights() -> dict:
    if OPT_PATH.is_file():
        return json.loads(OPT_PATH.read_text(encoding="utf-8"))
    return {}


def _next_draw_date(from_date: date | None = None) -> date:
    """Prossimo giorno di estrazione Lotto (mar/gio/sab)."""
    d = from_date or date.today()
    for _ in range(8):
        if d.weekday() in (1, 3, 5):  # Tue Thu Sat
            return d
        d += timedelta(days=1)
    return d


def _estimate_draw_index(recs: list[DrawRecord], target: date) -> int:
    if not recs:
        return 1
    last = recs[-1]
    if last.year == target.year:
        return last.draw_index + 1
    return 1


def _replay_tracker(recs: list[DrawRecord]) -> CouponTracker:
    tracker = CouponTracker()
    for r in recs:
        tracker.observe(r.nums)
    return tracker


def _record_from_quintina(
    tracker: CouponTracker,
    gidx: int,
    date_str: str,
    draw_index: int,
    quintina: list[int],
) -> DrawRecord:
    dt = date.fromisoformat(date_str)
    cid, cpos = tracker.observe(quintina)
    anchor = quintina[0]
    v = signed_diffs(quintina)
    flow_sum = ((sum(v) - 1) % 90) + 1 if v else anchor
    meta = {
        "weekday": dt.weekday(),
        "month": dt.month,
        "draw_index": draw_index,
        "day_of_year": dt.timetuple().tm_yday,
        "anchor": anchor,
        "anchor_decile": min(9, (anchor - 1) * 10 // 90),
        "cycle_pos": cpos,
        "cycle_id": cid,
        "flow_sum": flow_sum,
    }
    pv = build_phase_vector(meta)
    return DrawRecord(
        date=date_str,
        year=dt.year,
        month=dt.month,
        day=dt.day,
        weekday=meta["weekday"],
        day_of_year=meta["day_of_year"],
        draw_index=draw_index,
        global_idx=gidx,
        cycle_pos=cpos,
        cycle_id=cid,
        nums=quintina,
        anchor=anchor,
        flow_sum=flow_sum,
        phase_vec=pv,
    )


def _next_phase_record(
    tracker: CouponTracker,
    gidx: int,
    date_str: str,
    draw_index: int,
    anchor_proxy: int,
    flow_proxy: int,
) -> DrawRecord:
    """Fase calendario della prossima estrazione (numeri ancora sconosciuti)."""
    dt = date.fromisoformat(date_str)
    cid, cpos = tracker.cycle_id, tracker.pos_in_cycle
    meta = {
        "weekday": dt.weekday(),
        "month": dt.month,
        "draw_index": draw_index,
        "day_of_year": dt.timetuple().tm_yday,
        "anchor": anchor_proxy,
        "anchor_decile": min(9, (anchor_proxy - 1) * 10 // 90),
        "cycle_pos": cpos,
        "cycle_id": cid,
        "flow_sum": flow_proxy,
    }
    pv = build_phase_vector(meta)
    return DrawRecord(
        date=date_str,
        year=dt.year,
        month=dt.month,
        day=dt.day,
        weekday=meta["weekday"],
        day_of_year=meta["day_of_year"],
        draw_index=draw_index,
        global_idx=gidx,
        cycle_pos=cpos,
        cycle_id=cid,
        nums=[anchor_proxy],
        anchor=anchor_proxy,
        flow_sum=flow_proxy,
        phase_vec=pv,
    )


@lru_cache(maxsize=11)
def _wheel_models(wheel: str, train_years: int) -> dict:
    recs = load_records(WIDE, wheel)
    cut = max(0, len(recs) - train_years * 50)
    return fit_models(recs[cut:])


def predict_from_quintina(
    wheel: str,
    quintina: list[int],
    *,
    prev_date: str | None = None,
    next_date: str | None = None,
    draw_index: int | None = None,
    train_years: int = 30,
) -> dict:
    """
    Da ultima cinquina (ordine estrazione pos1..pos5) → suggerimenti sorti.

    Cross: funziona SOLO con la quintina (regole complemento/vertibile/diff).
    Science: aggiunge armoniche, fase calendario, k-NN su 156 anni di storico.
    """
    wheel = wheel.upper()
    if wheel not in ALL_WHEELS:
        raise ValueError(f"Ruota non valida: {wheel}")
    if len(quintina) != 5:
        raise ValueError("Servono esattamente 5 numeri (cinquina ordinata)")
    if not all(1 <= n <= 90 for n in quintina):
        raise ValueError("Numeri devono essere tra 1 e 90")

    recs = load_records(WIDE, wheel)
    opt = _load_opt_weights()
    wopt = opt.get(wheel, {})

    # Date
    if prev_date is None:
        prev_date = recs[-1].date if recs else date.today().isoformat()
    nd = date.fromisoformat(next_date) if next_date else _next_draw_date(
        date.fromisoformat(prev_date) + timedelta(days=1)
    )
    next_date_str = nd.isoformat()
    di_prev = _estimate_draw_index(recs, date.fromisoformat(prev_date))
    di_next = draw_index if draw_index is not None else di_prev + 1

    tracker = _replay_tracker(recs)
    gidx = len(recs)
    prev_rec = _record_from_quintina(tracker, gidx, prev_date, di_prev, quintina)
    next_rec = _next_phase_record(
        tracker, gidx + 1, next_date_str, di_next, prev_rec.anchor, prev_rec.flow_sum
    )

    models = _wheel_models(wheel, train_years)
    pool = sorted(pool_cross_opt_v1(quintina))
    cross_scores = score_cross_numbers(quintina)

    sorte_out: dict = {}
    for sorte_name, bet in SORTE.items():
        mode = MODE_OVERRIDE.get((wheel, sorte_name))
        if not mode:
            mode = wopt.get(sorte_name, {}).get("mode", "science")
        ow = wopt.get(sorte_name, {}).get("weights")
        cfg = WeightConfig(**ow) if ow else WeightConfig()
        strat = "best_pair" if sorte_name == "ambo" else "combo"

        cross_picks = pick_cross(quintina, bet, strat)
        if mode == "cross":
            picks = cross_picks
        else:
            picks = pick_science(prev_rec, next_rec, models, cfg, bet, strat)

        sorte_out[sorte_name] = {
            "mode": mode,
            "picks": picks,
            "cross_picks": cross_picks,
            "payout_eur": PAYOUT[sorte_name],
            "stake": "1€ intero sulla combinazione secca",
        }

    ambo_cfg = WeightConfig(**wopt.get("ambo", {}).get("weights", {})) if wopt.get("ambo", {}).get("weights") else WeightConfig()
    sc = science_score(prev_rec, next_rec, models, ambo_cfg)
    cinquina = select_cinquina_combo(
        sc, next_rec.phase_vec, models["cooc"],
        WeightConfig(pick_pool_size=22, w_cooc=60),
    )

    top_pool = top_n_from_pool(set(pool), cross_scores, 12)
    return {
        "wheel": wheel,
        "input_quintina": quintina,
        "anchor_pos1": quintina[0],
        "prev_date": prev_date,
        "next_draw_date": next_date_str,
        "draw_index_next": di_next,
        "pool_cross_size": len(pool),
        "pool_top12": top_pool,
        "sorte": sorte_out,
        "cinquina": cinquina,
        "confidence": round(score_confidence(sc, cinquina), 4),
        "how_it_works": {
            "step1": "Pool cross_opt_v1 dalla quintina: complementi 90-n, vertibili, diff, incroci",
            "step2": "Score cross: pesa incroci comp+diff e vicini gemello ±11",
            "step3_science": "Somma armoniche temporali + fase calendario + k-NN storico + eco seme pos1",
            "step4": "Ambo: miglior coppia nel top-8; terno/quaterna: combo ottimale nel top-pool",
            "note": "NAZIONALE ambo usa sempre cross (validato). Science richiede storico CSV.",
        },
    }
