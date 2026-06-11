#!/usr/bin/env python3
"""API predizione super motore da ultima cinquina (per web / CLI)."""

from __future__ import annotations

import json
from pathlib import Path

from cyclical_science_engine import ALL_WHEELS
from super_engine_sorte_backtest import SORTE, pick_cross
from simulate_ambi_2025 import score_cross_numbers, top_n_from_pool
from vertibile_rule_test import pool_cross_opt_v1

ROOT = Path(__file__).resolve().parent.parent
OPT_PATH = ROOT / "data" / "analysis" / "cyclical_science" / "optimized_weights.json"

PAYOUT = {
    "ambo": 250,
    "terno": 4500,
    "quaterna": 120_000,
    "cinquina": 6_000_000,
}


def predict_from_quintina(wheel: str, quintina: list[int]) -> dict:
    """
    Da ultima cinquina (ordine pos1..pos5) → ambo, terno, quaterna, cinquina.

  Solo la quintina conta: complementi 90-n, vertibili, diff, incroci.
    Data e n° concorso NON servono. La ruota indica su quale giocare
    (devi inserire la quintina uscita su quella ruota).
    """
    wheel = wheel.upper()
    if wheel not in ALL_WHEELS:
        raise ValueError(f"Ruota non valida: {wheel}")
    if len(quintina) != 5:
        raise ValueError("Servono esattamente 5 numeri (cinquina ordinata)")
    if not all(1 <= n <= 90 for n in quintina):
        raise ValueError("Numeri devono essere tra 1 e 90")
    if len(set(quintina)) != 5:
        raise ValueError("I 5 numeri devono essere tutti diversi")

    pool = pool_cross_opt_v1(quintina)
    cross_scores = score_cross_numbers(quintina)

    sorte_out: dict = {}
    for sorte_name, bet in SORTE.items():
        strat = "best_pair" if sorte_name == "ambo" else "combo"
        picks = pick_cross(quintina, bet, strat)
        sorte_out[sorte_name] = {
            "picks": picks,
            "payout_eur": PAYOUT[sorte_name],
            "stake": "1€ intero sulla combinazione secca",
        }

    top_pool = top_n_from_pool(set(pool), cross_scores, 12)
    return {
        "wheel": wheel,
        "input_quintina": quintina,
        "anchor_pos1": quintina[0],
        "pool_cross_size": len(pool),
        "pool_top12": top_pool,
        "sorte": sorte_out,
        "method": "cross_opt_v1",
        "note": (
            "Calcolo basato solo sulla quintina inserita. "
            "Se cambi ruota ma lasci la stessa quintina, i numeri restano uguali: "
            "usa la cinquina dell'ultima estrazione su quella ruota."
        ),
    }
