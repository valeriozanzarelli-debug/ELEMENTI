#!/usr/bin/env python3
"""
Verifica live: ultime 5 estrazioni Italia (Lotto) e Spagna (La Primitiva).
Dati estrazioni giugno 2026 verificati da Ilprincipa.it / LaVanguardia.com (giu 2026).

Italia: motore super (cross_opt_v1 + cyclical science) con pesi ottimizzati.
Spagna: esperimento cross adattato 1-49 (NON validato storicamente).
"""

from __future__ import annotations

import json
import sys
from datetime import date
from itertools import combinations
from pathlib import Path

from cyclical_science_engine import (
    ALL_WHEELS,
    CouponTracker,
    DrawRecord,
    WeightConfig,
    build_phase_vector,
    fit_models,
    load_records,
    signed_diffs,
)
from simulate_ambi_2025 import score_cross_numbers, top_n_from_pool
from simulate_sorti_proper import TAX, BetType, check_win
from super_engine_sorte_backtest import (
    MODE_OVERRIDE,
    SORTE,
    pick_cross,
    pick_science,
)
from vertibile_rule_test import pool_cross_opt_v1

# Fonti web (giu 2026): Ilprincipa.it estrazioni #85-#92, LaVanguardia Primitiva
BRIDGE_DRAWS = [
  {
    "date": "2026-05-28",
    "draw_index": 85,
    "wheels": {
      "BARI": [18, 25, 46, 74, 40],
      "CAGLIARI": [21, 3, 75, 5, 11],
      "FIRENZE": [2, 22, 90, 1, 56],
      "GENOVA": [73, 88, 37, 34, 28],
      "MILANO": [21, 75, 56, 10, 45],
      "NAPOLI": [16, 89, 58, 53, 68],
      "PALERMO": [7, 6, 89, 20, 66],
      "ROMA": [49, 54, 8, 37, 83],
      "TORINO": [80, 19, 86, 66, 53],
      "VENEZIA": [59, 63, 23, 30, 53],
      "NAZIONALE": [77, 89, 38, 27, 22],
    },
  },
  {
    "date": "2026-05-29",
    "draw_index": 86,
    "wheels": {
      "BARI": [87, 60, 29, 36, 7],
      "CAGLIARI": [7, 68, 27, 53, 6],
      "FIRENZE": [7, 33, 20, 30, 12],
      "GENOVA": [34, 30, 26, 13, 66],
      "MILANO": [60, 84, 81, 89, 10],
      "NAPOLI": [56, 33, 52, 15, 72],
      "PALERMO": [3, 36, 82, 60, 23],
      "ROMA": [28, 10, 6, 5, 81],
      "TORINO": [12, 76, 66, 61, 24],
      "VENEZIA": [34, 90, 7, 5, 77],
      "NAZIONALE": [6, 42, 2, 66, 87],
    },
  },
]

# Estrazione ponte (prev per la prima delle 5 live)
DRAW_20260530 = {
  "date": "2026-05-30",
  "draw_index": 87,
  "source": "Ilprincipa.it / TPI",
  "wheels": {
    "BARI": [87, 61, 14, 25, 38],
    "CAGLIARI": [72, 1, 87, 13, 26],
    "FIRENZE": [34, 60, 5, 42, 54],
    "GENOVA": [62, 5, 18, 69, 73],
    "MILANO": [14, 54, 29, 81, 83],
    "NAPOLI": [71, 76, 89, 19, 25],
    "PALERMO": [33, 15, 48, 26, 65],
    "ROMA": [67, 47, 2, 86, 42],
    "TORINO": [78, 87, 66, 10, 52],
    "VENEZIA": [84, 23, 40, 15, 24],
    "NAZIONALE": [77, 72, 73, 56, 83],
  },
}

LIVE_DRAWS_ITALY = [
  {
    "date": "2026-06-03",
    "draw_index": 88,
    "source": "Ilprincipa.it",
    "wheels": {
      "BARI": [6, 49, 53, 65, 25],
      "CAGLIARI": [23, 71, 63, 50, 27],
      "FIRENZE": [79, 7, 61, 8, 65],
      "GENOVA": [5, 48, 79, 23, 16],
      "MILANO": [15, 40, 78, 47, 58],
      "NAPOLI": [37, 81, 7, 34, 78],
      "PALERMO": [36, 65, 39, 69, 5],
      "ROMA": [4, 23, 53, 82, 5],
      "TORINO": [24, 40, 52, 35, 81],
      "VENEZIA": [53, 85, 38, 74, 67],
      "NAZIONALE": [57, 31, 74, 75, 30],
    },
  },
  {
    "date": "2026-06-04",
    "draw_index": 89,
    "source": "Ilprincipa.it",
    "wheels": {
      "BARI": [78, 89, 73, 50, 28],
      "CAGLIARI": [2, 59, 30, 60, 9],
      "FIRENZE": [62, 56, 28, 72, 23],
      "GENOVA": [20, 14, 19, 32, 24],
      "MILANO": [25, 30, 78, 72, 40],
      "NAPOLI": [33, 69, 25, 17, 26],
      "PALERMO": [63, 24, 82, 1, 8],
      "ROMA": [21, 17, 23, 71, 31],
      "TORINO": [2, 76, 71, 58, 72],
      "VENEZIA": [58, 48, 51, 78, 67],
      "NAZIONALE": [89, 53, 61, 62, 15],
    },
  },
  {
    "date": "2026-06-05",
    "draw_index": 90,
    "source": "Ilprincipa.it",
    "wheels": {
      "BARI": [11, 36, 33, 25, 57],
      "CAGLIARI": [57, 4, 70, 22, 73],
      "FIRENZE": [39, 71, 83, 31, 72],
      "GENOVA": [79, 89, 18, 80, 5],
      "MILANO": [15, 68, 19, 49, 87],
      "NAPOLI": [10, 56, 66, 58, 7],
      "PALERMO": [50, 27, 19, 6, 75],
      "ROMA": [60, 24, 39, 79, 62],
      "TORINO": [75, 47, 41, 84, 45],
      "VENEZIA": [56, 8, 84, 38, 9],
      "NAZIONALE": [61, 62, 68, 30, 75],
    },
  },
  {
    "date": "2026-06-06",
    "draw_index": 91,
    "source": "Ilprincipa.it / TPI / Today.it",
    "wheels": {
      "BARI": [79, 44, 34, 76, 86],
      "CAGLIARI": [16, 76, 87, 84, 37],
      "FIRENZE": [70, 35, 43, 72, 4],
      "GENOVA": [38, 59, 6, 79, 55],
      "MILANO": [40, 68, 61, 88, 83],
      "NAPOLI": [37, 61, 90, 22, 14],
      "PALERMO": [31, 45, 90, 44, 26],
      "ROMA": [72, 4, 6, 41, 23],
      "TORINO": [65, 62, 34, 6, 86],
      "VENEZIA": [34, 64, 59, 49, 71],
      "NAZIONALE": [7, 25, 36, 38, 8],
    },
  },
  {
    "date": "2026-06-09",
    "draw_index": 92,
    "source": "Ilprincipa.it",
    "wheels": {
      "BARI": [31, 81, 28, 45, 85],
      "CAGLIARI": [35, 85, 70, 25, 88],
      "FIRENZE": [27, 68, 41, 35, 69],
      "GENOVA": [52, 71, 88, 38, 27],
      "MILANO": [12, 82, 83, 25, 80],
      "NAPOLI": [37, 10, 17, 48, 59],
      "PALERMO": [74, 55, 30, 16, 29],
      "ROMA": [3, 90, 32, 37, 43],
      "TORINO": [35, 43, 64, 8, 67],
      "VENEZIA": [33, 66, 43, 85, 44],
      "NAZIONALE": [15, 32, 28, 67, 56],
    },
  },
]

# Ruote con ambo BEST profittevole su test 2015-26
PROFITABLE_WHEELS = ["PALERMO", "MILANO", "NAZIONALE", "ROMA", "VENEZIA"]

PRIMITIVA_DRAWS = [
  {
    "date": "2026-05-30",
    "nums": [2, 12, 32, 34, 48, 49],
    "complementario": 41,
    "reintegro": 3,
    "prizes": {3: 8.0, 4: 59.17, 5: 2125.77, 6: 1_237_007.11},
    "source": "LaVanguardia.com",
  },
  {
    "date": "2026-06-01",
    "nums": [1, 4, 11, 12, 23, 39],
    "complementario": 27,
    "reintegro": 9,
    "prizes": {3: 8.0, 4: 44.02, 5: 1800.79, 6: 0.0},
    "source": "LaVanguardia.com",
  },
  {
    "date": "2026-06-04",
    "nums": [16, 25, 32, 35, 38, 40],
    "complementario": 18,
    "reintegro": 9,
    "prizes": {3: 8.0, 4: 64.71, 5: 2766.01, 6: 1_302_539.42},
    "source": "LaVanguardia.com",
  },
  {
    "date": "2026-06-06",
    "nums": [1, 6, 16, 17, 37, 43],
    "complementario": 5,
    "reintegro": 7,
    "prizes": {3: 8.0, 4: 59.96, 5: 2897.48, 6: 618_876.23},
    "source": "LaVanguardia.com",
  },
  {
    "date": "2026-06-08",
    "nums": [2, 11, 13, 21, 33, 48],
    "complementario": 46,
    "reintegro": 5,
    "prizes": {3: 8.0, 4: 40.81, 5: 1357.44, 6: 671_194.34},
    "source": "LaVanguardia.com",
  },
]


def make_draw_record(
    tracker: CouponTracker,
    gidx: int,
    date_str: str,
    draw_index: int,
    nums: list[int],
) -> DrawRecord:
    dt = date.fromisoformat(date_str)
    cid, cpos = tracker.observe(nums)
    anchor = nums[0]
    v = signed_diffs(nums)
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
        nums=nums,
        anchor=anchor,
        flow_sum=flow_sum,
        phase_vec=pv,
    )


def extend_records(
    records: list[DrawRecord],
    additions: list[tuple[str, int, list[int]]],
) -> list[DrawRecord]:
    tracker = CouponTracker()
    for r in records:
        tracker.observe(r.nums)
    out = list(records)
    gidx = len(records)
    for date_str, draw_index, nums in additions:
        out.append(make_draw_record(tracker, gidx, date_str, draw_index, nums))
        gidx += 1
    return out


def build_additions_for_wheel(
    bridge: list[dict],
    pivot: dict,
    live: list[dict],
    wheel: str,
) -> list[tuple[str, int, list[int]]]:
    rows: list[tuple[str, int, list[int]]] = []
    for d in bridge + [pivot] + live:
        rows.append((d["date"], d["draw_index"], d["wheels"][wheel]))
    return rows


# --- Spagna: cross adattato 1-49 (sperimentale) ---

def comp49(n: int) -> int | None:
    if not 1 <= n <= 49:
        return None
    c = 50 - n
    return c if 1 <= c <= 49 else None


def vert49(n: int) -> int | None:
    if n < 10 or n > 49:
        return None
    s = str(n)
    if len(s) != 2:
        return None
    rev = int(s[::-1])
    if rev == n or rev < 1 or rev > 49:
        return None
    return rev


def pool_cross49(nums: list[int]) -> set[int]:
    pool: set[int] = set()
    for n in nums:
        if 1 <= n <= 49:
            pool.add(n)
        c = comp49(n)
        if c:
            pool.add(c)
        v = vert49(n)
        if v:
            pool.add(v)
            c2 = comp49(v)
            if c2:
                pool.add(c2)
        for d in (-1, 1):
            x = n + d
            if 1 <= x <= 49:
                pool.add(x)
    diffs = [abs(nums[i + 1] - nums[i]) for i in range(len(nums) - 1)]
    for d in diffs:
        if 1 <= d <= 49:
            pool.add(d)
            c = comp49(d)
            if c:
                pool.add(c)
    return pool


def score_cross49(nums: list[int]) -> dict[int, float]:
    pool = pool_cross49(nums)
    scores: dict[int, float] = {n: 0.0 for n in pool}
    for i, n in enumerate(nums):
        w = 5 - i
        scores[n] = scores.get(n, 0) + w * 3
        c = comp49(n)
        if c and c in scores:
            scores[c] += w * 2
        v = vert49(n)
        if v and v in scores:
            scores[v] += w * 1.5
    diffs = [abs(nums[i + 1] - nums[i]) for i in range(len(nums) - 1)]
    for d in diffs:
        if d in scores:
            scores[d] += 2.0
    return scores


def pick_top6_cross49(prev: list[int]) -> list[int]:
    pool = pool_cross49(prev)
    scores = score_cross49(prev)
    return top_n_from_pool(list(pool), scores, 6)


def primitiva_prize(hits: int, prize_table: dict[int, float]) -> float:
    if hits >= 6:
        return prize_table.get(6, 0.0)
    return prize_table.get(hits, 0.0)


def verify_italy(
    wide: Path,
    opt_weights: dict,
    sorte_name: str = "ambo",
) -> dict:
    bet = SORTE[sorte_name]
    strat = "best_pair" if sorte_name == "ambo" else "combo"
    per_wheel: dict = {}
    portfolio_all = {"spent": 0, "won_gross": 0.0, "hits": 0, "details": []}
    portfolio_best = {"spent": 0, "won_gross": 0.0, "hits": 0, "details": []}

    for wheel in ALL_WHEELS:
        recs = load_records(wide, wheel)
        additions = build_additions_for_wheel(
            BRIDGE_DRAWS, DRAW_20260530, LIVE_DRAWS_ITALY, wheel
        )
        extended = extend_records(recs, additions)
        models = fit_models(recs)

        mode = MODE_OVERRIDE.get((wheel, sorte_name))
        if not mode:
            mode = opt_weights.get(wheel, {}).get(sorte_name, {}).get("mode", "science")
        ow = opt_weights.get(wheel, {}).get(sorte_name, {}).get("weights")
        cfg = WeightConfig(**ow) if ow else WeightConfig()

        transitions = []
        # ultime 5 coppie: May29->May30 ... Jun5->Jun6
        start = len(extended) - len(LIVE_DRAWS_ITALY) - 1
        for i in range(start, start + len(LIVE_DRAWS_ITALY)):
            prev, nxt = extended[i], extended[i + 1]
            if mode == "cross":
                picks = pick_cross(prev.nums, bet, strat)
            else:
                picks = pick_science(prev, nxt, models, cfg, bet, strat)
            drawn = set(nxt.nums)
            hit = check_win(picks, drawn, bet)
            won = bet.mult if hit else 0.0
            transitions.append({
                "from_date": prev.date,
                "to_date": nxt.date,
                "draw_index": nxt.draw_index,
                "mode": mode,
                "prev_quintina": prev.nums,
                "picks": picks,
                "drawn": nxt.nums,
                "hit": hit,
                "won_gross_eur": won,
            })
            portfolio_all["spent"] += 1
            portfolio_all["won_gross"] += won
            if hit:
                portfolio_all["hits"] += 1
                portfolio_all["details"].append(f"{wheel} {nxt.date}")
            if wheel in PROFITABLE_WHEELS:
                portfolio_best["spent"] += 1
                portfolio_best["won_gross"] += won
                if hit:
                    portfolio_best["hits"] += 1
                    portfolio_best["details"].append(f"{wheel} {nxt.date}")

        per_wheel[wheel] = {
            "mode": mode,
            "transitions": transitions,
            "spent": len(transitions),
            "hits": sum(1 for t in transitions if t["hit"]),
            "won_gross": sum(t["won_gross_eur"] for t in transitions),
        }

    def finalize(p: dict) -> dict:
        net_gross = p["won_gross"] - p["spent"]
        net_net = p["won_gross"] * (1 - TAX) - p["spent"] if p["won_gross"] > 0 else -p["spent"]
        return {
            **p,
            "net_gross_eur": round(net_gross, 2),
            "net_net_eur": round(net_net, 2),
        }

    return {
        "sorte": sorte_name,
        "n_draws": len(LIVE_DRAWS_ITALY),
        "period": "2026-06-03 .. 2026-06-09 (ultime 5 estrazioni al 9 giu 2026)",
        "bridge_note": "Prev chain: May29→May30→Jun3…→Jun9; gap Mar17-May27 non in CSV",
        "sources": [d["source"] for d in LIVE_DRAWS_ITALY],
        "wheels": per_wheel,
        "portfolio_all_11_wheels": finalize(portfolio_all),
        "portfolio_best_5_wheels": finalize(portfolio_best),
    }


# Prev per catena Primitiva: giovedì 28 maggio 2026 (LaVanguardia)
PRIMITIVA_PREV_20260528 = [3, 9, 10, 26, 43, 46]


def verify_spain() -> dict:
    """Cross 1-49 su La Primitiva — 1€ per sorteggio, puntata 6 numeri."""
    details = []
    spent = 0
    won = 0.0
    ordered = PRIMITIVA_DRAWS
    for i, draw in enumerate(ordered):
        if i == 0:
            prev = PRIMITIVA_PREV_20260528
            prev_note = "2026-05-28"
        else:
            prev = ordered[i - 1]["nums"]
            prev_note = ordered[i - 1]["date"]
        picks = pick_top6_cross49(prev)
        drawn = set(draw["nums"])
        hits = sum(1 for p in picks if p in drawn)
        prize = primitiva_prize(hits, draw["prizes"])
        spent += 1
        won += prize
        details.append({
            "date": draw["date"],
            "prev_date": prev_note,
            "prev_nums": prev,
            "picks": picks,
            "drawn": draw["nums"],
            "hits": hits,
            "prize_eur": prize,
            "source": draw["source"],
        })

    net = won - spent
    return {
        "game": "La Primitiva (Spagna)",
        "method": "cross49_adapted_EXPERIMENTAL",
        "warning": "Motore NON validato su storico spagnolo; solo confronto esplorativo",
        "n_draws": len(ordered),
        "spent_eur": spent,
        "won_eur": round(won, 2),
        "net_eur": round(net, 2),
        "draws": details,
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    wide = root / "data" / "draws_wide.csv"
    opt_path = root / "data" / "analysis" / "cyclical_science" / "optimized_weights.json"
    opt_weights = json.loads(opt_path.read_text(encoding="utf-8")) if opt_path.is_file() else {}

    out_dir = root / "data" / "analysis" / "cyclical_science"
    out_dir.mkdir(parents=True, exist_ok=True)

    italy = verify_italy(wide, opt_weights, "ambo")
    spain = verify_spain()

    payload = {
        "verified_at": "2026-06-09",
        "italy_lotto": italy,
        "spain_primitiva": spain,
    }
    out_path = out_dir / "live_verify_jun2026.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Report console
    pa = italy["portfolio_all_11_wheels"]
    pb = italy["portfolio_best_5_wheels"]
    print("=== ITALIA — ultime 5 estrazioni (ambo, 1€/ruota/estrazione) ===")
    print(f"Portfolio 11 ruote: speso {pa['spent']}€ | vinto lordo {pa['won_gross']:.0f}€ | netto lordo {pa['net_gross_eur']:+.0f}€ | ambo {pa['hits']}")
    if pa["details"]:
        print(f"  Hit: {', '.join(pa['details'])}")
    print(f"Portfolio BEST 5 ruote: speso {pb['spent']}€ | vinto lordo {pb['won_gross']:.0f}€ | netto lordo {pb['net_gross_eur']:+.0f}€ | ambo {pb['hits']}")
    if pb["details"]:
        print(f"  Hit: {', '.join(pb['details'])}")

    print("\nDettaglio per ruota (ambo):")
    for w in ALL_WHEELS:
        wr = italy["wheels"][w]
        print(f"  {w} ({wr['mode']}): {wr['hits']}/{wr['spent']} hit, {wr['won_gross'] - wr['spent']:+.0f}€")
        for t in wr["transitions"]:
            mark = " *** AMBO ***" if t["hit"] else ""
            print(f"    {t['to_date']}: gioca {t['picks']} vs {t['drawn']}{mark}")

    print("\n=== SPAGNA — La Primitiva (cross49 sperimentale) ===")
    print(f"Speso {spain['spent_eur']}€ | vinto {spain['won_eur']:.2f}€ | netto {spain['net_eur']:+.2f}€")
    for d in spain["draws"]:
        print(f"  {d['date']}: {d['hits']} aciertos, picks {d['picks']} vs {d['drawn']} → {d['prize_eur']:.2f}€")

    print(f"\nScritto {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
