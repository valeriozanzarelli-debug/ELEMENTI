#!/usr/bin/env python3
"""
Pesi per *periodo* di estrazione e ancoraggio sul primo numero (pos1).

Ipotesi operativa (proxy misurabili su Franknet, senza dati su aria/operatore):
  - Il momento dell'estrazione modula il flusso: giorno settimana, indice estrazione
    nell'anno (draw_index_year), fase in un ciclo mod P.
  - Il primo numero estratto (pos1) è il 'seme' — le oscillazioni successive
    (pos2..pos5, velocità signed) possono ripetersi quando la fase è la stessa.

Metodo:
  1. Definisce chiavi di fase (weekday, draw_mod, anchor_decile, compositi).
  2. Su train: conta hit empirici pool cross → quintina successiva per fase.
  3. Score = cross_base + bonus_fase[n] (numeri che storicamente colgono in quella fase).
  4. Calendario PLAY/WAIT: fasi dove train top-8 batte MC p95.
  5. Pattern ancorato: profilo medio pos2..pos5 e velocità dato (fase, anchor_decile).

Output: data/analysis/phase_weights/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from simulate_ambi_2025 import score_cross_numbers, top_n_from_pool
from vertibile_rule_test import monte_carlo_baseline, pool_cross_opt_v1

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

WEEKDAY_IT = ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]


def digital_root(n: int) -> int:
    if n <= 0:
        return 0
    x = n
    while x > 9:
        x = sum(int(c) for c in str(x))
    return x


def anchor_decile(n: int) -> int:
    return min(9, (n - 1) * 10 // 90)


def signed_diffs(nums: list[int]) -> list[int]:
    return [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]


@dataclass(frozen=True)
class DrawMeta:
    date: str
    year: int
    draw_index: int
    weekday: int
    anchor: int
    anchor_decile: int
    anchor_dr: int


def load_wheel_with_meta(wide: Path, wheel: str) -> list[dict]:
    cols = [f"{wheel}_{k}" for k in range(1, 6)]
    rows: list[dict] = []
    with wide.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not all(row[c] for c in cols):
                continue
            nums = [int(row[c]) for c in cols]
            dt = date.fromisoformat(row["date"])
            idx = int(row["draw_index_year"])
            anchor = nums[0]
            rows.append(
                {
                    "date": row["date"],
                    "year": int(row["year"]),
                    "draw_index": idx,
                    "weekday": dt.weekday(),
                    "anchor": anchor,
                    "anchor_decile": anchor_decile(anchor),
                    "anchor_dr": digital_root(anchor),
                    "nums": nums,
                }
            )
    return rows


# --- Chiavi di fase (nome → funzione meta → chiave hashable) ---

PhaseFn = Callable[[dict], str]

PHASE_SCHEMAS: dict[str, PhaseFn] = {}


def _reg(name: str):
    def deco(fn: PhaseFn) -> PhaseFn:
        PHASE_SCHEMAS[name] = fn
        return fn

    return deco


@_reg("weekday")
def phase_weekday(m: dict) -> str:
    return f"wd_{m['weekday']}"


@_reg("draw_mod4")
def phase_draw_mod4(m: dict) -> str:
    return f"d4_{m['draw_index'] % 4}"


@_reg("draw_mod7")
def phase_draw_mod7(m: dict) -> str:
    return f"d7_{m['draw_index'] % 7}"


@_reg("draw_mod9")
def phase_draw_mod9(m: dict) -> str:
    return f"d9_{m['draw_index'] % 9}"


@_reg("anchor_decile")
def phase_anchor_decile(m: dict) -> str:
    return f"ad_{m['anchor_decile']}"


@_reg("anchor_dr")
def phase_anchor_dr(m: dict) -> str:
    return f"dr_{m['anchor_dr']}"


@_reg("weekday_x_draw_mod4")
def phase_wd_x_d4(m: dict) -> str:
    return f"wd{m['weekday']}_d4_{m['draw_index'] % 4}"


@_reg("anchor_decile_x_draw_mod7")
def phase_ad_x_d7(m: dict) -> str:
    return f"ad{m['anchor_decile']}_d7_{m['draw_index'] % 7}"


@_reg("anchor_x_weekday")
def phase_anchor_x_wd(m: dict) -> str:
    return f"a{m['anchor']}_wd{m['weekday']}"


@_reg("full_phase")
def phase_full(m: dict) -> str:
    """Composito: giorno + ciclo7 + decile seme."""
    return (
        f"wd{m['weekday']}_d7_{m['draw_index'] % 7}_ad{m['anchor_decile']}"
    )


class PhaseHitTable:
    """Hit empirici per (schema, phase_key, number) su train."""

    def __init__(self) -> None:
        self.hits: dict[str, dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self.trials: dict[str, Counter] = defaultdict(Counter)

    def observe(self, schema: str, phase_key: str, next_nums: set[int]) -> None:
        self.trials[schema][phase_key] += 1
        for n in next_nums:
            self.hits[schema][phase_key][n] += 1

    def bonus(self, schema: str, phase_key: str, n: int, *, smooth: float = 1.0) -> float:
        t = self.trials[schema].get(phase_key, 0)
        if t <= 0:
            return 0.0
        h = self.hits[schema][phase_key].get(n, 0)
        # rate smoothed vs uniform 5/90
        rate = (h + smooth) / (t + smooth * 90)
        return max(0.0, (rate - 5 / 90) * 100)


def build_phase_tables(
    pairs: list[tuple[dict, dict]],
    schemas: list[str],
) -> PhaseHitTable:
    tbl = PhaseHitTable()
    for prev, nxt in pairs:
        for name in schemas:
            if name not in PHASE_SCHEMAS:
                continue
            key = PHASE_SCHEMAS[name](prev)
            tbl.observe(name, key, set(nxt["nums"]))
    return tbl


def score_phase_adjusted(
    nums: list[int],
    meta: dict,
    table: PhaseHitTable,
    schema: str,
    *,
    cross_scale: float = 1.0,
    phase_scale: float = 1.0,
) -> dict[int, int]:
    base = score_cross_numbers(nums)
    phase_key = PHASE_SCHEMAS[schema](meta)
    out: dict[int, int] = {}
    pool = pool_cross_opt_v1(nums)
    for n in pool:
        b = base.get(n, 0)
        ph = table.bonus(schema, phase_key, n)
        out[n] = int(b * cross_scale + ph * phase_scale)
    return out


def to_num_pairs(pairs: list[tuple[dict, dict]]) -> list[tuple[list[int], list[int]]]:
    return [(a["nums"], b["nums"]) for a, b in pairs]


def mc_topk_p95(
    pairs: list[tuple[dict, dict]],
    top_k: int,
    reps: int,
    seed: int,
) -> int:
    num_pairs = to_num_pairs(pairs)
    _, p95, _ = monte_carlo_baseline(
        num_pairs, [top_k] * len(num_pairs), reps=reps, seed=seed
    )
    return p95


def eval_topk_hits(
    pairs: list[tuple[dict, dict]],
    score_fn: Callable[[dict], dict[int, int]],
    top_k: int,
) -> tuple[int, list[int]]:
    hits = 0
    sizes: list[int] = []
    for prev, nxt in pairs:
        pool = pool_cross_opt_v1(prev["nums"])
        scores = score_fn(prev)
        picks = top_n_from_pool(pool, scores, top_k)
        sizes.append(len(picks))
        hits += sum(1 for x in nxt["nums"] if x in picks)
    return hits, sizes


def anchor_oscillation_profile(
    rows: list[dict],
    schema: str,
) -> list[dict]:
    """Profilo medio pos2..pos5 e velocità per bucket di fase."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        k = PHASE_SCHEMAS[schema](r)
        nums = r["nums"]
        v = signed_diffs(nums)
        buckets[k].append(
            {
                "pos2": nums[1],
                "pos3": nums[2],
                "pos4": nums[3],
                "pos5": nums[4],
                "v1": v[0] if v else 0,
                "v2": v[1] if len(v) > 1 else 0,
                "v3": v[2] if len(v) > 2 else 0,
                "v4": v[3] if len(v) > 3 else 0,
            }
        )
    out = []
    for key, items in sorted(buckets.items()):
        if len(items) < 5:
            continue
        prof = {}
        for field in ("pos2", "pos3", "pos4", "pos5", "v1", "v2", "v3", "v4"):
            vals = [x[field] for x in items]
            prof[f"mean_{field}"] = round(statistics.mean(vals), 3)
            prof[f"std_{field}"] = round(
                statistics.stdev(vals) if len(vals) > 1 else 0, 3
            )
        out.append(
            {
                "phase_key": key,
                "n": len(items),
                **prof,
            }
        )
    return out


def play_calendar(
    pairs_train: list[tuple[dict, dict]],
    pairs_test: list[tuple[dict, dict]],
    schema: str,
    table: PhaseHitTable,
    top_k: int,
    mc_reps: int,
    seed: int,
) -> list[dict]:
    """Per ogni fase vista in train: se top-8 train > MC p95 → PLAY, altrimenti WAIT."""
    phase_train: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    phase_test: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for p in pairs_train:
        phase_train[PHASE_SCHEMAS[schema](p[0])].append(p)
    for p in pairs_test:
        phase_test[PHASE_SCHEMAS[schema](p[0])].append(p)

    rows = []
    for phase_key, tr in phase_train.items():
        if len(tr) < 8:
            continue

        def scorer(m: dict) -> dict[int, int]:
            return score_phase_adjusted(
                m["nums"], m, table, schema, phase_scale=2.0
            )

        tr_hits, _ = eval_topk_hits(tr, scorer, top_k)
        mc_p95 = mc_topk_p95(tr, top_k, mc_reps, seed)
        signal = "PLAY" if tr_hits > mc_p95 else "WAIT"

        te = phase_test.get(phase_key, [])
        te_hits = None
        te_n = 0
        if te:
            te_hits, _ = eval_topk_hits(te, scorer, top_k)
            te_n = len(te)

        rows.append(
            {
                "schema": schema,
                "phase_key": phase_key,
                "train_n": len(tr),
                "train_top8_hits": tr_hits,
                "train_mc_p95": mc_p95,
                "signal": signal,
                "test_n": te_n,
                "test_top8_hits": te_hits,
            }
        )
    return rows


@dataclass
class WheelPhaseReport:
    wheel: str
    best_schema: str
    test_top8_cross: int
    test_top8_phase: int
    test_mc_p95: int
    play_phases_train: int
    play_phases_test_hits: int
    play_phases_test_possible: int


def analyze_wheel(
    wheel: str,
    pairs_train: list[tuple[dict, dict]],
    pairs_test: list[tuple[dict, dict]],
    *,
    top_k: int,
    mc_reps: int,
    seed: int,
) -> tuple[WheelPhaseReport, PhaseHitTable, list[dict], list[dict]]:
    schemas = list(PHASE_SCHEMAS.keys())
    table = build_phase_tables(pairs_train, schemas)

    def cross_only(m: dict) -> dict[int, int]:
        return score_cross_numbers(m["nums"])

    base_hits, _ = eval_topk_hits(pairs_test, cross_only, top_k)
    mc_p95 = mc_topk_p95(pairs_test, top_k, mc_reps, seed)

    best_schema = schemas[0]
    best_hits = -1
    for schema in schemas:
        def scorer(m: dict, s=schema) -> dict[int, int]:
            return score_phase_adjusted(m["nums"], m, table, s, phase_scale=2.0)

        h, _ = eval_topk_hits(pairs_test, scorer, top_k)
        if h > best_hits:
            best_hits = h
            best_schema = schema

    def best_scorer(m: dict) -> dict[int, int]:
        return score_phase_adjusted(
            m["nums"], m, table, best_schema, phase_scale=2.0
        )

    calendar = play_calendar(
        pairs_train, pairs_test, best_schema, table, top_k, mc_reps, seed
    )
    play_rows = [c for c in calendar if c["signal"] == "PLAY"]
    play_test_draws = sum(c["test_n"] for c in play_rows)
    play_test_hit_total = sum(c["test_top8_hits"] or 0 for c in play_rows)

    profiles = anchor_oscillation_profile(
        [p[0] for p in pairs_train] + [p[0] for p in pairs_test],
        best_schema,
    )

    rep = WheelPhaseReport(
        wheel=wheel,
        best_schema=best_schema,
        test_top8_cross=base_hits,
        test_top8_phase=best_hits,
        test_mc_p95=mc_p95,
        play_phases_train=len(play_rows),
        play_phases_test_hits=play_test_hit_total,
        play_phases_test_possible=play_test_draws,
    )
    return rep, table, calendar, profiles


def main() -> int:
    ap = argparse.ArgumentParser(description="Pesi per periodo/fase estrazione")
    ap.add_argument("--wheel", default=None)
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--mc-reps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    wide = root / "data" / "draws_wide.csv"
    out_dir = root / "data" / "analysis" / "phase_weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    wheels = [args.wheel.upper()] if args.wheel else ALL_WHEELS
    reports: list[WheelPhaseReport] = []
    all_calendar: list[dict] = []

    for wheel in wheels:
        rows = load_wheel_with_meta(wide, wheel)
        need = args.train + args.test + 1
        if len(rows) < need:
            print(f"Skip {wheel}: servono {need}, trovati {len(rows)}")
            continue
        chunk = rows[-need:]
        pairs = [(chunk[i], chunk[i + 1]) for i in range(len(chunk) - 1)]
        pairs_train = pairs[: args.train]
        pairs_test = pairs[args.train : args.train + args.test]

        rep, _, cal, _ = analyze_wheel(
            wheel,
            pairs_train,
            pairs_test,
            top_k=args.top_k,
            mc_reps=args.mc_reps,
            seed=args.seed,
        )
        reports.append(rep)
        all_calendar.extend({**c, "wheel": wheel} for c in cal)
        delta = rep.test_top8_phase - rep.test_top8_cross
        print(
            f"{wheel}: cross={rep.test_top8_cross} phase={rep.test_top8_phase} "
            f"Δ={delta:+d} schema={rep.best_schema} PLAY_fasi={rep.play_phases_train}"
        )

    if not reports:
        print("Nessuna ruota.", file=sys.stderr)
        return 1

    summary = []
    for r in reports:
        summary.append(
            {
                "wheel": r.wheel,
                "best_schema": r.best_schema,
                "test_top8_cross": r.test_top8_cross,
                "test_top8_phase": r.test_top8_phase,
                "delta": r.test_top8_phase - r.test_top8_cross,
                "test_mc_p95": r.test_mc_p95,
                "beats_mc": r.test_top8_phase > r.test_mc_p95,
                "play_phases_train": r.play_phases_train,
                "play_test_draws": r.play_phases_test_possible,
            }
        )

    with (out_dir / "wheel_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    def human_phase(schema: str, key: str) -> str:
        if key.startswith("wd_"):
            d = int(key.split("_")[1])
            return f"{WEEKDAY_IT[d]} ({key})"
        if key.startswith("ad_"):
            return f"seme decile {key.split('_')[1]} ({key})"
        if key.startswith("d4_"):
            return f"ciclo4 pos {key.split('_')[1]} ({key})"
        if key.startswith("d7_"):
            return f"ciclo7 pos {key.split('_')[1]} ({key})"
        if key.startswith("d9_"):
            return f"ciclo9 pos {key.split('_')[1]} ({key})"
        return key

    for c in all_calendar:
        c["phase_human"] = human_phase(c["schema"], c["phase_key"])

    play_only = [c for c in all_calendar if c["signal"] == "PLAY"]
    play_only.sort(key=lambda x: (-(x.get("test_top8_hits") or 0), x["wheel"]))
    with (out_dir / "play_calendar.csv").open("w", newline="", encoding="utf-8") as f:
        if play_only:
            w = csv.DictWriter(f, fieldnames=list(play_only[0].keys()))
            w.writeheader()
            w.writerows(play_only[:200])

    # Profili oscillazione per NAZIONALE (o prima ruota) come esempio
    example_wheel = "NAZIONALE" if any(r.wheel == "NAZIONALE" for r in reports) else reports[0].wheel
    ex_rows = load_wheel_with_meta(wide, example_wheel)
    ex_chunk = ex_rows[-(args.train + args.test) :]
    best_s = next(r.best_schema for r in reports if r.wheel == example_wheel)
    profiles = anchor_oscillation_profile(ex_chunk, best_s)
    profiles.sort(key=lambda x: -x["n"])
    with (out_dir / f"anchor_profiles_{example_wheel.lower()}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        if profiles:
            w = csv.DictWriter(f, fieldnames=list(profiles[0].keys()))
            w.writeheader()
            w.writerows(profiles[:50])

    # Prossima estrazione: ultima riga wide → fase attesa (draw_index+1, stesso weekday pattern)
    forecasts: list[dict] = []
    with wide.open(encoding="utf-8", newline="") as f:
        all_wide = list(csv.DictReader(f))
    if all_wide:
        last = all_wide[-1]
        next_idx = int(last["draw_index_year"]) + 1
        # stima data: non incrementiamo senza calendario; usiamo meta ultima ruota NAZIONALE
        naz_rows = load_wheel_with_meta(wide, "NAZIONALE")
        if naz_rows:
            last_n = naz_rows[-1]
            for schema_name, fn in PHASE_SCHEMAS.items():
                meta_next = {
                    **last_n,
                    "draw_index": next_idx,
                }
                forecasts.append(
                    {
                        "schema": schema_name,
                        "next_draw_index": next_idx,
                        "phase_key": fn(meta_next),
                        "phase_human": human_phase(schema_name, fn(meta_next)),
                        "note": "draw_index+1; weekday da ultima estrazione (se stesso giorno ruota)",
                    }
                )

    payload = {
        "method": "extraction_phase_weights",
        "train": args.train,
        "test": args.test,
        "top_k": args.top_k,
        "phase_schemas": list(PHASE_SCHEMAS.keys()),
        "interpretation": (
            "Proxy di 'periodo estrazione': weekday, draw_index mod P, decile del primo "
            "numero (pos1=seme). Peso empirico = numeri che nella stessa fase hanno colpito "
            "più del caso sulla quintina T+1. PLAY = fasi con train top-8 > MC p95. "
            "Non abbiamo pressione aria né operatore nel dataset — solo coordinate temporali."
        ),
        "wheels": [asdict(r) for r in reports],
        "aggregate": {
            "wheels": len(reports),
            "phase_beats_cross": sum(1 for s in summary if s["delta"] > 0),
            "phase_beats_mc": sum(1 for s in summary if s["beats_mc"]),
            "best_delta": max(summary, key=lambda x: x["delta"]),
            "total_play_phases": sum(s["play_phases_train"] for s in summary),
        },
        "weekday_legend": {i: WEEKDAY_IT[i] for i in range(7)},
        "next_draw_phase_forecast": forecasts[:12],
        "play_calendar_human": [
            {
                "wheel": c["wheel"],
                "phase": c.get("phase_human", c["phase_key"]),
                "signal": c["signal"],
                "train_n": c["train_n"],
                "test_top8_hits": c.get("test_top8_hits"),
            }
            for c in play_only
        ],
    }
    (out_dir / "phase_weights_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        "# Pesi per periodo di estrazione",
        "",
        "Seme = **primo numero** (pos1). Fase = giorno settimana × ciclo estrazione × decile seme.",
        "",
        "## Ruote (top-8 test)",
        "",
        "| ruota | schema | cross | phase | Δ | PLAY fasi |",
        "|-------|--------|-------|-------|---|-----------|",
    ]
    for s in summary:
        md.append(
            f"| {s['wheel']} | {s['best_schema']} | {s['test_top8_cross']} | "
            f"{s['test_top8_phase']} | {s['delta']:+d} | {s['play_phases_train']} |"
        )
    md.append("")
    md.append("## Fasi PLAY (estratto)")
    md.append("")
    for c in play_only[:15]:
        md.append(
            f"- **{c['wheel']}** {c.get('phase_human', c['phase_key'])} "
            f"train={c['train_n']} signal={c['signal']} test_hits={c.get('test_top8_hits')}"
        )
    md.append("")
    md.append("## Legenda weekday")
    md.append("lun=0 … dom=6 (Python). Lotto classico: mar/gio/sab ≈ wd 1,3,5.")
    (out_dir / "phase_weights_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nScritto in {out_dir}")
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
