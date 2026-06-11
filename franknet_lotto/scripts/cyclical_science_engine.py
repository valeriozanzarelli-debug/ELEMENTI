#!/usr/bin/env python3
"""
Motore 'scienza ciclica': pesi complessi per ogni numero 1..90 in funzione di
  - calendario (giorno anno, mese, weekday)
  - indice estrazione nell'anno (draw_index) e armoniche mod P
  - posizione nel ciclo copertura 1..90
  - seme pos1 e profilo flusso
  - memoria k-NN su vettori di fase storici

Obiettivo dichiarato: spingere verso cinquina (top-5 = quintina esatta).
Valuta train/test cronologico su 156 anni Franknet (1871-2026).

Output: data/analysis/cyclical_science/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np

from simulate_ambi_2025 import score_cross_numbers, top_n_from_pool
from vertibile_rule_test import pool_cross_opt_v1

ALL_WHEELS = [
    "BARI", "CAGLIARI", "FIRENZE", "GENOVA", "MILANO", "NAPOLI",
    "PALERMO", "ROMA", "TORINO", "VENEZIA", "NAZIONALE",
]

HARMONIC_PERIODS = (5, 7, 9, 11, 13, 17, 23, 29, 37, 52, 90)
MOD_CYCLES = (3, 4, 5, 7, 9, 13, 17, 52)


def digital_root(n: int) -> int:
    x = n
    while x > 9:
        x = sum(int(c) for c in str(x))
    return x


def signed_diffs(nums: list[int]) -> list[int]:
    return [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]


@dataclass
class DrawRecord:
    date: str
    year: int
    month: int
    day: int
    weekday: int
    day_of_year: int
    draw_index: int
    global_idx: int
    cycle_pos: int
    cycle_id: int
    nums: list[int]
    anchor: int
    flow_sum: int
    phase_vec: tuple[int, ...]


class CouponTracker:
    """Posizione nel ciclo copertura 1..90 (come coupon_cycle_stream)."""

    def __init__(self) -> None:
        self.seen: set[int] = set()
        self.cycle_id = 0
        self.pos_in_cycle = 0

    def observe(self, nums: list[int]) -> tuple[int, int]:
        cid = self.cycle_id
        pos = self.pos_in_cycle
        for x in nums:
            if 1 <= x <= 90 and x not in self.seen:
                self.seen.add(x)
                self.pos_in_cycle += 1
                if len(self.seen) == 90:
                    self.seen = set()
                    self.cycle_id += 1
                    self.pos_in_cycle = 0
        return cid, pos


def build_phase_vector(meta: dict) -> tuple[int, ...]:
    di = meta["draw_index"]
    doy = meta["day_of_year"]
    return (
        meta["weekday"],
        meta["month"],
        di % 4,
        di % 7,
        di % 9,
        di % 13,
        di % 52,
        doy // 30,
        meta["anchor_decile"],
        digital_root(meta["anchor"]),
        meta["cycle_pos"] // 9,
        meta["cycle_id"] % 7,
        meta["flow_sum"] // 15,
    )


def load_records(wide: Path, wheel: str) -> list[DrawRecord]:
    cols = [f"{wheel}_{k}" for k in range(1, 6)]
    tracker = CouponTracker()
    records: list[DrawRecord] = []
    gidx = 0
    with wide.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not all(row[c] for c in cols):
                continue
            nums = [int(row[c]) for c in cols]
            dt = date.fromisoformat(row["date"])
            cid, cpos = tracker.observe(nums)
            anchor = nums[0]
            v = signed_diffs(nums)
            flow_sum = ((sum(v) - 1) % 90) + 1 if v else anchor
            meta = {
                "weekday": dt.weekday(),
                "month": dt.month,
                "draw_index": int(row["draw_index_year"]),
                "day_of_year": dt.timetuple().tm_yday,
                "anchor": anchor,
                "anchor_decile": min(9, (anchor - 1) * 10 // 90),
                "cycle_pos": cpos,
                "cycle_id": cid,
                "flow_sum": flow_sum,
            }
            pv = build_phase_vector(meta)
            records.append(
                DrawRecord(
                    date=row["date"],
                    year=dt.year,
                    month=dt.month,
                    day=dt.day,
                    weekday=dt.weekday(),
                    day_of_year=meta["day_of_year"],
                    draw_index=meta["draw_index"],
                    global_idx=gidx,
                    cycle_pos=cpos,
                    cycle_id=cid,
                    nums=nums,
                    anchor=anchor,
                    flow_sum=flow_sum,
                    phase_vec=pv,
                )
            )
            gidx += 1
    return records


class HarmonicPredictor:
    """Per ogni numero: regressione ridge su armoniche temporali global_idx."""

    def __init__(self, periods: tuple[int, ...] = HARMONIC_PERIODS, ridge: float = 2.0) -> None:
        self.periods = periods
        self.ridge = ridge
        self.coef: dict[int, np.ndarray] = {}
        self.n_feat = 1 + 2 * len(periods)

    def _features(self, t: np.ndarray) -> np.ndarray:
        cols = [np.ones_like(t, dtype=np.float64)]
        for p in self.periods:
            ang = 2 * math.pi * t / p
            cols.append(np.sin(ang))
            cols.append(np.cos(ang))
        return np.column_stack(cols)

    def fit(self, records: list[DrawRecord]) -> None:
        t = np.array([r.global_idx for r in records], dtype=np.float64)
        X = self._features(t)
        xt_x = X.T @ X + self.ridge * np.eye(self.n_feat)
        xt_x_inv = np.linalg.inv(xt_x)
        for n in range(1, 91):
            y = np.array([1.0 if n in r.nums else 0.0 for r in records])
            self.coef[n] = xt_x_inv @ (X.T @ y)

    def predict(self, global_idx: int) -> dict[int, float]:
        t = np.array([global_idx], dtype=np.float64)
        x = self._features(t)
        out: dict[int, float] = {}
        for n, c in self.coef.items():
            out[n] = float((x @ c).flat[0])
        return out


class PhaseLattice:
    """Conta apparizioni numero per bucket di fase (multi-livello)."""

    def __init__(self) -> None:
        self.hits: dict[tuple, Counter] = defaultdict(Counter)
        self.trials: Counter = Counter()

    def _keys(self, vec: tuple[int, ...]) -> list[tuple]:
        keys = [
            ("full", vec),
            ("wd", (vec[0],)),
            ("mod7", (vec[2], vec[5])),
            ("month_mod13", (vec[1], vec[5])),
            ("anchor_cycle", (vec[8], vec[10], vec[11])),
            ("doy_slot", (vec[7], vec[6])),
        ]
        return keys

    def observe(self, vec: tuple[int, ...], nums: list[int]) -> None:
        for prefix, key in self._keys(vec):
            fk = (prefix, key)
            self.trials[fk] += 1
            for n in nums:
                self.hits[fk][n] += 1

    def score(self, vec: tuple[int, ...]) -> dict[int, float]:
        acc: dict[int, float] = defaultdict(float)
        for prefix, key in self._keys(vec):
            fk = (prefix, key)
            t = self.trials.get(fk, 0)
            if t < 12:
                continue
            for n, h in self.hits[fk].items():
                rate = h / t
                lift = rate / (5 / 90)
                acc[n] += math.log(max(lift, 0.05))
        return dict(acc)


class PhaseCooccurrence:
    """Coppie che escono insieme in draws con fase simile (bucket ridotto)."""

    def __init__(self) -> None:
        self.pair_hits: dict[tuple, Counter] = defaultdict(Counter)
        self.trials: Counter = Counter()

    @staticmethod
    def bucket(vec: tuple[int, ...]) -> tuple:
        return (vec[0], vec[1], vec[5], vec[8])

    def observe(self, vec: tuple[int, ...], nums: list[int]) -> None:
        b = self.bucket(vec)
        self.trials[b] += 1
        for i, a in enumerate(nums):
            for bnum in nums[i + 1 :]:
                pair = (min(a, bnum), max(a, bnum))
                self.pair_hits[b][pair] += 1

    def combo_bonus(self, vec: tuple[int, ...], combo: tuple[int, ...]) -> float:
        b = self.bucket(vec)
        t = self.trials.get(b, 0)
        if t < 15:
            return 0.0
        s = 0.0
        for a, c in combinations(combo, 2):
            pair = (min(a, c), max(a, c))
            h = self.pair_hits[b].get(pair, 0)
            s += h / t
        return s


class PhaseKNN:
    """k vicini storici per vettore fase → distribuzione numeri uscita successiva."""

    def __init__(self, k: int = 40) -> None:
        self.k = k
        self.vecs: np.ndarray | None = None
        self.next_nums: list[set[int]] = []

    def fit(self, records: list[DrawRecord]) -> None:
        if len(records) < 2:
            return
        self.vecs = np.array([r.phase_vec for r in records[:-1]], dtype=np.float64)
        self.next_nums = [set(records[i + 1].nums) for i in range(len(records) - 1)]

    def score(self, vec: tuple[int, ...]) -> dict[int, float]:
        if self.vecs is None or len(self.vecs) == 0:
            return {}
        q = np.array(vec, dtype=np.float64)
        d = np.sum((self.vecs - q) ** 2, axis=1)
        idx = np.argpartition(d, min(self.k, len(d) - 1))[: self.k]
        ctr: Counter = Counter()
        for i in idx:
            for n in self.next_nums[i]:
                ctr[n] += 1
        tot = sum(ctr.values())
        if tot == 0:
            return {}
        return {n: c / tot * 90 for n, c in ctr.items()}


@dataclass(frozen=True)
class WeightConfig:
    w_cross: float = 1.0
    w_harmonic: float = 80.0
    w_lattice: float = 12.0
    w_knn: float = 25.0
    w_anchor_echo: float = 8.0
    w_cooc: float = 50.0
    score_all_90: bool = True
    pick_pool_size: int = 18
    confidence_min_margin: float = 0.0


def anchor_echo_score(anchor: int, flow_sum: int) -> dict[int, float]:
    """Eco del seme: trasformazioni deterministiche del primo numero."""
    s: dict[int, float] = defaultdict(float)

    def bump(x: int, w: float) -> None:
        if 1 <= x <= 90:
            s[x] += w

    bump(anchor, 5)
    bump(90 - anchor if anchor not in (45, 90) else 45, 4)
    if 10 <= anchor <= 90:
        rev = int(str(anchor)[::-1])
        if 1 <= rev <= 90:
            bump(rev, 3)
    bump(flow_sum, 4)
    bump((anchor + flow_sum - 1) % 90 + 1, 3)
    bump((anchor - flow_sum - 1) % 90 + 1, 3)
    for k in range(1, 6):
        bump((anchor + k * 11 - 1) % 90 + 1, 2)
        bump((anchor + k * 13 - 1) % 90 + 1, 2)
    return dict(s)


def science_score(
    prev: DrawRecord,
    next_phase: DrawRecord,
    models: dict,
    cfg: WeightConfig,
) -> dict[int, float]:
    nums = prev.nums
    pool = pool_cross_opt_v1(nums)
    cross = score_cross_numbers(nums)
    harm = models["harmonic"].predict(next_phase.global_idx)
    lattice = models["lattice"].score(next_phase.phase_vec)
    knn = models["knn"].score(next_phase.phase_vec)
    echo = anchor_echo_score(prev.anchor, prev.flow_sum)
    universe = range(1, 91) if cfg.score_all_90 else pool

    scores: dict[int, float] = {}
    for n in universe:
        sc = (
            cfg.w_cross * cross.get(n, 0)
            + cfg.w_harmonic * harm.get(n, 0)
            + cfg.w_lattice * lattice.get(n, 0)
            + cfg.w_knn * knn.get(n, 0)
            + cfg.w_anchor_echo * echo.get(n, 0)
        )
        if sc > 0:
            scores[n] = sc
    return scores


def select_cinquina_combo(
    scores: dict[int, float],
    phase_vec: tuple[int, ...],
    cooc: PhaseCooccurrence,
    cfg: WeightConfig,
) -> list[int]:
    """Sceglie 5 numeri: combinatoria su top-M + bonus coppie fase."""
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    if len(ranked) < 5:
        return [n for n, _ in ranked]
    m = min(cfg.pick_pool_size, len(ranked))
    candidates = [n for n, _ in ranked[:m]]
    best_combo: tuple[int, ...] | None = None
    best_s = -1.0
    for combo in combinations(candidates, 5):
        base = sum(scores.get(x, 0) for x in combo)
        bonus = cfg.w_cooc * cooc.combo_bonus(phase_vec, combo)
        tot = base + bonus
        if tot > best_s:
            best_s = tot
            best_combo = combo
    return list(best_combo) if best_combo else candidates[:5]


def score_confidence(scores: dict[int, float], picks: list[int]) -> float:
    ranked = sorted(scores.values(), reverse=True)
    if len(ranked) < 6:
        return 0.0
    top5_avg = sum(ranked[:5]) / 5
    rest_avg = sum(ranked[5:15]) / max(1, len(ranked[5:15]))
    return top5_avg - rest_avg


def evaluate_transitions(
    pairs: list[tuple[DrawRecord, DrawRecord]],
    models: dict,
    cfg: WeightConfig,
    *,
    use_combo: bool = True,
    gated: bool = False,
) -> dict:
    stats = Counter()
    monthly: dict[str, Counter] = defaultdict(Counter)
    cooc: PhaseCooccurrence = models["cooc"]
    for prev, nxt in pairs:
        sc = science_score(prev, nxt, models, cfg)
        if use_combo:
            picks = select_cinquina_combo(sc, nxt.phase_vec, cooc, cfg)
        else:
            pool = pool_cross_opt_v1(prev.nums)
            picks = top_n_from_pool(pool, sc, 5)
        conf = score_confidence(sc, picks)
        if gated and conf < cfg.confidence_min_margin:
            stats["skipped_low_conf"] += 1
            continue
        drawn = set(nxt.nums)
        h = sum(1 for x in picks if x in drawn)
        stats["draws"] += 1
        stats[f"hit_{h}"] += 1
        if h == 5:
            stats["cinquina"] += 1
            monthly[nxt.date[:7]]["cinquina"] += 1
        if h >= 4:
            stats["quaterna_plus"] += 1
            monthly[nxt.date[:7]]["quaterna_plus"] += 1
        if h >= 3:
            stats["terno_plus"] += 1
    return {"stats": dict(stats), "monthly": {k: dict(v) for k, v in monthly.items()}}


def grid_configs() -> list[WeightConfig]:
    cfgs: list[WeightConfig] = []
    for wh in (80, 120):
        for wl in (12, 20):
            for wk in (25, 40):
                for wa in (8, 14):
                    for wc in (30, 60):
                        for ps in (15, 18, 22):
                            cfgs.append(
                                WeightConfig(
                                    w_cross=1.0,
                                    w_harmonic=wh,
                                    w_lattice=wl,
                                    w_knn=wk,
                                    w_anchor_echo=wa,
                                    w_cooc=wc,
                                    score_all_90=True,
                                    pick_pool_size=ps,
                                )
                            )
    return cfgs


def fit_models(train: list[DrawRecord]) -> dict:
    harm = HarmonicPredictor()
    harm.fit(train)
    lattice = PhaseLattice()
    cooc = PhaseCooccurrence()
    for r in train:
        lattice.observe(r.phase_vec, r.nums)
        cooc.observe(r.phase_vec, r.nums)
    knn = PhaseKNN(k=35)
    knn.fit(train)
    return {"harmonic": harm, "lattice": lattice, "knn": knn, "cooc": cooc}


def pairs_from(records: list[DrawRecord]) -> list[tuple[DrawRecord, DrawRecord]]:
    return [(records[i], records[i + 1]) for i in range(len(records) - 1)]


def portfolio_monthly(
    all_data: dict[str, list[DrawRecord]],
    test_from: int,
    cfg: WeightConfig,
) -> dict:
    """11 ruote: almeno una cinquina nel mese?"""
    by_month: dict[str, list] = defaultdict(list)
    for wheel, recs in all_data.items():
        if len(recs) < 50:
            continue
        split = next((i for i, r in enumerate(recs) if r.year >= test_from), len(recs) // 2)
        train, test_recs = recs[:split], recs[split:]
        if len(train) < 100 or len(test_recs) < 20:
            continue
        models = fit_models(train)
        for prev, nxt in pairs_from(test_recs):
            sc = science_score(prev, nxt, models, cfg)
            picks = select_cinquina_combo(sc, nxt.phase_vec, models["cooc"], cfg)
            h = sum(1 for x in picks if x in set(nxt.nums))
            by_month[nxt.date[:7]].append(
                {"wheel": wheel, "hits": h, "cinquina": h == 5, "date": nxt.date}
            )
    months_with_cinquina = sum(
        1 for m, evs in by_month.items() if any(e["cinquina"] for e in evs)
    )
    total_cinquina = sum(1 for evs in by_month.values() for e in evs if e["cinquina"])
    return {
        "months": len(by_month),
        "months_with_any_cinquina": months_with_cinquina,
        "total_wheel_cinquina": total_cinquina,
        "by_month": {
            m: {
                "events": len(evs),
                "best_hit": max(e["hits"] for e in evs),
                "cinquina_wheels": [e["wheel"] for e in evs if e["cinquina"]],
            }
            for m, evs in sorted(by_month.items())
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", default=None)
    ap.add_argument("--train-until-year", type=int, default=2014)
    ap.add_argument("--test-from-year", type=int, default=2015)
    ap.add_argument("--grid", action="store_true", help="Grid search pesi su train")
    ap.add_argument("--portfolio", action="store_true", help="11 ruote mese per mese")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    wide = root / "data" / "draws_wide.csv"
    out = root / "data" / "analysis" / "cyclical_science"
    out.mkdir(parents=True, exist_ok=True)

    wheels = [args.wheel.upper()] if args.wheel else ALL_WHEELS
    all_data: dict[str, list[DrawRecord]] = {}
    for w in wheels:
        all_data[w] = load_records(wide, w)

    reports = []
    best_global_cfg = WeightConfig()

    for wheel, recs in all_data.items():
        if len(recs) < 200:
            continue
        split = next((i for i, r in enumerate(recs) if r.year > args.train_until_year), int(len(recs) * 0.75))
        train_recs, test_recs = recs[:split], recs[split:]
        tr_pairs = pairs_from(train_recs)
        te_pairs = pairs_from(test_recs)

        if args.grid and len(tr_pairs) > 100:
            best_cfg = WeightConfig()
            best_cinq = -1
            best_q = -1
            models = fit_models(train_recs)
            for cfg in grid_configs():
                ev = evaluate_transitions(tr_pairs[-600:], models, cfg, use_combo=True)
                st = ev["stats"]
                cinq = st.get("cinquina", 0)
                quat = st.get("quaterna_plus", 0)
                score = cinq * 1000 + quat * 10 + st.get("hit_3", 0)
                if score > best_cinq:
                    best_cinq, best_q, best_cfg = score, quat, cfg
            cfg = best_cfg
        else:
            cfg = WeightConfig(
                w_harmonic=120,
                w_lattice=20,
                w_knn=40,
                w_anchor_echo=14,
                w_cooc=60,
                score_all_90=True,
                pick_pool_size=18,
            )

        models = fit_models(train_recs)
        cross_stats = Counter()
        for prev, nxt in te_pairs:
            pool = pool_cross_opt_v1(prev.nums)
            picks = top_n_from_pool(pool, score_cross_numbers(prev.nums), 5)
            h = sum(1 for x in picks if x in set(nxt.nums))
            cross_stats[f"hit_{h}"] += 1
            if h == 5:
                cross_stats["cinquina"] += 1
            if h >= 4:
                cross_stats["quaterna_plus"] += 1

        sci_ev = evaluate_transitions(te_pairs, models, cfg, use_combo=True)
        sci_gated = evaluate_transitions(
            te_pairs, models, cfg, use_combo=True, gated=True,
        )
        if sci_gated["stats"].get("draws", 0) > 0:
            rep_gate = sci_gated["stats"]
        else:
            rep_gate = {}
        st = sci_ev["stats"]
        rep = {
            "wheel": wheel,
            "train_draws": len(train_recs),
            "test_draws": len(te_pairs),
            "train_years": f"{train_recs[0].year}-{train_recs[-1].year}",
            "test_years": f"{test_recs[0].year}-{test_recs[-1].year}",
            "weights": asdict(cfg),
            "test_cross": dict(cross_stats),
            "test_science": st,
            "test_science_gated": rep_gate,
            "months_cinquina_science": sum(
                1 for m, c in sci_ev["monthly"].items() if c.get("cinquina", 0) > 0
            ),
            "monthly_science": sci_ev["monthly"],
        }
        reports.append(rep)
        print(
            f"{wheel}: test cinquina cross={cross_stats.get('cinquina',0)} "
            f"science={st.get('cinquina',0)} quaterna+ sci={st.get('quaterna_plus',0)} "
            f"hit4 sci={st.get('hit_4',0)}"
        )
        best_global_cfg = cfg

    port = {}
    if args.portfolio:
        port = portfolio_monthly(all_data, args.test_from_year, best_global_cfg)
        print(
            f"PORTFOLIO 11 ruote: cinquine totali={port['total_wheel_cinquina']} "
            f"mesi con cinquina={port['months_with_any_cinquina']}/{port['months']}"
        )

    payload = {
        "method": "cyclical_science_engine",
        "data_span": "1871-2026 Franknet",
        "phase_vector_dims": 13,
        "harmonic_periods": list(HARMONIC_PERIODS),
        "interpretation": (
            "Pesi = cross + armoniche Fourier su global_idx + reticolo fase "
            "(weekday, mese, draw mod, ciclo90, seme) + k-NN fase + eco seme. "
            "Cinquina = top-5 scored == quintina estratta."
        ),
        "wheels": reports,
        "portfolio_11_wheels": port,
        "honest_note": (
            "Cinquina esatta è ~1/44M a caso per 5 numeri fissi. Il motore cerca "
            "risnanze cicliche; obiettivo '1/mese' richiede verifica su portfolio."
        ),
    }
    (out / "cyclical_science_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        "# Scienza ciclica — report",
        "",
        f"Train fino al {args.train_until_year}, test dal {args.test_from_year}.",
        "",
        "| ruota | cinquina cross | cinquina science | quaterna+ science | mesi con cinquina |",
        "|-------|----------------|------------------|-------------------|-------------------|",
    ]
    for r in reports:
        md.append(
            f"| {r['wheel']} | {r['test_cross'].get('cinquina', 0)} | "
            f"{r['test_science'].get('cinquina', 0)} | "
            f"{r['test_science'].get('quaterna_plus', 0)} | {r['months_cinquina_science']} |"
        )
    if port:
        md.extend(
            [
                "",
                f"**Portfolio 11 ruote:** {port['total_wheel_cinquina']} cinquine, "
                f"{port['months_with_any_cinquina']}/{port['months']} mesi con almeno una.",
            ]
        )
    (out / "cyclical_science_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nScritto {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
