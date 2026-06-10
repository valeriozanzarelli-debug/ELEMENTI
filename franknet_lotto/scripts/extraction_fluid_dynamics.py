#!/usr/bin/env python3
"""
Fluidodinamica dell'ordine di estrazione (pos1..pos5 nella quintina).

Prospettiva: la quintina non è un insieme — è un flusso ordinato.
  - velocità v_k = n_{k+1} - n_k  (gap tra posizioni consecutive)
  - accelerazione a_k = v_{k+1} - v_k
  - flusso cumulativo mod 90, inversioni di segno ("turbolenza")

Cerca:
  1. Asimmetria per posizione (marginali, entropia, peso predittivo)
  2. Accoppiamento inter-estrazione pos_k(T) -> pos_j(T+1)
  3. Pool/score con pesi posizionali + ricette sul flusso
  4. Validazione train/test vs cross_opt_v1 e Monte Carlo

Output: data/analysis/fluid_dynamics/
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
from pathlib import Path
from typing import Callable

from simulate_ambi_2025 import score_cross_numbers
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

POSITION_WEIGHT_PRESETS: dict[str, tuple[int, ...]] = {
    "uniform": (1, 1, 1, 1, 1),
    "edge_heavy": (3, 1, 1, 1, 3),
    "center_heavy": (1, 1, 3, 1, 1),
    "pos1_heavy": (4, 1, 1, 1, 1),
    "pos5_heavy": (1, 1, 1, 1, 4),
    "ascending": (1, 2, 3, 4, 5),
    "descending": (5, 4, 3, 2, 1),
}


@dataclass(frozen=True)
class FlowProfile:
    velocities: tuple[int, ...]
    accelerations: tuple[int, ...]
    cumulative_flux_mod90: tuple[int, ...]
    flow_sum_mod90: int
    reversal_count: int
    reynolds: float
    mean_abs_velocity: float


def mod90(x: int) -> int:
    return ((x - 1) % 90) + 1


def signed_diffs(nums: list[int]) -> list[int]:
    """Velocità con segno lungo l'ordine di estrazione pos1→pos5."""
    return [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]


def quintina_flow(nums: list[int]) -> FlowProfile:
    velocities = tuple(signed_diffs(nums))
    accelerations = tuple(
        velocities[i + 1] - velocities[i] for i in range(len(velocities) - 1)
    )
    cum: list[int] = []
    s = 0
    for v in velocities:
        s += v
        cum.append(mod90(s))
    abs_v = [abs(v) for v in velocities]
    mean_abs = statistics.mean(abs_v) if abs_v else 0.0
    std_v = statistics.stdev(velocities) if len(velocities) > 1 else 1.0
    reynolds = mean_abs / std_v if std_v > 1e-9 else 0.0
    reversals = sum(
        1
        for i in range(len(velocities) - 1)
        if velocities[i] * velocities[i + 1] < 0
    )
    return FlowProfile(
        velocities=velocities,
        accelerations=accelerations,
        cumulative_flux_mod90=tuple(cum),
        flow_sum_mod90=mod90(sum(velocities)),
        reversal_count=reversals,
        reynolds=round(reynolds, 6),
        mean_abs_velocity=round(mean_abs, 4),
    )


def shannon_entropy(counts: Counter, n_cat: int) -> float:
    tot = sum(counts.values())
    if tot <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / tot
        h -= p * math.log2(p)
    return h


def position_marginals(draws: list[list[int]]) -> list[dict]:
    rows = []
    for k in range(5):
        ctr: Counter = Counter(d[k] for d in draws)
        h = shannon_entropy(ctr, 90)
        h_max = math.log2(90)
        vals = [d[k] for d in draws]
        rows.append(
            {
                "position": k + 1,
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0,
                "entropy_bits": round(h, 6),
                "max_entropy_bits": round(h_max, 6),
                "entropy_ratio": round(h / h_max, 6) if h_max > 0 else 0,
                "mode": ctr.most_common(1)[0][0] if ctr else None,
                "mode_freq": round(ctr.most_common(1)[0][1] / len(vals), 6) if ctr else 0,
            }
        )
    return rows


def inter_draw_position_coupling(
    pairs: list[tuple[list[int], list[int]]],
) -> list[dict]:
    """Per ogni (pos_prev, pos_next) misura hit rate se predici next[j] = prev[i]."""
    rows = []
    for i in range(5):
        for j in range(5):
            hits = sum(1 for prev, nxt in pairs if prev[i] == nxt[j])
            rows.append(
                {
                    "from_pos": i + 1,
                    "to_pos": j + 1,
                    "same_number_hits": hits,
                    "hit_rate": round(hits / len(pairs), 6) if pairs else 0,
                    "uniform_baseline": round(1 / 90, 6),
                    "lift_vs_uniform": round(
                        (hits / len(pairs)) / (1 / 90) if pairs else 0, 4
                    ),
                }
            )
    return rows


def ordering_shape_stats(draws: list[list[int]]) -> dict:
    """Quante quintine sono monotone crescenti/decrescenti nell'ordine HTML."""
    n = len(draws)
    if n == 0:
        return {}
    asc = desc = strict_asc = strict_desc = 0
    for nums in draws:
        v = signed_diffs(nums)
        if all(x >= 0 for x in v):
            asc += 1
        if all(x <= 0 for x in v):
            desc += 1
        if all(x > 0 for x in v):
            strict_asc += 1
        if all(x < 0 for x in v):
            strict_desc += 1
    return {
        "n_draws": n,
        "non_decreasing_pos1_to_pos5": round(asc / n, 4),
        "non_increasing_pos1_to_pos5": round(desc / n, 4),
        "strictly_increasing": round(strict_asc / n, 4),
        "strictly_decreasing": round(strict_desc / n, 4),
        "note": "Ordine colonne = ordine Franknet HTML, non necessariamente ordine fisico estrazione.",
    }


def velocity_gap_stats(draws: list[list[int]]) -> list[dict]:
    """Distribuzione empirica di v_k per gap k (1->2, 2->3, ...)."""
    by_gap: list[list[int]] = [[] for _ in range(4)]
    for nums in draws:
        v = signed_diffs(nums)
        for k, val in enumerate(v):
            by_gap[k].append(val)
    rows = []
    for k, vals in enumerate(by_gap):
        abs_vals = [abs(x) for x in vals]
        rows.append(
            {
                "gap": f"pos{k + 1}_to_pos{k + 2}",
                "mean_signed": round(statistics.mean(vals), 4),
                "std_signed": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0,
                "mean_abs": round(statistics.mean(abs_vals), 4),
                "pct_positive": round(sum(1 for x in vals if x > 0) / len(vals), 4),
                "pct_negative": round(sum(1 for x in vals if x < 0) / len(vals), 4),
                "pct_zero": round(sum(1 for x in vals if x == 0) / len(vals), 4),
            }
        )
    return rows


def expand_flow_value(pool: set[int], x: int, *, twin11: bool = True, w: int = 1) -> None:
    """Aggiunge x e trasformazioni standard con peso implicito (chiamante gestisce score)."""
    add(pool, x)
    add(pool, comp(x))
    v = vertibile(x)
    if v:
        add(pool, v)
        add(pool, comp(v))
    if w > 0:
        for nb in neighbors(x, twin11):
            add(pool, nb)


def pool_fluid_augmented(
    nums: list[int],
    pos_weights: tuple[int, ...] = (1, 1, 1, 1, 1),
) -> set[int]:
    """cross_opt_v1 + ricette sul profilo di flusso (velocità, accel, cumulo)."""
    pool = pool_cross_opt_v1(nums)
    flow = quintina_flow(nums)
    for v in flow.velocities:
        expand_flow_value(pool, abs(v))
        expand_flow_value(pool, mod90(v))
        if v != 0:
            expand_flow_value(pool, mod90(90 - abs(v)))
    for a in flow.accelerations:
        expand_flow_value(pool, abs(a))
        expand_flow_value(pool, mod90(a))
    for cf in flow.cumulative_flux_mod90:
        add(pool, cf)
        add(pool, comp(cf))
    add(pool, flow.flow_sum_mod90)
    add(pool, comp(flow.flow_sum_mod90))
    # incrocio flusso × numero sorgente con peso posizionale
    for k, n in enumerate(nums):
        pw = pos_weights[k]
        if pw <= 0:
            continue
        for v in flow.velocities:
            add(pool, mod90(n + v))
            add(pool, mod90(abs(n - v)))
            cn = comp(n)
            if cn is not None:
                add(pool, mod90(cn + v))
                if pw >= 3:
                    add(pool, mod90(cn + v * 2))
    return pool


def score_position_weighted(
    nums: list[int],
    pos_weights: tuple[int, ...] = (1, 1, 1, 1, 1),
) -> dict[int, int]:
    """Come score_cross_numbers ma ogni ricetta da pos k scala con pos_weights[k]."""
    scores: defaultdict[int, int] = defaultdict(int)

    def bump(n: int | None, w: int) -> None:
        if n is not None and 1 <= n <= 90 and w > 0:
            scores[n] += w

    diffs = adj_diffs(nums)
    for k, n in enumerate(nums):
        pw = pos_weights[k]
        cn = comp(n)
        for d in diffs:
            bump(n + d, 2 * pw)
            bump(abs(n - d), 1 * pw)
            if cn is not None:
                bump(cn + d, 4 * pw)
                bump(abs(cn - d), 2 * pw)
        for base in filter(None, [n, cn, vertibile(n)]):
            for x in neighbors(base, True):
                bump(x, (2 if is_twin(base) else 1) * pw)
    if 45 in nums:
        bump(45, 3)
    if 90 in nums:
        bump(9, 2)

    flow = quintina_flow(nums)
    for v in flow.velocities:
        bump(abs(v), 2)
        bump(mod90(v), 1)
        vd = vertibile(abs(v))
        if vd:
            bump(vd, 3)
    for a in flow.accelerations:
        bump(abs(a), 2)
    bump(flow.flow_sum_mod90, 3)
    return dict(scores)


def pool_scored_fluid(
    nums: list[int],
    pos_weights: tuple[int, ...],
    top_k: int = 35,
) -> set[int]:
    pool = pool_fluid_augmented(nums)
    scores = score_position_weighted(nums, pos_weights)
    ranked = sorted(
        ((n, scores.get(n, 0)) for n in pool),
        key=lambda x: (-x[1], x[0]),
    )
    return {n for n, _ in ranked[:top_k]}


def eval_topk_hits(
    pairs: list[tuple[list[int], list[int]]],
    pool_fn: Callable[[list[int]], set[int]],
    score_fn: Callable[[list[int]], dict[int, int]],
    top_k: int,
) -> int:
    """Quanti dei 5 numeri uscenti cadono nel top-K scored del pool."""
    hits = 0
    for prev, nxt in pairs:
        pool = pool_fn(prev)
        scores = score_fn(prev)
        ranked = sorted(
            ((n, scores.get(n, 0)) for n in pool),
            key=lambda x: (-x[1], x[0]),
        )
        top = {n for n, _ in ranked[:top_k]}
        hits += sum(1 for x in nxt if x in top)
    return hits


def eval_position_solo_hits(
    pairs: list[tuple[list[int], list[int]]],
) -> list[dict]:
    """Hit rate se usi SOLO il numero a pos k della quintina precedente come pool."""
    rows = []
    for k in range(5):
        hits = 0
        for prev, nxt in pairs:
            anchor = prev[k]
            pool = {anchor}
            c = comp(anchor)
            if c:
                pool.add(c)
            v = vertibile(anchor)
            if v:
                pool.add(v)
            hits += sum(1 for x in nxt if x in pool)
        rows.append(
            {
                "position": k + 1,
                "hits_of_5": round(hits / len(pairs), 4) if pairs else 0,
                "total_hits": hits,
            }
        )
    return rows


def score_cross_position_weighted(
    nums: list[int],
    pos_weights: tuple[int, ...] = (1, 1, 1, 1, 1),
) -> dict[int, int]:
    """Score cross puro con pesi per posizione sorgente (senza ricette flusso)."""
    scores: defaultdict[int, int] = defaultdict(int)

    def bump(n: int | None, w: int) -> None:
        if n is not None and 1 <= n <= 90 and w > 0:
            scores[n] += w

    diffs = adj_diffs(nums)
    for k, n in enumerate(nums):
        pw = pos_weights[k]
        cn = comp(n)
        for d in diffs:
            bump(n + d, 2 * pw)
            bump(abs(n - d), 1 * pw)
            if cn is not None:
                bump(cn + d, 4 * pw)
                bump(abs(cn - d), 2 * pw)
        for base in filter(None, [n, cn, vertibile(n)]):
            for x in neighbors(base, True):
                bump(x, (2 if is_twin(base) else 1) * pw)
    if 45 in nums:
        bump(45, 3)
    if 90 in nums:
        bump(9, 2)
    return dict(scores)


def pool_cross_position_weighted(
    nums: list[int],
    pos_weights: tuple[int, ...],
) -> set[int]:
    """cross_opt_v1 + riordino implicito: stesso pool, numeri 'pesati' per posizione."""
    pool = pool_cross_opt_v1(nums)
    scores = score_cross_position_weighted(nums, pos_weights)
    # Tieni numeri con score>0 nel pool base; aggiungi top bonus da gap asimmetrici
    flow = quintina_flow(nums)
    for v in flow.velocities:
        vd = vertibile(abs(v))
        if vd and scores.get(vd, 0) >= 3:
            add(pool, vd)
    return pool


def grid_position_weights(
    pairs_train: list[tuple[list[int], list[int]]],
    rule_fn: Callable[[list[int], tuple[int, ...]], set[int]],
    weight_cap: int = 6,
) -> tuple[tuple[int, ...], int]:
    best_w = (1, 1, 1, 1, 1)
    best_hits = -1
    for w in POSITION_WEIGHT_PRESETS.values():
        er = eval_rule(pairs_train, lambda n, ww=w: rule_fn(n, ww), "preset")
        if er.hits > best_hits:
            best_hits = er.hits
            best_w = w
    for k in range(5):
        for mult in (2, 3, 4):
            w = list(best_w)
            w[k] = min(weight_cap, w[k] * mult)
            ww = tuple(w)
            er = eval_rule(
                pairs_train,
                lambda n, www=ww: rule_fn(n, www),
                f"local_pos{k + 1}_x{mult}",
            )
            if er.hits > best_hits:
                best_hits = er.hits
                best_w = ww
    return best_w, best_hits


@dataclass
class WheelFluidReport:
    wheel: str
    n_train: int
    n_test: int
    position_marginals: list[dict]
    velocity_gaps: list[dict]
    ordering_shape: dict
    inter_draw_coupling_top5: list[dict]
    position_solo_hits: list[dict]
    best_weights_cross: tuple[int, ...]
    best_weights_fluid: tuple[int, ...]
    test_hits_cross_v1: int
    test_hits_cross_weighted: int
    test_hits_fluid_augmented: int
    test_hits_fluid_weighted: int
    test_top8_cross: int
    test_top8_cross_weighted: int
    test_top8_fluid_weighted: int
    test_mc_mean_cross: float
    test_mc_p95_cross: float
    test_mc_mean_fluid: float
    test_mc_p95_fluid: float
    flow_profile_recent: dict


def analyze_wheel(
    wheel: str,
    pairs_train: list[tuple[list[int], list[int]]],
    pairs_test: list[tuple[list[int], list[int]]],
    *,
    top_k: int,
    mc_reps: int,
    seed: int,
) -> WheelFluidReport:
    train_draws = [p[0] for p in pairs_train]
    test_draws = [p[0] for p in pairs_test]
    all_draws = train_draws + test_draws

    marginals = position_marginals(all_draws)
    velocity = velocity_gap_stats(all_draws)
    shape = ordering_shape_stats(all_draws)
    coupling = inter_draw_position_coupling(pairs_test)
    coupling_top = sorted(coupling, key=lambda x: -x["lift_vs_uniform"])[:5]
    solo = eval_position_solo_hits(pairs_test)

    best_w_cross, _ = grid_position_weights(
        pairs_train, pool_cross_position_weighted
    )
    best_w_fluid, _ = grid_position_weights(pairs_train, pool_fluid_augmented)

    er_cross = eval_rule(pairs_test, pool_cross_opt_v1, "cross_v1")
    er_cross_w = eval_rule(
        pairs_test,
        lambda n, w=best_w_cross: pool_cross_position_weighted(n, w),
        "cross_weighted",
    )
    er_fluid = eval_rule(pairs_test, pool_fluid_augmented, "fluid_augmented")
    er_fluid_w = eval_rule(
        pairs_test,
        lambda n, w=best_w_fluid: pool_fluid_augmented(n, w),
        "fluid_weighted",
    )

    _, mc_p95_cross, _ = monte_carlo_baseline(
        pairs_test, er_cross.pool_sizes, reps=mc_reps, seed=seed
    )
    _, mc_p95_fluid, _ = monte_carlo_baseline(
        pairs_test, er_fluid.pool_sizes, reps=mc_reps, seed=seed + 1
    )

    top8_cross = eval_topk_hits(
        pairs_test, pool_cross_opt_v1, score_cross_numbers, 8
    )
    top8_cross_w = eval_topk_hits(
        pairs_test,
        lambda n, w=best_w_cross: pool_cross_position_weighted(n, w),
        lambda n, w=best_w_cross: score_cross_position_weighted(n, w),
        8,
    )
    top8_fluid_w = eval_topk_hits(
        pairs_test,
        lambda n, w=best_w_fluid: pool_fluid_augmented(n, w),
        lambda n, w=best_w_fluid: score_position_weighted(n, w),
        8,
    )

    recent_flow = quintina_flow(test_draws[-1]) if test_draws else quintina_flow([1, 2, 3, 4, 5])

    return WheelFluidReport(
        wheel=wheel,
        n_train=len(pairs_train),
        n_test=len(pairs_test),
        position_marginals=marginals,
        velocity_gaps=velocity,
        ordering_shape=shape,
        inter_draw_coupling_top5=coupling_top,
        position_solo_hits=solo,
        best_weights_cross=best_w_cross,
        best_weights_fluid=best_w_fluid,
        test_hits_cross_v1=er_cross.hits,
        test_hits_cross_weighted=er_cross_w.hits,
        test_hits_fluid_augmented=er_fluid.hits,
        test_hits_fluid_weighted=er_fluid_w.hits,
        test_top8_cross=top8_cross,
        test_top8_cross_weighted=top8_cross_w,
        test_top8_fluid_weighted=top8_fluid_w,
        test_mc_mean_cross=round(mc_mean_analytic(er_cross.pool_sizes), 2),
        test_mc_p95_cross=mc_p95_cross,
        test_mc_mean_fluid=round(mc_mean_analytic(er_fluid.pool_sizes), 2),
        test_mc_p95_fluid=mc_p95_fluid,
        flow_profile_recent=asdict(recent_flow),
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    fn = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fluidodinamica ordine estrazione quintina")
    ap.add_argument("--wheel", default=None, help="Singola ruota (default: tutte)")
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--top-k", type=int, default=35)
    ap.add_argument("--mc-reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    wide = root / "data" / "draws_wide.csv"
    if not wide.is_file():
        print("Manca data/draws_wide.csv", file=sys.stderr)
        return 1

    out_dir = root / "data" / "analysis" / "fluid_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)

    wheels = [args.wheel.upper()] if args.wheel else ALL_WHEELS
    reports: list[WheelFluidReport] = []

    for wheel in wheels:
        rows = load_wheel(wide, wheel)
        need = args.train + args.test + 1
        if len(rows) < need:
            print(f"Skip {wheel}: servono {need} estrazioni, trovate {len(rows)}")
            continue
        chunk = rows[-need:]
        pairs = [(chunk[i]["nums"], chunk[i + 1]["nums"]) for i in range(len(chunk) - 1)]
        pairs_train = pairs[: args.train]
        pairs_test = pairs[args.train : args.train + args.test]
        rep = analyze_wheel(
            wheel,
            pairs_train,
            pairs_test,
            top_k=args.top_k,
            mc_reps=args.mc_reps,
            seed=args.seed,
        )
        reports.append(rep)
        print(
            f"{wheel}: cross={rep.test_hits_cross_v1} "
            f"cross_w={rep.test_hits_cross_weighted} "
            f"fluid={rep.test_hits_fluid_augmented} "
            f"weights={rep.best_weights_cross} mc_p95={rep.test_mc_p95_cross}"
        )

    if not reports:
        print("Nessuna ruota analizzata.", file=sys.stderr)
        return 1

    # CSV riepilogo
    summary_rows = []
    for r in reports:
        cross_beats = r.test_hits_cross_v1 > r.test_mc_p95_cross
        fluid_beats = r.test_hits_fluid_augmented > r.test_mc_p95_fluid
        fluid_vs_cross = r.test_hits_fluid_augmented - r.test_hits_cross_v1
        cross_w_vs_cross = r.test_hits_cross_weighted - r.test_hits_cross_v1
        summary_rows.append(
            {
                "wheel": r.wheel,
                "best_weights_cross": str(r.best_weights_cross),
                "test_hits_cross_v1": r.test_hits_cross_v1,
                "test_hits_cross_weighted": r.test_hits_cross_weighted,
                "test_hits_fluid_augmented": r.test_hits_fluid_augmented,
                "fluid_minus_cross": fluid_vs_cross,
                "cross_weighted_minus_cross": cross_w_vs_cross,
                "test_top8_cross": r.test_top8_cross,
                "test_top8_cross_weighted": r.test_top8_cross_weighted,
                "test_top8_fluid_weighted": r.test_top8_fluid_weighted,
                "top8_fluid_minus_cross": r.test_top8_fluid_weighted - r.test_top8_cross,
                "test_mc_p95_cross": r.test_mc_p95_cross,
                "cross_beats_mc_p95": cross_beats,
                "fluid_beats_mc_p95": fluid_beats,
                "strongest_solo_pos": max(
                    r.position_solo_hits, key=lambda x: x["hits_of_5"]
                )["position"],
            }
        )
    write_csv(out_dir / "wheel_summary.csv", summary_rows)

    # Accoppiamenti migliori aggregati
    all_coupling: list[dict] = []
    for r in reports:
        for c in r.inter_draw_coupling_top5:
            all_coupling.append({"wheel": r.wheel, **c})
    write_csv(out_dir / "top_coupling_by_wheel.csv", all_coupling)

    # Gap velocity aggregate (media su ruote)
    gap_acc: dict[str, list[float]] = defaultdict(list)
    for r in reports:
        for g in r.velocity_gaps:
            gap_acc[g["gap"]].append(g["mean_signed"])
    gap_asymmetry = [
        {
            "gap": gap,
            "mean_signed_across_wheels": round(statistics.mean(vals), 4),
            "wheels": len(vals),
        }
        for gap, vals in sorted(gap_acc.items())
    ]

    # Posizione con entropia più bassa (più "concentrata")
    pos_entropy: dict[int, list[float]] = defaultdict(list)
    for r in reports:
        for m in r.position_marginals:
            pos_entropy[m["position"]].append(m["entropy_ratio"])
    pos_entropy_mean = [
        {
            "position": p,
            "mean_entropy_ratio": round(statistics.mean(vs), 6),
        }
        for p, vs in sorted(pos_entropy.items())
    ]

    payload = {
        "method": "extraction_fluid_dynamics",
        "train_transitions": args.train,
        "test_transitions": args.test,
        "top_k_pool": args.top_k,
        "interpretation": (
            "La quintina è trattata come flusso ordinato pos1→pos5. "
            "Velocità SIGNED v_k=n_{k+1}-n_k (non adj_diffs assoluto del metodo cross). "
            "adj_diffs del cross ignora la direzione — qui si vede il verso del flusso. "
            "Pesi posizionali ottimizzati su train per score cross asimmetrico. "
            "Lift accoppiamento >1 = stesso numero da pos i a pos j più frequente del caso."
        ),
        "gap_velocity_asymmetry": gap_asymmetry,
        "position_entropy_mean": pos_entropy_mean,
        "wheels": [asdict(r) for r in reports],
        "aggregate": {
            "wheels_analyzed": len(reports),
            "fluid_beats_cross_count": sum(
                1 for s in summary_rows if s["fluid_minus_cross"] > 0
            ),
            "cross_weighted_beats_cross_count": sum(
                1 for s in summary_rows if s["cross_weighted_minus_cross"] > 0
            ),
            "fluid_beats_mc_p95_count": sum(
                1 for s in summary_rows if s["fluid_beats_mc_p95"]
            ),
            "cross_beats_mc_p95_count": sum(
                1 for s in summary_rows if s["cross_beats_mc_p95"]
            ),
            "best_fluid_wheel": max(
                summary_rows, key=lambda x: x["fluid_minus_cross"]
            ),
            "worst_fluid_wheel": min(
                summary_rows, key=lambda x: x["fluid_minus_cross"]
            ),
        },
    }

    json_path = out_dir / "fluid_dynamics_report.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Fluidodinamica estrazione — ordine pos1..pos5",
        "",
        "## Idea",
        "",
        "Nessuno guardava la quintina come **tubo di flusso**: il peso del numero dipende da *dove*",
        "esce nell'ordine, non solo dal valore. Il metodo cross usa `adj_diffs` = |n_{k+1}-n_k|",
        "(ordine cieco); qui v_k = n_{k+1}-n_k rivela direzione e accelerazione del flusso.",
        "",
        "## Asimmetria gap (media signed velocity per ruota)",
        "",
        "| gap | mean_signed |",
        "|-----|-------------|",
    ]
    for g in gap_asymmetry:
        md_lines.append(f"| {g['gap']} | {g['mean_signed_across_wheels']} |")
    md_lines.extend(
        [
            "",
            "## Entropia posizione (media ratio su ruote)",
            "",
            "| pos | entropy_ratio |",
            "|-----|---------------|",
        ]
    )
    for p in pos_entropy_mean:
        md_lines.append(f"| {p['position']} | {p['mean_entropy_ratio']} |")
    md_lines.extend(
        [
            "",
            "## Test 500 transizioni — pool pieno (confronto equo)",
            "",
            "### Pool pieno (hit su 5 numeri)",
            "",
            "| ruota | pesi | cross | fluid | Δ fluid |",
            "|-------|------|-------|-------|---------|",
        ]
    )
    for s in summary_rows:
        md_lines.append(
            f"| {s['wheel']} | {s['best_weights_cross']} | {s['test_hits_cross_v1']} | "
            f"{s['test_hits_fluid_augmented']} | {s['fluid_minus_cross']:+d} |"
        )
    md_lines.extend(
        [
            "",
            "### Top-8 scored (rilevante per ambo)",
            "",
            "| ruota | cross top8 | cross_w top8 | fluid top8 | Δ fluid |",
            "|-------|------------|--------------|------------|---------|",
        ]
    )
    for s in summary_rows:
        md_lines.append(
            f"| {s['wheel']} | {s['test_top8_cross']} | {s['test_top8_cross_weighted']} | "
            f"{s['test_top8_fluid_weighted']} | {s['top8_fluid_minus_cross']:+d} |"
        )
    md_lines.append("")
    md_lines.append(f"Report JSON: `{json_path.name}`")

    (out_dir / "fluid_dynamics_report.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

    print(f"\nScritto in {out_dir}")
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
