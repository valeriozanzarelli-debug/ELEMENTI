#!/usr/bin/env python3
"""
Test regole derivate (complementi, vertibili, diff, gemelli ±11, speciali 45/90)
vs baseline Monte Carlo a parità di dimensione pool — BARI, transizioni giorno T -> T+1.
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def vertibile(n: int) -> int | None:
    if n < 10 or n > 90:
        return None
    s = str(n)
    if len(s) != 2:
        return None
    rev = int(s[::-1])
    if rev == n or rev < 1 or rev > 90:
        return None
    return rev


def is_twin(n: int) -> bool:
    return 10 <= n <= 88 and n % 11 == 0


def is_special_no_comp(n: int) -> bool:
    """45 è auto-complementare; 90 non ha complemento valido in 1..90."""
    return n in (45, 90)


def comp(n: int) -> int | None:
    if is_special_no_comp(n):
        return None
    c = 90 - n
    return c if 1 <= c <= 90 else None


def adj_diffs(nums: list[int]) -> list[int]:
    return [max(nums[i], nums[i + 1]) - min(nums[i], nums[i + 1]) for i in range(4)]


def neighbors(n: int, use_twin_step: bool) -> list[int]:
    if use_twin_step and is_twin(n):
        step = 11
    else:
        step = 1
    out = []
    for d in (-step, step):
        x = n + d
        if 1 <= x <= 90:
            out.append(x)
    return out


def add(pool: set[int], n: int | None) -> None:
    if n is not None and 1 <= n <= 90:
        pool.add(n)


def expand_num(pool: set[int], n: int, *, twin11: bool, with_neighbors: bool) -> None:
    add(pool, n)
    c = comp(n)
    add(pool, c)
    if is_special_no_comp(n):
        if n == 45:
            add(pool, 45)  # enfatizza auto-complemento
        if n == 90:
            add(pool, 9)  # radice / cifra ridotta
            add(pool, 10)  # 90-80, vicini simbolici
    v = vertibile(n)
    if v:
        add(pool, v)
        add(pool, comp(v))
    if with_neighbors:
        for x in neighbors(n, twin11):
            add(pool, x)
        if c is not None:
            for x in neighbors(c, twin11):
                add(pool, x)
        if v:
            for x in neighbors(v, twin11):
                add(pool, x)


def pool_user_full(nums: list[int], *, twin11: bool = False) -> set[int]:
    """Regola utente estesa: orig, comp, vert, diff, incroci."""
    pool: set[int] = set()
    for n in nums:
        expand_num(pool, n, twin11=twin11, with_neighbors=True)
    diffs = adj_diffs(nums)
    for d in diffs:
        expand_num(pool, d, twin11=twin11, with_neighbors=True)
    for i, n in enumerate(nums):
        cn = comp(n)
        for d in diffs:
            add(pool, n + d)
            add(pool, abs(n - d))
            if cn is not None:
                add(pool, cn + d)
                add(pool, abs(cn - d))
            vd = vertibile(d)
            if vd:
                add(pool, vd + 1)
                add(pool, vd - 1)
                if twin11 and is_twin(vd):
                    add(pool, vd + 11)
                    add(pool, vd - 11)
    return pool


def pool_user_tight(nums: list[int], *, twin11: bool = False) -> set[int]:
    """Solo complementi, vertibili, diff e ±1/±11 — niente incroci orig+diff."""
    pool: set[int] = set()
    for n in nums:
        expand_num(pool, n, twin11=twin11, with_neighbors=True)
    for d in adj_diffs(nums):
        expand_num(pool, d, twin11=twin11, with_neighbors=True)
    return pool


def pool_cross_only(nums: list[int], *, twin11: bool = False) -> set[int]:
    """Solo incroci comp+diff e orig+diff con vicini gemello."""
    pool: set[int] = set()
    diffs = adj_diffs(nums)
    for n in nums:
        cn = comp(n)
        for d in diffs:
            add(pool, n + d)
            add(pool, abs(n - d))
            if cn is not None:
                add(pool, cn + d)
                add(pool, abs(cn - d))
            for base in filter(None, [n, cn, vertibile(n)]):
                for x in neighbors(base, twin11):
                    add(pool, x)
    return pool


def pool_diff_vert_chain(nums: list[int], *, twin11: bool = False) -> set[int]:
    """Diff consecutive + vertibili delle diff + complementi diff."""
    pool: set[int] = set()
    diffs = adj_diffs(nums)
    for d in diffs:
        expand_num(pool, d, twin11=twin11, with_neighbors=True)
    # somma cumulativa delle diff (catena)
    s = 0
    for d in diffs:
        s += d
        expand_num(pool, s, twin11=twin11, with_neighbors=False)
        add(pool, comp(s))
    return pool


def pool_positional_shift(nums: list[int], *, twin11: bool = False) -> set[int]:
    """Per ogni posizione k, predici da nums[k] e diff(k,k+1) se esiste."""
    pool: set[int] = set()
    diffs = adj_diffs(nums)
    for k, n in enumerate(nums):
        expand_num(pool, n, twin11=twin11, with_neighbors=True)
        if k < len(diffs):
            d = diffs[k]
            add(pool, n + d)
            add(pool, abs(n - d))
            cn = comp(n)
            if cn is not None:
                add(pool, cn + d)
        if k > 0:
            d = diffs[k - 1]
            vd = vertibile(d)
            if vd:
                for x in neighbors(vd, twin11):
                    add(pool, x)
    return pool


def pool_special_4590(nums: list[int], *, twin11: bool = True) -> set[int]:
    """Enfasi su 45/90 e gemelli: se c'è un gemello, espandi ±11 su tutta la quintina."""
    pool: set[int] = set()
    has_twin = any(is_twin(n) for n in nums)
    has_45 = 45 in nums
    has_90 = 90 in nums
    for n in nums:
        expand_num(pool, n, twin11=twin11, with_neighbors=True)
    diffs = adj_diffs(nums)
    for d in diffs:
        expand_num(pool, d, twin11=twin11, with_neighbors=True)
    if has_twin:
        for n in nums:
            if is_twin(n):
                for m in nums:
                    add(pool, m + 11)
                    add(pool, m - 11)
                    add(pool, m + 22)
                    add(pool, m - 22)
    if has_45:
        for n in nums:
            add(pool, 45)
            add(pool, n + 45)
            add(pool, abs(n - 45))
    if has_90:
        for n in nums:
            add(pool, 9)
            add(pool, 90 - n if n != 90 else 9)
            add(pool, (n + 9) % 90 or 90)
    for i, n in enumerate(nums):
        cn = comp(n)
        for d in diffs:
            if cn is not None:
                add(pool, cn + d)
            add(pool, n + d)
    return pool


def pool_hybrid_best(nums: list[int]) -> set[int]:
    """Mix: tight + cross solo per vertibili diff + speciali."""
    a = pool_user_tight(nums, twin11=True)
    b = pool_cross_only(nums, twin11=True)
    c = pool_special_4590(nums, twin11=True)
    return a | b | c


def pool_minimal_comp_vert(nums: list[int], *, twin11: bool = True) -> set[int]:
    """Solo complementi e vertibili degli originali (+ vicini)."""
    pool: set[int] = set()
    for n in nums:
        expand_num(pool, n, twin11=twin11, with_neighbors=False)
        c = comp(n)
        add(pool, c)
        v = vertibile(n)
        if v:
            add(pool, v)
            add(pool, comp(v))
        for x in neighbors(n, twin11):
            add(pool, x)
    return pool


def pool_vert_diff_only(nums: list[int], *, twin11: bool = True) -> set[int]:
    """Solo diff vertibili + complementi diff + vicini gemello."""
    pool: set[int] = set()
    for d in adj_diffs(nums):
        if vertibile(d):
            expand_num(pool, d, twin11=twin11, with_neighbors=True)
        else:
            add(pool, d)
            add(pool, comp(d))
    return pool


def pool_scored_topk(nums: list[int], k: int = 35, *, twin11: bool = True) -> set[int]:
    """Conta quante ricette generano ogni numero; tieni i top-K."""
    scores: defaultdict[int, int] = defaultdict(int)

    def bump(n: int | None, w: int = 1) -> None:
        if n is not None and 1 <= n <= 90:
            scores[n] += w

    diffs = adj_diffs(nums)
    for n in nums:
        bump(n, 3)
        bump(comp(n), 3)
        bump(vertibile(n), 2)
        for x in neighbors(n, twin11):
            bump(x, 1)
    for d in diffs:
        bump(d, 2)
        bump(comp(d), 2)
        bump(vertibile(d), 3)
        for x in neighbors(d, twin11):
            bump(x, 1)
    for i, n in enumerate(nums):
        cn = comp(n)
        for d in diffs:
            bump(n + d, 2)
            bump(abs(n - d), 1)
            if cn is not None:
                bump(cn + d, 3)
                bump(abs(cn - d), 1)
    if 45 in nums:
        for n in nums:
            bump(45, 2)
            bump(abs(n - 45), 1)
    if 90 in nums:
        bump(9, 2)
        bump(10, 1)
    for n in nums:
        if is_twin(n):
            for m in nums:
                bump(m + 11, 2)
                bump(m - 11, 2)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return {n for n, _ in ranked[:k]}


def pool_intersection(nums: list[int]) -> set[int]:
    """Numeri che compaiono in almeno 2 famiglie indipendenti."""
    a = pool_minimal_comp_vert(nums, twin11=True)
    b = pool_vert_diff_only(nums, twin11=True)
    c = pool_cross_only(nums, twin11=True)
    counts: defaultdict[int, int] = defaultdict(int)
    for s in (a, b, c):
        for n in s:
            counts[n] += 1
    return {n for n, c in counts.items() if c >= 2}


def pool_twin_trigger(nums: list[int]) -> set[int]:
    """Espansione ±11 solo se nella quintina c'è un gemello."""
    twin11 = any(is_twin(n) for n in nums)
    if twin11:
        return pool_user_tight(nums, twin11=True)
    return pool_user_tight(nums, twin11=False)


def pool_cross_opt_v1(nums: list[int]) -> set[int]:
    """cross_twin11 + complementi/vertibili originali + vicini vert(diff)."""
    pool = pool_cross_only(nums, twin11=True)
    for n in nums:
        add(pool, comp(n))
        add(pool, vertibile(n))
    for d in adj_diffs(nums):
        vd = vertibile(d)
        if vd:
            for x in neighbors(vd, True):
                add(pool, x)
    return pool


def pool_cross_opt_v2(nums: list[int]) -> set[int]:
    """Ottimizzato su train 500x10: v1 senza vicini vert(diff) — pool piu stretto."""
    pool = pool_cross_only(nums, twin11=True)
    for n in nums:
        add(pool, comp(n))
        add(pool, vertibile(n))
    return pool


def pool_comp_plus_best_diff(nums: list[int], *, twin11: bool = True) -> set[int]:
    """Complementi originali + la diff vertibile più grande + incroci comp+diff."""
    pool: set[int] = set()
    for n in nums:
        expand_num(pool, n, twin11=twin11, with_neighbors=False)
        add(pool, comp(n))
        v = vertibile(n)
        if v:
            add(pool, v)
    diffs = adj_diffs(nums)
    if diffs:
        d = max(diffs, key=lambda x: (vertibile(x) is not None, x))
        expand_num(pool, d, twin11=twin11, with_neighbors=True)
        for n in nums:
            cn = comp(n)
            if cn is not None:
                add(pool, cn + d)
    return pool


RULES: dict[str, Callable[[list[int]], set[int]]] = {
    "user_full": lambda n: pool_user_full(n, twin11=False),
    "user_full_twin11": lambda n: pool_user_full(n, twin11=True),
    "user_tight": lambda n: pool_user_tight(n, twin11=False),
    "user_tight_twin11": lambda n: pool_user_tight(n, twin11=True),
    "cross_only": lambda n: pool_cross_only(n, twin11=False),
    "cross_twin11": lambda n: pool_cross_only(n, twin11=True),
    "diff_chain": lambda n: pool_diff_vert_chain(n, twin11=True),
    "positional": lambda n: pool_positional_shift(n, twin11=True),
    "special_4590": pool_special_4590,
    "hybrid": pool_hybrid_best,
    "minimal_comp_vert": pool_minimal_comp_vert,
    "vert_diff_only": pool_vert_diff_only,
    "scored_top35": lambda n: pool_scored_topk(n, 35),
    "scored_top25": lambda n: pool_scored_topk(n, 25),
    "scored_top19": lambda n: pool_scored_topk(n, 19),
    "scored_top20": lambda n: pool_scored_topk(n, 20),
    "intersection_2of3": pool_intersection,
    "twin_trigger": pool_twin_trigger,
    "comp_best_diff": pool_comp_plus_best_diff,
    "cross_opt_v1": pool_cross_opt_v1,
    "cross_opt_v2": pool_cross_opt_v2,
}


@dataclass
class EvalResult:
    name: str
    hits: int
    total: int
    avg_pool: float
    per_draw: list[int] = field(default_factory=list)
    pool_sizes: list[int] = field(default_factory=list)


def load_wheel(wide: Path, wheel: str) -> list[dict]:
    cols = [f"{wheel}_{k}" for k in range(1, 6)]
    rows = []
    with wide.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if all(row[c] for c in cols):
                rows.append(
                    {
                        "date": row["date"],
                        "nums": [int(row[c]) for c in cols],
                    }
                )
    return rows


def eval_rule(
    pairs: list[tuple[list[int], list[int]]],
    rule_fn: Callable[[list[int]], set[int]],
    name: str,
) -> EvalResult:
    hits = 0
    per_draw: list[int] = []
    pool_sizes: list[int] = []
    for prev, nxt in pairs:
        pool = rule_fn(prev)
        pool_sizes.append(len(pool))
        h = sum(1 for t in nxt if t in pool)
        hits += h
        per_draw.append(h)
    total = len(pairs) * 5
    return EvalResult(
        name=name,
        hits=hits,
        total=total,
        avg_pool=statistics.mean(pool_sizes) if pool_sizes else 0,
        per_draw=per_draw,
        pool_sizes=pool_sizes,
    )


def monte_carlo_baseline(
    pairs: list[tuple[list[int], list[int]]],
    pool_sizes: list[int],
    *,
    reps: int,
    seed: int,
) -> tuple[float, float, list[int]]:
    rng = random.Random(seed)
    totals: list[int] = []
    for _ in range(reps):
        h = 0
        for i, (prev, nxt) in enumerate(pairs):
            sz = pool_sizes[i]
            pool = set(rng.sample(range(1, 91), sz))
            h += sum(1 for t in nxt if t in pool)
        totals.append(h)
    totals.sort()
    mean = statistics.mean(totals)
    p95 = totals[int(0.95 * reps)]
    return mean, p95, totals


def mc_mean_analytic(pool_sizes: list[int]) -> float:
    """E[hit] con pool uniforme casuale di stessa dimensione: 5 * |pool| / 90 per draw."""
    return sum(5 * sz / 90 for sz in pool_sizes)


def search_combos(
    pairs: list[tuple[list[int], list[int]]],
    seed: int,
    mc_reps: int,
) -> list[dict]:
    """Valuta tutte le regole vs MC."""
    results = []
    for name, fn in RULES.items():
        er = eval_rule(pairs, fn, name)
        mc_mean = mc_mean_analytic(er.pool_sizes)
        _, mc_p95, _ = monte_carlo_baseline(
            pairs, er.pool_sizes, reps=mc_reps, seed=seed
        )
        results.append(
            {
                "name": name,
                "hits": er.hits,
                "avg_pool": er.avg_pool,
                "mc_mean": mc_mean,
                "mc_p95": mc_p95,
                "beats_mc": er.hits > mc_mean,
                "beats_p95": er.hits > mc_p95,
                "margin": er.hits - mc_mean,
            }
        )
    return sorted(results, key=lambda x: (-x["margin"], -x["hits"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", default="BARI")
    ap.add_argument("--transitions", type=int, default=20)
    ap.add_argument("--mc-reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--grid-topk",
        action="store_true",
        help="Cerca miglior K per pool_scored_topk",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    wide = root / "data" / "draws_wide.csv"
    rows = load_wheel(wide, args.wheel)
    need = args.transitions + 1
    if len(rows) < need:
        raise SystemExit(f"Servono almeno {need} estrazioni, trovate {len(rows)}")

    chunk = rows[-need:]
    pairs = [(chunk[i]["nums"], chunk[i + 1]["nums"]) for i in range(args.transitions)]
    date_from = chunk[0]["date"]
    date_to = chunk[-1]["date"]

    print(f"Ruota: {args.wheel} | transizioni: {args.transitions}")
    print(f"Periodo: {date_from} -> {date_to}")
    print()

    if args.grid_topk:
        print("Grid search pool_scored_topk:")
        best_k = None
        best_margin = -999.0
        for k in range(10, 41):
            fn = lambda nums, kk=k: pool_scored_topk(nums, kk)
            er = eval_rule(pairs, fn, f"scored_top{k}")
            mc = mc_mean_analytic(er.pool_sizes)
            margin = er.hits - mc
            flag = " *" if margin > 0 else ""
            print(
                f"  k={k:2d} hits={er.hits:4d} MC={mc:6.1f} margin={margin:+5.1f} pool={er.avg_pool:4.1f}{flag}"
            )
            if margin > best_margin:
                best_margin = margin
                best_k = k
        print(f"Miglior K: {best_k} (margin {best_margin:+.1f})\n")

    results = search_combos(pairs, args.seed, args.mc_reps)
    winners = [r for r in results if r["beats_mc"]]

    print(f"{'Metodo':<42} {'Hits':>5} {'Pool':>6} {'MCavg':>6} {'MCp95':>6} {'>MC':>4}")
    print("-" * 72)
    for r in results[:25]:
        flag = "YES" if r["beats_mc"] else "no"
        star = "*" if r["beats_p95"] else " "
        print(
            f"{star}{r['name']:<41} {r['hits']:>5} {r['avg_pool']:>6.1f} "
            f"{r['mc_mean']:>6.1f} {r['mc_p95']:>6.0f} {flag:>4}"
        )

    print()
    if winners:
        print(f"Metodi che battono MC (media): {len(winners)}")
        best = max(winners, key=lambda x: (x["hits"] - x["mc_mean"], -x["avg_pool"]))
        print(
            f"Migliore vs MC: {best['name']} -> {best['hits']} hit "
            f"(MC avg={best['mc_mean']:.1f}, margine +{best['hits']-best['mc_mean']:.1f})"
        )
        if best["beats_p95"]:
            print(f"  Batte anche il p95 Monte Carlo ({best['mc_p95']:.0f})")
        else:
            print(f"  Non batte p95 Monte Carlo ({best['mc_p95']:.0f}) — possibile fluttuazione")
    else:
        print("Nessun metodo batte la media Monte Carlo su questo campione.")

    # dettaglio ultima transizione col migliore
    if winners:
        best_name = best["name"]
        if best_name in RULES:
            fn = RULES[best_name]
        else:
            # parse union - skip detail
            fn = None
        if fn:
            prev, nxt = pairs[-1]
            pool = fn(prev)
            print()
            print(f"Ultima transizione ({chunk[-2]['date']} -> {chunk[-1]['date']}):")
            print(f"  Da: {prev}")
            print(f"  A:  {nxt}")
            for j, t in enumerate(nxt, 1):
                print(f"    pos{j} {t}: {'OK' if t in pool else 'MISS'}")
            print(f"  Pool size: {len(pool)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
