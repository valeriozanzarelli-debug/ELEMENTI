#!/usr/bin/env python3
"""
VALIDAZIONE DEFINITIVA — ultimi 5 anni, tutte le 11 ruote.

Risponde alla "domanda aperta" della teoria cross:
il pool derivato dalla quintina precedente ha potere predittivo reale,
oppure l'effetto e' (a) copertura del pool + (b) selezione a posteriori
delle celle fortunate?

Test eseguiti:
  A. POOL TEST  — ogni regola (tutte le ricette cross esistenti + regole
     "furbe" data-driven: Markov lift, hot, cold/ritardatari) confrontata
     con il null ESATTO ipergeometrico a parita' di dimensione pool.
     Z-score, p-value, intervallo di confidenza 95%, correzione Bonferroni.
  B. BETTING TEST — 1 EUR intero per sorte (top-1 estratto, top-2 ambo,
     top-3 terno...) su ogni ruota per 5 anni. Hit osservati vs binomiale
     esatta con p0 = probabilita' caso. P-value esatto per cella e globale.
  C. SELECTION BIAS TEST — griglia ruote x anni x sorti: distribuzione
     nulla del MIGLIOR risultato di cella con giocate casuali (Monte Carlo).
     Se il best osservato (es. TORINO 2025 ambo) sta dentro la distribuzione
     del massimo nullo, e' selezione a posteriori, non segnale.
  D. COVERAGE — quanti hit/estrazione spiega la sola copertura del pool
     (il "ritrova 3-4 numeri su 5" della teoria).

Tutte le regole "smart" sono addestrate SOLO su dati precedenti alla
finestra di test (out-of-sample rigoroso).

Uso:
  python scripts/five_year_validation.py
  python scripts/five_year_validation.py --years-back 5 --mc-reps 2000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_ambi_2025 import load_draws_by_date, score_cross_numbers, top_n_from_pool
from vertibile_rule_test import RULES, pool_cross_opt_v1, pool_user_tight

WHEELS = [
    "BARI", "CAGLIARI", "FIRENZE", "GENOVA", "MILANO", "NAPOLI",
    "PALERMO", "ROMA", "TORINO", "VENEZIA", "NAZIONALE",
]

PAYOUT = {
    "estratto": 11.23,
    "ambo": 250.0,
    "terno": 4500.0,
    "quaterna": 120_000.0,
    "cinquina": 6_000_000.0,
}
K_FOR = {"estratto": 1, "ambo": 2, "terno": 3, "quaterna": 4, "cinquina": 5}


# ----------------------------------------------------------------------------
# probabilita' esatte
# ----------------------------------------------------------------------------

def random_hit_prob(bet: str) -> float:
    """P(combinazione fissa di k numeri tutta tra i 5 estratti su 90)."""
    k = K_FOR[bet]
    return math.comb(5, k) / math.comb(90, k) if k <= 5 else 0.0


def hyper_mean_var(pool_size: int) -> tuple[float, float]:
    """Hit attesi e varianza per UNA estrazione: 5 palline da 90, K nel pool."""
    p = pool_size / 90.0
    mean = 5.0 * p
    var = 5.0 * p * (1.0 - p) * (90 - 5) / (90 - 1)
    return mean, var


def normal_sf(z: float) -> float:
    """P(Z > z) per normale standard."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def binom_sf(hits: int, n: int, p: float) -> float:
    """P(X >= hits), X ~ Binomial(n, p), esatta in log-spazio."""
    if hits <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    total = 0.0
    lp, lq = math.log(p), math.log1p(-p)
    for k in range(hits, n + 1):
        lt = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        lt += k * lp + (n - k) * lq
        if lt < -745:  # exp underflow
            if k > hits and total > 0:
                break
            continue
        total += math.exp(lt)
    return min(total, 1.0)


def wilson_ci(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervallo di confidenza 95% (Wilson) sull'hit rate."""
    if total == 0:
        return 0.0, 0.0
    ph = hits / total
    den = 1 + z * z / total
    cen = (ph + z * z / (2 * total)) / den
    rad = z * math.sqrt(ph * (1 - ph) / total + z * z / (4 * total * total)) / den
    return max(0.0, cen - rad), min(1.0, cen + rad)


# ----------------------------------------------------------------------------
# regole "smart" data-driven (addestrate out-of-sample)
# ----------------------------------------------------------------------------

class MarkovLift:
    """Lift di transizione a->b appreso su dati PRIMA della finestra di test.

    lift[a][b] = P(b tra i 5 di T+1 | a tra i 5 di T) / (5/90)
    Pool = top-K numeri per somma dei lift dei 5 numeri precedenti.
    """

    def __init__(self, train_pairs: list[tuple[list[int], list[int]]]):
        cnt = [[0] * 91 for _ in range(91)]
        occ = [0] * 91
        for prev, nxt in train_pairs:
            nxt_set = set(nxt)
            for a in prev:
                occ[a] += 1
                for b in nxt_set:
                    cnt[a][b] += 1
        base = 5.0 / 90.0
        self.lift = [[0.0] * 91 for _ in range(91)]
        for a in range(1, 91):
            if occ[a] == 0:
                continue
            for b in range(1, 91):
                self.lift[a][b] = (cnt[a][b] / occ[a]) / base

    def scores(self, prev: list[int]) -> dict[int, float]:
        sc: dict[int, float] = {}
        for b in range(1, 91):
            s = sum(self.lift[a][b] for a in prev)
            sc[b] = s
        return sc

    def pool(self, prev: list[int], k: int = 25) -> set[int]:
        sc = self.scores(prev)
        ranked = sorted(sc.items(), key=lambda x: (-x[1], x[0]))
        return {n for n, _ in ranked[:k]}

    def picks(self, prev: list[int], k: int) -> list[int]:
        sc = self.scores(prev)
        ranked = sorted(sc.items(), key=lambda x: (-x[1], x[0]))
        return [n for n, _ in ranked[:k]]


def hot_pool(history: list[list[int]], window: int = 30, k: int = 25) -> set[int]:
    """K numeri piu' frequenti nelle ultime `window` estrazioni della ruota."""
    cnt: defaultdict[int, int] = defaultdict(int)
    for draw in history[-window:]:
        for n in draw:
            cnt[n] += 1
    ranked = sorted(range(1, 91), key=lambda n: (-cnt[n], n))
    return set(ranked[:k])


def cold_pool(history: list[list[int]], k: int = 25) -> set[int]:
    """K numeri piu' ritardatari (ultima uscita piu' lontana)."""
    last_seen = {n: -1 for n in range(1, 91)}
    for i, draw in enumerate(history):
        for n in draw:
            last_seen[n] = i
    ranked = sorted(range(1, 91), key=lambda n: (last_seen[n], n))
    return set(ranked[:k])


# ----------------------------------------------------------------------------
# caricamento dati
# ----------------------------------------------------------------------------

def build_transitions(root: Path) -> tuple[dict[str, list[dict]], str, str]:
    """Per ruota: lista ordinata di {date, prev, nxt, history_idx}."""
    dates, by_date = load_draws_by_date(root / "data" / "draws_wide.csv")
    per_wheel: dict[str, list[dict]] = {w: [] for w in WHEELS}
    seq: dict[str, list[tuple[str, list[int]]]] = {w: [] for w in WHEELS}
    for d in dates:
        for w in WHEELS:
            if w in by_date.get(d, {}):
                seq[w].append((d, by_date[d][w]))
    for w in WHEELS:
        s = seq[w]
        for i in range(1, len(s)):
            per_wheel[w].append({
                "date": s[i][0],
                "prev": s[i - 1][1],
                "nxt": s[i][1],
                "idx": i,  # indice in seq[w], per history
            })
    return per_wheel, dates[0], dates[-1]


def shift_years(date_iso: str, years: int) -> str:
    y, rest = date_iso.split("-", 1)
    return f"{int(y) - years}-{rest}"


# ----------------------------------------------------------------------------
# TEST A — pool vs null ipergeometrico esatto
# ----------------------------------------------------------------------------

def eval_pool_rule(transitions: list[dict], pool_for: callable) -> dict:
    hits = 0
    exp = 0.0
    var = 0.0
    total = 0
    pool_sizes = []
    for t in transitions:
        pool = pool_for(t)
        sz = len(pool)
        pool_sizes.append(sz)
        m, v = hyper_mean_var(sz)
        exp += m
        var += v
        hits += sum(1 for n in t["nxt"] if n in pool)
        total += 5
    z = (hits - exp) / math.sqrt(var) if var > 0 else 0.0
    p_one_sided = normal_sf(z)
    lo, hi = wilson_ci(hits, total)
    exp_rate = exp / total if total else 0.0
    return {
        "var": var,
        "transitions": len(transitions),
        "avg_pool": round(statistics.mean(pool_sizes), 2) if pool_sizes else 0,
        "hits": hits,
        "expected_hits": round(exp, 1),
        "margin": round(hits - exp, 1),
        "z": round(z, 3),
        "p_one_sided": round(p_one_sided, 6),
        "hit_rate": round(hits / total, 5) if total else 0,
        "expected_rate": round(exp_rate, 5),
        "ci95_lo": round(lo, 5),
        "ci95_hi": round(hi, 5),
        "ci_contains_null": lo <= exp_rate <= hi,
    }


# ----------------------------------------------------------------------------
# TEST B — betting 1 EUR/sorte
# ----------------------------------------------------------------------------

def eval_betting(transitions: list[dict], picks_for: callable, year_filter: int | None = None) -> dict[str, dict]:
    """picks_for(t) -> lista top-5 ordinata. Ritorna risultati per sorte."""
    out: dict[str, dict] = {}
    counters = {b: [0, 0] for b in K_FOR}  # bet -> [draws, hits]
    for t in transitions:
        if year_filter is not None and not t["date"].startswith(str(year_filter)):
            continue
        picks = picks_for(t)
        if len(picks) < 5:
            continue
        drawn = set(t["nxt"])
        for bet, k in K_FOR.items():
            counters[bet][0] += 1
            if all(p in drawn for p in picks[:k]):
                counters[bet][1] += 1
    for bet, (n, h) in counters.items():
        p0 = random_hit_prob(bet)
        won = h * PAYOUT[bet]
        out[bet] = {
            "draws": n,
            "hits": h,
            "hit_rate": round(h / n, 5) if n else 0,
            "p0_random": round(p0, 6),
            "expected_hits_random": round(n * p0, 2),
            "p_value_binom": round(binom_sf(h, n, p0), 4) if n else 1.0,
            "spent_eur": n,
            "won_gross_eur": round(won, 2),
            "net_gross_eur": round(won - n, 2),
        }
    return out


# ----------------------------------------------------------------------------
# TEST C — selection bias: distribuzione nulla del best-of-grid
# ----------------------------------------------------------------------------

def binom_sampler(n: int, p: float) -> tuple[list[float], int]:
    """CDF troncata per campionare velocemente Binomial(n, p) con p piccolo."""
    cdf = []
    acc = 0.0
    lp, lq = math.log(p), math.log1p(-p)
    k = 0
    while acc < 1.0 - 1e-12 and k <= n:
        lt = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        lt += k * lp + (n - k) * lq
        acc += math.exp(lt)
        cdf.append(min(acc, 1.0))
        k += 1
    return cdf, k


def selection_bias_test(
    cells: list[dict],
    mc_reps: int,
    seed: int,
) -> dict:
    """cells: [{wheel, year, bet, draws, net_observed}]. Null: giocate casuali."""
    rng = random.Random(seed)
    samplers: dict[tuple[int, str], tuple[list[float], int]] = {}
    for c in cells:
        key = (c["draws"], c["bet"])
        if key not in samplers and c["draws"] > 0:
            samplers[key] = binom_sampler(c["draws"], random_hit_prob(c["bet"]))

    max_nets: list[float] = []
    n_pos_cells: list[int] = []
    for _ in range(mc_reps):
        best = -1e18
        pos = 0
        for c in cells:
            n = c["draws"]
            if n == 0:
                continue
            cdf, _ = samplers[(n, c["bet"])]
            u = rng.random()
            h = bisect_left(cdf, u)
            net = h * PAYOUT[c["bet"]] - n
            if net > best:
                best = net
            if net > 0:
                pos += 1
        max_nets.append(best)
        n_pos_cells.append(pos)

    max_nets.sort()
    obs_best = max(c["net_observed"] for c in cells)
    obs_pos = sum(1 for c in cells if c["net_observed"] > 0)
    rank = bisect_left(max_nets, obs_best)
    pct = rank / len(max_nets)
    return {
        "n_cells": len(cells),
        "mc_reps": mc_reps,
        "observed_best_cell_net": obs_best,
        "observed_positive_cells": obs_pos,
        "null_max_net_median": max_nets[len(max_nets) // 2],
        "null_max_net_p05": max_nets[int(0.05 * len(max_nets))],
        "null_max_net_p95": max_nets[int(0.95 * len(max_nets))],
        "observed_best_percentile_in_null": round(pct, 4),
        "null_positive_cells_mean": round(statistics.mean(n_pos_cells), 2),
        "verdict_best_cell_is_luck": pct < 0.95,
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years-back", type=int, default=5)
    ap.add_argument("--mc-reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--markov-pool-k", type=int, default=25)
    ap.add_argument("--out", default="data/analysis/five_year_validation.json")
    ap.add_argument("--out-md", default="data/analysis/FIVE_YEAR_VALIDATION.md")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    per_wheel, first_date, last_date = build_transitions(root)
    start = shift_years(last_date, args.years_back)

    test_w: dict[str, list[dict]] = {}
    train_w: dict[str, list[dict]] = {}
    for w in WHEELS:
        test_w[w] = [t for t in per_wheel[w] if t["date"] >= start]
        train_w[w] = [t for t in per_wheel[w] if t["date"] < start]

    n_test = sum(len(v) for v in test_w.values())
    print(f"Finestra di test: {start} -> {last_date} | transizioni totali: {n_test}")
    print(f"Training (regole smart): tutto prima di {start} "
          f"({sum(len(v) for v in train_w.values())} transizioni)")
    print()

    # --- regole smart addestrate su PRE-finestra (out-of-sample rigoroso) ---
    train_pairs = [(t["prev"], t["nxt"]) for w in WHEELS for t in train_w[w]]
    markov = MarkovLift(train_pairs)

    # storico per-ruota per hot/cold (serve indice nella sequenza)
    full_seq: dict[str, list[list[int]]] = {}
    for w in WHEELS:
        seq = []
        for t in per_wheel[w]:
            if not seq:
                seq.append(t["prev"])
            seq.append(t["nxt"])
        full_seq[w] = seq

    k = args.markov_pool_k

    def make_pool_rules(w: str) -> dict[str, callable]:
        rules: dict[str, callable] = {}
        for name, fn in RULES.items():
            rules[name] = lambda t, f=fn: f(t["prev"])
        rules["smart_markov_lift"] = lambda t: markov.pool(t["prev"], k)
        rules["smart_hot30"] = lambda t: hot_pool(full_seq[w][: t["idx"]], 30, k)
        rules["smart_cold_ritardatari"] = lambda t: cold_pool(full_seq[w][: t["idx"]], k)
        rng_ctrl = random.Random(args.seed * 7919 + sum(ord(c) for c in w))
        rules["control_random25"] = lambda t, r=rng_ctrl: set(r.sample(range(1, 91), k))
        return rules

    # ======================= TEST A =======================
    print("=" * 78)
    print("TEST A — POOL vs NULL IPERGEOMETRICO ESATTO (a parita' di pool size)")
    print("=" * 78)
    rule_names = list(RULES.keys()) + [
        "smart_markov_lift", "smart_hot30", "smart_cold_ritardatari", "control_random25",
    ]
    pool_results: dict[str, dict[str, dict]] = {rn: {} for rn in rule_names}
    agg: dict[str, dict] = {}
    for w in WHEELS:
        rules = make_pool_rules(w)
        for rn in rule_names:
            pool_results[rn][w] = eval_pool_rule(test_w[w], rules[rn])

    n_tests = len(rule_names) * len(WHEELS)
    alpha_bonf = 0.05 / n_tests
    for rn in rule_names:
        hits = sum(pool_results[rn][w]["hits"] for w in WHEELS)
        exp = sum(pool_results[rn][w]["expected_hits"] for w in WHEELS)
        var = sum(pool_results[rn][w].pop("var") for w in WHEELS)
        z = (hits - exp) / math.sqrt(var) if var > 0 else 0.0
        agg[rn] = {
            "hits": hits,
            "expected": round(exp, 1),
            "margin": round(hits - exp, 1),
            "z": round(z, 3),
            "p_one_sided": round(normal_sf(z), 6),
            "avg_pool": round(statistics.mean(
                pool_results[rn][w]["avg_pool"] for w in WHEELS), 1),
            "wheels_beating_null": sum(
                1 for w in WHEELS if pool_results[rn][w]["margin"] > 0),
            "wheels_sig_bonferroni": sum(
                1 for w in WHEELS
                if pool_results[rn][w]["p_one_sided"] < alpha_bonf),
        }

    print(f"{'Regola':<28} {'Pool':>5} {'Hit':>6} {'Attesi':>8} {'Margine':>8} "
          f"{'Z':>6} {'p':>8} {'Ruote+':>6} {'Bonf':>4}")
    print("-" * 88)
    for rn in sorted(agg, key=lambda x: -agg[x]["z"]):
        a = agg[rn]
        print(f"{rn:<28} {a['avg_pool']:>5.1f} {a['hits']:>6} {a['expected']:>8.1f} "
              f"{a['margin']:>+8.1f} {a['z']:>6.2f} {a['p_one_sided']:>8.4f} "
              f"{a['wheels_beating_null']:>4}/11 {a['wheels_sig_bonferroni']:>4}")
    print(f"\nCelle totali testate: {n_tests} | soglia Bonferroni: p < {alpha_bonf:.2e}")

    # ======================= TEST D (coverage) =======================
    cov_hits = exp_cov = 0.0
    n_tr = 0
    for w in WHEELS:
        for t in test_w[w]:
            pool = pool_user_tight(t["prev"], twin11=False)
            tol = set(pool)
            for n in pool:
                if n > 1:
                    tol.add(n - 1)
                if n < 90:
                    tol.add(n + 1)
            cov_hits += sum(1 for n in t["nxt"] if n in tol)
            exp_cov += 5 * len(tol) / 90
            n_tr += 1
    coverage = {
        "rule": "pool teoria utente (orig+diff+vert+comp) con tolleranza +/-1",
        "avg_hits_per_draw": round(cov_hits / n_tr, 3),
        "expected_by_coverage_alone": round(exp_cov / n_tr, 3),
    }
    print("\nTEST D — il 'ritrova 3-4 numeri su 5' spiegato dalla copertura:")
    print(f"  hit medi/estrazione osservati:        {coverage['avg_hits_per_draw']}")
    print(f"  attesi per sola copertura del pool:   {coverage['expected_by_coverage_alone']}")

    # ======================= TEST B =======================
    print()
    print("=" * 78)
    print("TEST B — BETTING 1 EUR/SORTE, 5 ANNI, per metodo (aggregato 11 ruote)")
    print("=" * 78)

    def picks_cross(t: dict) -> list[int]:
        pool = pool_cross_opt_v1(t["prev"])
        return top_n_from_pool(pool, score_cross_numbers(t["prev"]), 5)

    def picks_markov(t: dict) -> list[int]:
        return markov.picks(t["prev"], 5)

    bet_methods = {"cross_opt_v1": picks_cross, "smart_markov_lift": picks_markov}
    betting: dict[str, dict] = {}
    for mname, pfn in bet_methods.items():
        per_wheel_res: dict[str, dict] = {}
        for w in WHEELS:
            per_wheel_res[w] = eval_betting(test_w[w], pfn)
        tot: dict[str, dict] = {}
        for bet in K_FOR:
            n = sum(per_wheel_res[w][bet]["draws"] for w in WHEELS)
            h = sum(per_wheel_res[w][bet]["hits"] for w in WHEELS)
            p0 = random_hit_prob(bet)
            tot[bet] = {
                "draws": n, "hits": h,
                "expected_hits_random": round(n * p0, 2),
                "p_value_binom": round(binom_sf(h, n, p0), 4),
                "spent_eur": n,
                "won_gross_eur": round(h * PAYOUT[bet], 2),
                "net_gross_eur": round(h * PAYOUT[bet] - n, 2),
            }
        betting[mname] = {"per_wheel": per_wheel_res, "total": tot}
        print(f"\nMetodo: {mname}")
        print(f"{'Sorte':<10} {'Giocate':>8} {'Hit':>5} {'AttesiCaso':>10} "
              f"{'p-val':>7} {'Speso':>8} {'Vinto':>10} {'Netto':>10}")
        for bet in K_FOR:
            r = tot[bet]
            print(f"{bet:<10} {r['draws']:>8} {r['hits']:>5} "
                  f"{r['expected_hits_random']:>10.1f} {r['p_value_binom']:>7.3f} "
                  f"{r['spent_eur']:>8} {r['won_gross_eur']:>10.2f} "
                  f"{r['net_gross_eur']:>+10.2f}")

    # ======================= TEST C =======================
    print()
    print("=" * 78)
    print("TEST C — SELECTION BIAS: best cell osservata vs max nullo Monte Carlo")
    print("=" * 78)
    years = sorted({int(t["date"][:4]) for w in WHEELS for t in test_w[w]})
    cells: list[dict] = []
    for w in WHEELS:
        for y in years:
            res = eval_betting(test_w[w], picks_cross, year_filter=y)
            for bet in ("estratto", "ambo", "terno"):
                r = res[bet]
                if r["draws"] >= 20:
                    cells.append({
                        "wheel": w, "year": y, "bet": bet,
                        "draws": r["draws"], "hits": r["hits"],
                        "net_observed": r["net_gross_eur"],
                    })
    sel = selection_bias_test(cells, args.mc_reps, args.seed)
    best_cell = max(cells, key=lambda c: c["net_observed"])
    print(f"Celle (ruota x anno x sorte, >=20 giocate): {sel['n_cells']}")
    print(f"Best cell osservata (cross_opt_v1): {best_cell['wheel']} {best_cell['year']} "
          f"{best_cell['bet']} -> {best_cell['net_observed']:+.2f} EUR "
          f"({best_cell['hits']} hit su {best_cell['draws']})")
    print(f"Max nullo (giocate casuali, {args.mc_reps} rep): "
          f"mediana {sel['null_max_net_median']:+.0f} EUR | "
          f"p95 {sel['null_max_net_p95']:+.0f} EUR")
    print(f"Percentile del best osservato nella distribuzione del max nullo: "
          f"{sel['observed_best_percentile_in_null']*100:.1f}%")
    print(f"Celle in profitto: osservate {sel['observed_positive_cells']} | "
          f"attese per caso {sel['null_positive_cells_mean']}")
    print(f"VERDETTO: best cell compatibile col caso? "
          f"{'SI' if sel['verdict_best_cell_is_luck'] else 'NO'}")

    # ======================= report =======================
    sig_rules = [rn for rn in agg if agg[rn]["p_one_sided"] < 0.05 / len(rule_names)]
    verdict = {
        "pool_rules_significant_after_correction": sig_rules,
        "any_betting_method_profitable_5y": any(
            sum(b["total"][bet]["net_gross_eur"] for bet in K_FOR) > 0
            for b in betting.values()
        ),
        "best_cell_explainable_by_selection": sel["verdict_best_cell_is_luck"],
        "conclusion": (
            "Nessuna regola (cross o smart) batte il null esatto dopo correzione "
            "per test multipli; il betting 5 anni e' in perdita per tutti i metodi; "
            "le celle in profitto sono compatibili con la selezione a posteriori "
            "del massimo su una griglia di celle casuali."
            if not sig_rules and sel["verdict_best_cell_is_luck"]
            else "ATTENZIONE: trovato risultato che supera i controlli — verificare."
        ),
    }

    report = {
        "window": {"start": start, "end": last_date, "years_back": args.years_back,
                   "transitions_total": n_test},
        "test_A_pool_vs_exact_null": {
            "aggregate_by_rule": agg,
            "per_wheel": pool_results,
            "n_tests": n_tests,
            "bonferroni_alpha": alpha_bonf,
        },
        "test_B_betting": betting,
        "test_C_selection_bias": {**sel, "best_cell": best_cell, "cells": cells},
        "test_D_coverage": coverage,
        "verdict": verdict,
    }
    out = root / args.out
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport JSON: {out}")

    write_markdown(root / args.out_md, report, agg, betting, sel, best_cell, coverage,
                   start, last_date)
    print(f"Report MD:   {root / args.out_md}")
    return 0


def write_markdown(path: Path, report: dict, agg: dict, betting: dict, sel: dict,
                   best_cell: dict, coverage: dict, start: str, end: str) -> None:
    L: list[str] = []
    L.append("# Validazione definitiva — teoria cross, ultimi 5 anni\n")
    L.append(f"Finestra: **{start} → {end}**, tutte le 11 ruote "
             f"({report['window']['transitions_total']} transizioni totali).\n")
    L.append("## Test A — Pool vs null esatto (ipergeometrico, a parità di pool)\n")
    L.append("Per ogni regola: hit osservati vs attesi da un pool CASUALE della stessa "
             "dimensione. Z-score e p-value one-sided; `Bonf` = ruote significative "
             "dopo Bonferroni.\n")
    L.append("| Regola | Pool medio | Hit | Attesi | Margine | Z | p | Ruote>null | Bonf |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for rn in sorted(agg, key=lambda x: -agg[x]["z"]):
        a = agg[rn]
        L.append(f"| {rn} | {a['avg_pool']} | {a['hits']} | {a['expected']} | "
                 f"{a['margin']:+} | {a['z']:.2f} | {a['p_one_sided']:.4f} | "
                 f"{a['wheels_beating_null']}/11 | {a['wheels_sig_bonferroni']} |")
    sig = report["verdict"]["pool_rules_significant_after_correction"]
    L.append(f"\nRegole significative dopo correzione: **{sig if sig else 'nessuna'}**.\n")
    top_rule = max(agg, key=lambda x: agg[x]["z"])
    if top_rule == "control_random25":
        L.append("Nota: la regola col punteggio migliore è il **controllo casuale** "
                 "(pool di 25 numeri estratti a caso). Se un pool puramente casuale "
                 "“batte” tutte le ricette cross, la classifica misura solo rumore.\n")
    L.append("## Test D — Il “ritrova 3-4 numeri su 5” è copertura\n")
    L.append(f"- Hit medi/estrazione del pool teoria (±1): "
             f"**{coverage['avg_hits_per_draw']}**")
    L.append(f"- Attesi per la SOLA dimensione del pool: "
             f"**{coverage['expected_by_coverage_alone']}**\n")
    L.append("## Test B — Betting 1 € intero/sorte, 5 anni, 11 ruote\n")
    for m, b in betting.items():
        L.append(f"### {m}\n")
        L.append("| Sorte | Giocate | Hit | Attesi (caso) | p-value | Netto € |")
        L.append("|---|---|---|---|---|---|")
        for bet, r in b["total"].items():
            L.append(f"| {bet} | {r['draws']} | {r['hits']} | "
                     f"{r['expected_hits_random']} | {r['p_value_binom']} | "
                     f"{r['net_gross_eur']:+.2f} |")
        L.append("")
    L.append("## Test C — Selection bias (la “cella fortunata”)\n")
    L.append(f"- Best cell osservata: **{best_cell['wheel']} {best_cell['year']} "
             f"{best_cell['bet']} = {best_cell['net_observed']:+.2f} €**")
    L.append(f"- Max nullo su {sel['n_cells']} celle con giocate casuali "
             f"({sel['mc_reps']} rep): mediana {sel['null_max_net_median']:+.0f} €, "
             f"p95 {sel['null_max_net_p95']:+.0f} €")
    L.append(f"- Percentile del best osservato nel max nullo: "
             f"**{sel['observed_best_percentile_in_null']*100:.1f}%**")
    L.append(f"- Celle in profitto: osservate {sel['observed_positive_cells']}, "
             f"attese per caso {sel['null_positive_cells_mean']}\n")
    L.append("## Verdetto\n")
    L.append(report["verdict"]["conclusion"] + "\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
