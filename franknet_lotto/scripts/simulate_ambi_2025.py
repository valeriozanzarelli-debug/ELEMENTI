#!/usr/bin/env python3
"""
Simulazione gioco AMBI (ambo) su tutte le ruote, anno 2025.
Usa pool cross (cross_opt_v1 / config ottimizzata) -> top-N numeri -> tutte le coppie.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

from cross_opt_train_test import CrossOptConfig, make_fn, pool_cross_configurable
from vertibile_rule_test import (
    adj_diffs,
    comp,
    is_twin,
    neighbors,
    pool_cross_only,
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

# Regole ufficiali ADM (come-si-gioca): minimo 1€ per schedina, moltiplicatore ambo 250x
# su singola ruota; la posta si ripartisce tra le C(n,2) combinazioni ambo generate.
DEFAULT_STAKE_EUR = 1.0
DEFAULT_AMBO_MULT = 250.0
DEFAULT_TOP_N = 8
DEFAULT_YEAR = 2025
DEFAULT_TAX_RATE = 0.08  # aliquota 8% vigente nel 2025 (lordo -> netto)


@dataclass(frozen=True)
class Assumptions:
    year: int
    stake_eur_per_wheel: float
    ambo_multiplier: float
    top_n: int
    tax_rate: float
    strategy: str
    payout_mode: str

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "bet_cost_eur_per_wheel_per_draw": self.stake_eur_per_wheel,
            "ambo_multiplier_gross": self.ambo_multiplier,
            "top_n_numbers": self.top_n,
            "pairs_per_wheel": math.comb(self.top_n, 2),
            "stake_per_ambo_eur": round(
                self.stake_eur_per_wheel / math.comb(self.top_n, 2), 6
            ),
            "win_per_single_ambo_eur_gross": round(
                self.stake_eur_per_wheel / math.comb(self.top_n, 2) * self.ambo_multiplier,
                4,
            ),
            "tax_rate_on_winnings": self.tax_rate,
            "strategy": self.strategy,
            "payout_mode": self.payout_mode,
            "rules_reference": (
                "Ambo su 1 ruota: vincono le coppie di numeri giocati se ENTRAMBI "
                "compaiono tra i 5 estratti su quella ruota. "
                "Fonte moltiplicatore 250x e posta minima 1€: ADM / regolamento Lotto."
            ),
        }


def score_cross_numbers(nums: list[int]) -> dict[int, int]:
    """Punteggio ricette cross (come cross_method_report.pool_cross_scored)."""
    scores: defaultdict[int, int] = defaultdict(int)

    def bump(n: int | None, w: int) -> None:
        if n is not None and 1 <= n <= 90:
            scores[n] += w

    diffs = adj_diffs(nums)
    for n in nums:
        cn = comp(n)
        for d in diffs:
            bump(n + d, 2)
            bump(abs(n - d), 1)
            if cn is not None:
                bump(cn + d, 4)
                bump(abs(cn - d), 2)
        for base in filter(None, [n, cn, vertibile(n)]):
            for x in neighbors(base, True):
                bump(x, 2 if is_twin(base) else 1)
    if 45 in nums:
        bump(45, 3)
    if 90 in nums:
        bump(9, 2)
    return dict(scores)


def top_n_from_pool(pool: set[int], scores: dict[int, int], n: int) -> list[int]:
    """Top-N per punteggio, solo numeri presenti nel pool."""
    ranked = sorted(
        ((num, scores.get(num, 0)) for num in pool),
        key=lambda x: (-x[1], x[0]),
    )
    return [num for num, _ in ranked[:n]]


def count_winning_ambi(picks: list[int], drawn: set[int]) -> int:
    return sum(1 for a, b in combinations(picks, 2) if a in drawn and b in drawn)


def net_from_gross(gross: float, tax_rate: float) -> float:
    if gross <= 0:
        return 0.0
    return gross * (1.0 - tax_rate)


def load_draws_by_date(wide: Path) -> tuple[list[str], dict[str, dict[str, list[int]]]]:
    """date -> wheel -> quintina (solo ruote complete)."""
    cols = {w: [f"{w}_{k}" for k in range(1, 6)] for w in ALL_WHEELS}
    by_date: dict[str, dict[str, list[int]]] = {}
    dates: list[str] = []
    with wide.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            d = row["date"]
            wheels_today: dict[str, list[int]] = {}
            for w in ALL_WHEELS:
                if all(row[c] for c in cols[w]):
                    wheels_today[w] = [int(row[c]) for c in cols[w]]
            if wheels_today:
                by_date[d] = wheels_today
                dates.append(d)
    dates.sort()
    return dates, by_date


def load_optimized_config(root: Path) -> CrossOptConfig | None:
    path = root / "data" / "analysis" / "cross_opt_train_test_500.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    opt = data.get("optimized", {})
    cfg = opt.get("config")
    if not cfg:
        return None
    return CrossOptConfig(**cfg)


def simulate_method(
    *,
    dates_all: list[str],
    by_date: dict[str, dict[str, list[int]]],
    year: int,
    pool_fn: Callable[[list[int]], set[int]],
    method_name: str,
    assumptions: Assumptions,
) -> dict:
    year_dates = [d for d in dates_all if d.startswith(f"{year}-")]
    prev_quintina: dict[str, list[int]] = {}

    # seed history from last draw before year start
    for d in dates_all:
        if d >= f"{year}-01-01":
            break
        for w, nums in by_date.get(d, {}).items():
            prev_quintina[w] = nums

    daily: list[dict] = []
    by_wheel = {w: {"spent": 0.0, "won_gross": 0.0, "won_net": 0.0, "ambi_hits": 0, "draws": 0} for w in ALL_WHEELS}
    by_month: dict[str, dict] = defaultdict(lambda: {"spent": 0.0, "won_gross": 0.0, "won_net": 0.0, "days": 0})

    total_spent = 0.0
    total_won_gross = 0.0
    total_ambi_hits = 0
    total_wheels_played = 0
    pairs_n = math.comb(assumptions.top_n, 2)
    stake_per_ambo = assumptions.stake_eur_per_wheel / pairs_n

    for d in year_dates:
        wheels_today = by_date[d]
        day_spent = 0.0
        day_won_gross = 0.0
        day_ambi = 0
        day_wheels = 0

        for w in ALL_WHEELS:
            if w not in wheels_today or w not in prev_quintina:
                continue

            pool = pool_fn(prev_quintina[w])
            scores = score_cross_numbers(prev_quintina[w])
            picks = top_n_from_pool(pool, scores, assumptions.top_n)
            if len(picks) < 2:
                continue

            spent = assumptions.stake_eur_per_wheel
            drawn = set(wheels_today[w])
            hits = count_winning_ambi(picks, drawn)
            won_gross = hits * stake_per_ambo * assumptions.ambo_multiplier
            won_net = net_from_gross(won_gross, assumptions.tax_rate)

            day_spent += spent
            day_won_gross += won_gross
            day_ambi += hits
            day_wheels += 1

            bw = by_wheel[w]
            bw["spent"] += spent
            bw["won_gross"] += won_gross
            bw["won_net"] += won_net
            bw["ambi_hits"] += hits
            bw["draws"] += 1

        for w, nums in wheels_today.items():
            prev_quintina[w] = nums

        if day_wheels == 0:
            continue

        day_won_net = net_from_gross(day_won_gross, assumptions.tax_rate)
        month = d[:7]
        by_month[month]["spent"] += day_spent
        by_month[month]["won_gross"] += day_won_gross
        by_month[month]["won_net"] += day_won_net
        by_month[month]["days"] += 1

        total_spent += day_spent
        total_won_gross += day_won_gross
        total_ambi_hits += day_ambi
        total_wheels_played += day_wheels

        daily.append(
            {
                "date": d,
                "wheels_played": day_wheels,
                "spent_eur": round(day_spent, 2),
                "won_gross_eur": round(day_won_gross, 2),
                "won_net_eur": round(day_won_net, 2),
                "net_gross_eur": round(day_won_gross - day_spent, 2),
                "net_net_eur": round(day_won_net - day_spent, 2),
                "ambi_hits": day_ambi,
            }
        )

    total_won_net = net_from_gross(total_won_gross, assumptions.tax_rate)
    n_days = len(daily)
    avg_daily_spend = total_spent / n_days if n_days else 0.0

    monthly_rows = []
    for m in sorted(by_month):
        bm = by_month[m]
        monthly_rows.append(
            {
                "month": m,
                "days": bm["days"],
                "spent_eur": round(bm["spent"], 2),
                "won_gross_eur": round(bm["won_gross"], 2),
                "won_net_eur": round(bm["won_net"], 2),
                "net_gross_eur": round(bm["won_gross"] - bm["spent"], 2),
                "net_net_eur": round(bm["won_net"] - bm["spent"], 2),
            }
        )

    best_month = max(monthly_rows, key=lambda x: x["net_gross_eur"]) if monthly_rows else None
    worst_month = min(monthly_rows, key=lambda x: x["net_gross_eur"]) if monthly_rows else None

    wheel_rows = []
    for w in ALL_WHEELS:
        bw = by_wheel[w]
        if bw["draws"] == 0:
            continue
        wheel_rows.append(
            {
                "wheel": w,
                "draws": bw["draws"],
                "spent_eur": round(bw["spent"], 2),
                "won_gross_eur": round(bw["won_gross"], 2),
                "won_net_eur": round(bw["won_net"], 2),
                "net_gross_eur": round(bw["won_gross"] - bw["spent"], 2),
                "ambi_hits": bw["ambi_hits"],
                "hit_rate_per_draw": round(bw["ambi_hits"] / bw["draws"], 4),
            }
        )

    return {
        "method": method_name,
        "summary": {
            "draw_days": n_days,
            "wheels_played_total": total_wheels_played,
            "total_spent_eur": round(total_spent, 2),
            "total_won_gross_eur": round(total_won_gross, 2),
            "total_won_net_eur": round(total_won_net, 2),
            "net_gross_eur": round(total_won_gross - total_spent, 2),
            "net_net_eur": round(total_won_net - total_spent, 2),
            "roi_gross_pct": round(100 * (total_won_gross - total_spent) / total_spent, 2) if total_spent else 0,
            "avg_daily_spend_eur": round(avg_daily_spend, 2),
            "avg_daily_won_gross_eur": round(total_won_gross / n_days, 2) if n_days else 0,
            "total_ambi_hits": total_ambi_hits,
            "expected_random_ambi_per_draw": round(
                pairs_n * math.comb(5, 2) / math.comb(90, 2), 6
            ),
        },
        "monthly": monthly_rows,
        "best_month": best_month,
        "worst_month": worst_month,
        "by_wheel": sorted(wheel_rows, key=lambda x: x["net_gross_eur"], reverse=True),
        "daily": daily,
    }


def weekly_summary(daily: list[dict]) -> list[dict]:
    """Aggregato ISO week per tabella compatta."""
    buckets: dict[str, dict] = defaultdict(lambda: {"spent": 0.0, "won": 0.0, "days": 0})
    for row in daily:
        y, m, dd = map(int, row["date"].split("-"))
        # ISO week key via datetime would be cleaner; approximate with date prefix
        key = row["date"][:7] + "-W" + str((dd - 1) // 7 + 1)
        buckets[key]["spent"] += row["spent_eur"]
        buckets[key]["won"] += row["won_gross_eur"]
        buckets[key]["days"] += 1
    out = []
    for k in sorted(buckets):
        b = buckets[k]
        out.append(
            {
                "period": k,
                "days": b["days"],
                "spent_eur": round(b["spent"], 2),
                "won_gross_eur": round(b["won"], 2),
                "net_gross_eur": round(b["won"] - b["spent"], 2),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Simula AMBI 2025 con metodi cross")
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--stake", type=float, default=DEFAULT_STAKE_EUR)
    ap.add_argument("--ambo-mult", type=float, default=DEFAULT_AMBO_MULT)
    ap.add_argument("--tax-rate", type=float, default=DEFAULT_TAX_RATE)
    ap.add_argument(
        "--out",
        default="data/analysis/ambi_simulation_2025.json",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    wide = root / "data" / "draws_wide.csv"
    dates_all, by_date = load_draws_by_date(wide)

    strategy_text = (
        f"Per ogni estrazione {args.year}, su ogni ruota con quintina precedente: "
        f"(1) pool dal metodo cross sulla quintina T-1; "
        f"(2) top-{args.top_n} numeri per punteggio ricette cross nel pool; "
        f"(3) giocate tutte le C({args.top_n},2)={math.comb(args.top_n, 2)} coppie ambo; "
        f"(4) posta {args.stake}€/ruota ripartita equamente sulle coppie."
    )
    assumptions = Assumptions(
        year=args.year,
        stake_eur_per_wheel=args.stake,
        ambo_multiplier=args.ambo_mult,
        top_n=args.top_n,
        tax_rate=args.tax_rate,
        strategy=strategy_text,
        payout_mode="fixed_multiplier_official",
    )

    methods: list[tuple[str, Callable[[list[int]], set[int]]]] = [
        ("cross_opt_v1", pool_cross_opt_v1),
        ("cross_only", lambda n: pool_cross_only(n, twin11=False)),
        ("cross_twin11", lambda n: pool_cross_only(n, twin11=True)),
    ]

    opt_cfg = load_optimized_config(root)
    if opt_cfg is not None:
        methods.append(("cross_opt_optimized_v1+no_vdiff", make_fn(opt_cfg)))

    results = {}
    for name, fn in methods:
        results[name] = simulate_method(
            dates_all=dates_all,
            by_date=by_date,
            year=args.year,
            pool_fn=fn,
            method_name=name,
            assumptions=assumptions,
        )

    primary = results["cross_opt_v1"]
    comparison = {
        name: {
            "total_spent_eur": r["summary"]["total_spent_eur"],
            "total_won_gross_eur": r["summary"]["total_won_gross_eur"],
            "net_gross_eur": r["summary"]["net_gross_eur"],
            "total_ambi_hits": r["summary"]["total_ambi_hits"],
            "vs_cross_opt_v1_net": round(
                r["summary"]["net_gross_eur"] - primary["summary"]["net_gross_eur"], 2
            ),
        }
        for name, r in results.items()
    }

    report = {
        "assumptions": assumptions.to_dict(),
        "optimized_config_loaded": opt_cfg is not None,
        "optimized_config": None if opt_cfg is None else {
            k: getattr(opt_cfg, k) for k in CrossOptConfig.__dataclass_fields__
        },
        "methods": results,
        "comparison": comparison,
        "primary_method": "cross_opt_v1",
        "weekly_cross_opt_v1": weekly_summary(primary["daily"]),
        "disclaimer": (
            "Simulazione storica con regole e quote semplificate. "
            "Performance passate non garantiscono risultati futuri. "
            "Il Lotto ha EV negativo in media; eventuali margini osservati "
            "possono essere fluttuazione statistica."
        ),
    }

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    s = primary["summary"]
    print(f"=== Simulazione AMBI {args.year} | metodo cross_opt_v1 ===")
    print(f"Giorni estrazione: {s['draw_days']} | Ruote-giocata totali: {s['wheels_played_total']}")
    print(f"Speso totale:     {s['total_spent_eur']:,.2f} €")
    print(f"Vinto lordo:      {s['total_won_gross_eur']:,.2f} €")
    print(f"Netto lordo:      {s['net_gross_eur']:+,.2f} €")
    print(f"Vinto netto (-8%): {s['total_won_net_eur']:,.2f} € | Netto netto: {s['net_net_eur']:+,.2f} €")
    print(f"Spesa media/giorno: {s['avg_daily_spend_eur']:.2f} €")
    print(f"Ambi vinti: {s['total_ambi_hits']}")
    if primary["best_month"]:
        bm = primary["best_month"]
        wm = primary["worst_month"]
        print(f"Mese migliore: {bm['month']} ({bm['net_gross_eur']:+,.2f} €)")
        print(f"Mese peggiore: {wm['month']} ({wm['net_gross_eur']:+,.2f} €)")
    print("\nConfronto metodi (netto lordo):")
    for name, c in sorted(comparison.items(), key=lambda x: -x[1]["net_gross_eur"]):
        print(f"  {name:<28} {c['net_gross_eur']:+10,.2f} €  (ambi={c['total_ambi_hits']})")
    print(f"\nReport JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
