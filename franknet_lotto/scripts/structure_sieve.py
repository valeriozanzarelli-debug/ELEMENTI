#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setaccio strutturale con nulli casuali (NON è decrittazione).

Le estrazioni del lotto non sono un ciphertext: non esiste un algoritmo nascosto
da "rompere". Questo script applica molte statistiche (autocorrelazione, spettro,
indici tipo-frattali, test delle run, DFA) su una serie numerica e confronta ogni
valore con la distribuzione ottenuta **permutando** la serie (stessa distribuzione
marginale, ordine casuale).

- Non esegue un loop "finché la probabilità è buona": sarebbe p-hacking garantito.
- Segnala quante metriche superano soglie nominali e ricorda la correzione per
  test multipli.

Requisito: numpy (requirements_ml.txt o pip install numpy).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Callable

try:
    import numpy as np
except ImportError:
    print("Serve numpy: python -m pip install numpy", file=sys.stderr)
    raise SystemExit(1)


def digital_root(n: int) -> int:
    while n > 9:
        n = sum(int(c) for c in str(n))
    return n


def load_series_n(
    wide_csv: Path, wheel: str, position: int, from_year: int
) -> np.ndarray:
    col = f"{wheel}_{position}"
    xs: list[int] = []
    with wide_csv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["year"]) < from_year:
                continue
            v = row[col]
            if v == "":
                continue
            xs.append(int(v))
    return np.asarray(xs, dtype=np.float64)


def load_series_dr(
    wide_csv: Path, wheel: str, position: int, from_year: int
) -> np.ndarray:
    n = load_series_n(wide_csv, wheel, position, from_year)
    return np.asarray([digital_root(int(x)) for x in n.tolist()], dtype=np.float64)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return x * 0.0
    return (x - m) / s


def metric_lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 5:
        return 0.0
    a, b = x[:-1], x[1:]
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def metric_spectral_peak_ratio(x: np.ndarray) -> float:
    """Quota di energia nelle 5 bin FFT più alte (dopo rimozione DC)."""
    x = zscore(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n < 32:
        return 0.0
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec[0] = 0.0
    t = spec.sum()
    if t <= 0:
        return 0.0
    k = min(5, spec.size)
    return float(np.partition(spec, -k)[-k:].sum() / t)


def metric_higuchi_fd(x: np.ndarray, kmax: int = 16) -> float:
    """Dimensione frattale di Higuchi (1D classica). ~1.5 rumore, <1.5 trend, >1.5 memoria."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < kmax * 4:
        return 1.5
    lk: list[float] = []
    kvals: list[float] = []
    for k in range(1, kmax + 1):
        lm = []
        for m in range(1, k + 1):
            ll = 0.0
            idx = np.arange(m - 1, n, k)
            if len(idx) < 2:
                continue
            dif = np.abs(np.diff(x[idx]))
            ll = float(dif.sum()) * (n - 1) / (len(idx) * k)
            lm.append(ll)
        if not lm:
            continue
        lk.append(float(np.mean(lm)))
        kvals.append(1.0 / k)
    if len(lk) < 3:
        return 1.5
    lx = np.log(np.asarray(kvals))
    ly = np.log(np.asarray(lk))
    slope, _ = np.polyfit(lx, ly, 1)
    return float(2 - slope)


def metric_hurst_rs(x: np.ndarray) -> float:
    """Hurst da R/S su finestre crescenti (grossolano). 0.5 = random walk-ish."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 128:
        return 0.5
    y = np.cumsum(x - x.mean())
    sizes = np.unique(
        np.linspace(16, min(n // 4, 512), 12, dtype=int)
    )
    rs = []
    ss = []
    for size in sizes:
        if size < 8:
            continue
        chunks = n // size
        if chunks < 2:
            continue
        vals = []
        for j in range(chunks):
            seg = y[j * size : (j + 1) * size]
            r = float(seg.max() - seg.min())
            s = float(np.std(x[j * size : (j + 1) * size]))
            if s > 1e-9:
                vals.append(r / s)
        if vals:
            rs.append(float(np.mean(vals)))
            ss.append(float(size))
    if len(rs) < 3:
        return 0.5
    lx = np.log(ss)
    ly = np.log(rs)
    h = float(np.polyfit(lx, ly, 1)[0])
    return max(0.0, min(1.5, h))


def metric_dfa_alpha(x: np.ndarray) -> float:
    """DFA: pendenza log F(s) vs log s (≈ Hurst per serie integrate)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 100:
        return 0.5
    y = np.cumsum(x - x.mean())
    scales = np.unique(
        np.logspace(math.log10(10), math.log10(n // 5), 10, dtype=int)
    )
    fs = []
    ls = []
    for s in scales:
        if s < 10:
            continue
        m = n // s
        if m < 4:
            continue
        fluct = []
        for j in range(m):
            seg = y[j * s : (j + 1) * s]
            t = np.arange(len(seg), dtype=np.float64)
            a, b = np.polyfit(t, seg, 1)
            det = seg - (a * t + b)
            fluct.append(float(np.sqrt((det**2).mean())))
        if fluct:
            fs.append(float(np.mean(fluct)))
            ls.append(float(s))
    if len(fs) < 3:
        return 0.5
    return float(np.polyfit(np.log(ls), np.log(fs), 1)[0])


def metric_runs_z(x: np.ndarray) -> float:
    """Z-score delle run sopra/sotto mediana (simmetria ~ N(0,1) sotto nulla i.i.d.)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 30:
        return 0.0
    med = np.median(x)
    b = (x > med).astype(np.int8)
    runs = 1
    for i in range(1, n):
        if b[i] != b[i - 1]:
            runs += 1
    n1 = int(b.sum())
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        return 0.0
    exp_r = 1.0 + 2.0 * n0 * n1 / n
    var_r = 2.0 * n0 * n1 * (2.0 * n0 * n1 - n) / (n * n * (n - 1))
    if var_r <= 0:
        return 0.0
    return float((runs - exp_r) / math.sqrt(var_r))


def metric_diff_variance_ratio(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 5:
        return 1.0
    d = np.diff(x)
    vx = float(np.var(x))
    vd = float(np.var(d))
    if vx < 1e-12:
        return 1.0
    return vd / vx


def metric_skew_kurt_gap(x: np.ndarray) -> float:
    """Distanza da skew/kurtosi gaussiani (|skew| + |kurt-3|)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 20:
        return 0.0
    z = zscore(x)
    sk = float((z**3).mean())
    ku = float((z**4).mean())
    return abs(sk) + abs(ku - 3.0)


METRICS: list[tuple[str, Callable[[np.ndarray], float]]] = [
    ("lag1_autocorr", metric_lag1_autocorr),
    ("spectral_peak_ratio", metric_spectral_peak_ratio),
    ("higuchi_fractal_dim", metric_higuchi_fd),
    ("hurst_rs", metric_hurst_rs),
    ("dfa_alpha", metric_dfa_alpha),
    ("runs_z_above_median", metric_runs_z),
    ("diff_variance_ratio", metric_diff_variance_ratio),
    ("skew_kurtosis_gap", metric_skew_kurt_gap),
]


def two_sided_pvalue(real: float, nulls: list[float]) -> float:
    arr = np.asarray(nulls, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    lo = float((arr <= real).mean())
    hi = float((arr >= real).mean())
    return float(min(1.0, 2 * min(lo, hi)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Setaccio statistico con nulli a permutazione (non decrittazione)."
    )
    ap.add_argument("--wheel", default="MILANO")
    ap.add_argument("--position", type=int, default=1, help="1..5")
    ap.add_argument("--from-year", type=int, default=1900)
    ap.add_argument(
        "--mode",
        choices=("n", "dr"),
        default="n",
        help="Serie: numeri grezzi o radice digitale",
    )
    ap.add_argument("--null-reps", type=int, default=199, help="Permutazioni per metrica")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not 1 <= args.position <= 5:
        print("--position 1..5", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent.parent
    wide = root / "data" / "draws_wide.csv"
    if not wide.is_file():
        print("Manca data/draws_wide.csv", file=sys.stderr)
        return 1

    if args.mode == "n":
        x = load_series_n(wide, args.wheel, args.position, args.from_year)
    else:
        x = load_series_dr(wide, args.wheel, args.position, args.from_year)

    if len(x) < 80:
        print("Serie troppo corta.", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    results = []
    nominal_hits_05 = 0
    nominal_hits_01 = 0

    for name, fn in METRICS:
        real = float(fn(x))
        nulls = []
        for _ in range(args.null_reps):
            xp = rng.permutation(x)
            nulls.append(float(fn(xp)))
        p = two_sided_pvalue(real, nulls)
        pct = float(np.mean(np.asarray(nulls) <= real) * 100.0)
        row = {
            "metric": name,
            "value_on_data": round(real, 6),
            "p_value_two_sided_perm": round(p, 6),
            "percentile_rank_le": round(pct, 2),
        }
        results.append(row)
        if p < 0.05:
            nominal_hits_05 += 1
        if p < 0.01:
            nominal_hits_01 += 1

    n_metrics = len(METRICS)
    bonf_05 = 0.05 / n_metrics
    bonf_hits = sum(1 for r in results if r["p_value_two_sided_perm"] < bonf_05)

    report = {
        "disclaimer": (
            "Non è decrittazione. I p-value sono nominali; con molte metriche "
            "ci si aspetta falsi positivi ~5% sotto nulla. Usare Bonferroni o FDR."
        ),
        "series": {
            "wheel": args.wheel,
            "position": args.position,
            "mode": args.mode,
            "from_year": args.from_year,
            "n": int(len(x)),
        },
        "null": {"reps": args.null_reps, "method": "iid_permutation"},
        "metrics": results,
        "summary": {
            "n_metrics": n_metrics,
            "nominal_p_lt_0.05_count": nominal_hits_05,
            "nominal_p_lt_0.01_count": nominal_hits_01,
            "bonferroni_alpha_0.05_per_test": round(bonf_05, 6),
            "metrics_below_bonferroni_0.05": bonf_hits,
        },
    }

    out_dir = root / "data" / "analysis" / "structure_sieve"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.wheel.lower()}_pos{args.position}_{args.mode}_y{args.from_year}"
    jpath = out_dir / f"{tag}_sieve_report.json"
    jpath.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"\nReport completo: {jpath}")
    if bonf_hits == 0:
        print(
            "\nNessuna metrica resta significativa dopo Bonferroni: "
            "nulla di robusto tipo 'decrittato'."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
