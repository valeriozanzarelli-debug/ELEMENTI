#!/usr/bin/env python3
"""
Monte Carlo pesante: null per stream di cifre (uniforme i.i.d.) e permutazione
della componente "to" nelle transizioni decile->decile tra passi consecutivi
(first_hit_steps). Più Benjamini-Hochberg FDR su un pannello di p-value.

Requisito: numpy. Uso: python scripts/heavy_null_engine.py [--tier full|standard|quick]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def chi2_uniform_digits(d: np.ndarray, n_bins: int = 10) -> float:
    counts = np.bincount(d, minlength=n_bins)
    n = int(d.size)
    e = n / n_bins
    return float(((counts.astype(np.float64) - e) ** 2 / e).sum())


def chi2_independence_table(mat: np.ndarray) -> float:
    """mat shape (r,c) nonnegative."""
    r, c = mat.shape
    rs = mat.sum(axis=1, keepdims=True)
    cs = mat.sum(axis=0, keepdims=True)
    tot = mat.sum()
    if tot <= 0:
        return 0.0
    exp = rs @ cs / tot
    mask = exp > 0
    return float((((mat - exp) ** 2 / exp) * mask).sum())


def benjamini_hochberg_adjusted(
    pvals: list[float], fdr: float = 0.05
) -> list[tuple[float, float, bool]]:
    """Adjusted p-values (Benjamini-Hochberg); reject if adj <= fdr."""
    m = len(pvals)
    if m == 0:
        return []
    arr = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(arr)
    ranked = arr[order]
    adj_sorted = np.empty(m)
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * m / rank)
        adj_sorted[i] = running
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return [
        (float(arr[i]), float(adj[i]), bool(adj[i] <= fdr)) for i in range(m)
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("quick", "standard", "full"), default="standard")
    args = ap.parse_args()

    presets = {
        "quick": {"digit_mc": 2000, "trans_perm": 800},
        "standard": {"digit_mc": 8000, "trans_perm": 4000},
        "full": {"digit_mc": 25000, "trans_perm": 12000},
    }
    cfg = presets[args.tier]

    root = Path(__file__).resolve().parent.parent
    cdir = root / "data" / "analysis" / "coupon_cycles"
    fhdir = root / "data" / "analysis" / "first_hit_chain"
    out_dir = root / "data" / "analysis" / "heavy_null"
    out_dir.mkdir(parents=True, exist_ok=True)

    stream_path = cdir / "concat_digit_stream.txt"
    steps_path = fhdir / "first_hit_steps.csv"
    if not stream_path.is_file():
        print("Manca coupon_cycles/concat_digit_stream.txt — esegui coupon_cycle_stream.py", file=sys.stderr)
        return 1
    if not steps_path.is_file():
        print("Manca first_hit_chain/first_hit_steps.csv — esegui first_hit_chain_analysis.py", file=sys.stderr)
        return 1

    raw = stream_path.read_text(encoding="utf-8").strip()
    digits = np.frombuffer(raw.encode(), dtype=np.uint8) - ord("0")
    n = int(digits.size)
    obs_digit_chi2 = chi2_uniform_digits(digits)

    rng = np.random.default_rng(2026)
    null_digit = np.array(
        [
            chi2_uniform_digits(rng.integers(0, 10, size=n, dtype=np.int64))
            for _ in range(cfg["digit_mc"])
        ]
    )
    # p conservativo (1+k)/(1+N) per non restituire 0.0 quando obs è estremo
    k_digit = int((null_digit >= obs_digit_chi2).sum())
    p_digit = (1 + k_digit) / (1 + cfg["digit_mc"])

    # Transizioni decile-decile da first_hit_steps (valori consecutivi nello stesso ciclo)
    from_d: list[int] = []
    to_d: list[int] = []
    with steps_path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    by_cycle: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        cid = row["cycle_id"]
        step = int(row["step"])
        v = int(row["first_hit_value"])
        by_cycle.setdefault(cid, []).append((step, v))
    for cid, lst in by_cycle.items():
        lst.sort(key=lambda x: x[0])
        vals = [v for _, v in lst]
        for i in range(len(vals) - 1):
            a = min(9, (vals[i] - 1) * 10 // 90)
            b = min(9, (vals[i + 1] - 1) * 10 // 90)
            from_d.append(a)
            to_d.append(b)

    fd = np.asarray(from_d, dtype=np.int64)
    td = np.asarray(to_d, dtype=np.int64)
    mat = np.zeros((10, 10), dtype=np.float64)
    for i in range(fd.size):
        mat[fd[i], td[i]] += 1
    obs_trans_chi2 = chi2_independence_table(mat)

    null_trans = []
    td_copy = td.copy()
    for _ in range(cfg["trans_perm"]):
        rng.shuffle(td_copy)
        m2 = np.zeros((10, 10), dtype=np.float64)
        for i in range(fd.size):
            m2[fd[i], td_copy[i]] += 1
        null_trans.append(chi2_independence_table(m2))
    null_trans_arr = np.asarray(null_trans)
    k_trans = int((null_trans_arr >= obs_trans_chi2).sum())
    p_trans = (1 + k_trans) / (1 + cfg["trans_perm"])

    # Pannello p-value (estendibile)
    names = [
        "digit_stream_chi2_vs_iid_uniform",
        "decile_transition_chi2_perm_independence",
    ]
    pvals = [p_digit, p_trans]
    bh = benjamini_hochberg_adjusted(pvals, fdr=0.05)

    bh_rows = [
        {
            "name": names[i],
            "p_value": round(bh[i][0], 6),
            "adjusted_p": round(bh[i][1], 6),
            "reject": bh[i][2],
        }
        for i in range(len(names))
    ]
    n_rej = sum(1 for r in bh_rows if r["reject"])
    interp_parts: list[str] = []
    if p_digit >= 0.05:
        interp_parts.append(
            "Stream cifre (sum mod 10): chi2 non estremo vs i.i.d. uniforme (p_MC non piccolo)."
        )
    else:
        interp_parts.append(
            "Stream cifre: chi2 piu estremo del null i.i.d. uniforme (p_MC piccolo)."
        )
    if p_trans >= 0.05:
        interp_parts.append(
            "Transizioni decile->decile (colonna to permutata): compatibili con indipendenza condizionata al margine from."
        )
    else:
        interp_parts.append(
            "Transizioni decile->decile: il chi2 osservato supera quasi tutti i null permutati; "
            "associazione residua from->to oltre i margini (o violazione delle ipotesi del test)."
        )
    if n_rej == 0:
        interp_parts.append("BH-FDR q=0.05: nessun test rigettato dopo aggiustamento.")
    else:
        interp_parts.append(
            f"BH-FDR q=0.05: {n_rej} test rigettato/i su {len(bh_rows)} (vedi reject per nome)."
        )

    report = {
        "tier": args.tier,
        "monte_carlo": {
            "digit_chi2_observed": round(obs_digit_chi2, 6),
            "digit_mc_reps": cfg["digit_mc"],
            "p_value_digit": round(p_digit, 6),
            "transitions_pooled": int(fd.size),
            "trans_chi2_observed": round(obs_trans_chi2, 6),
            "trans_perm_reps": cfg["trans_perm"],
            "p_value_transition": round(p_trans, 6),
        },
        "benjamini_hochberg_fdr_0.05": bh_rows,
        "interpretation": " ".join(interp_parts),
    }

    out_path = out_dir / f"heavy_null_{args.tier}_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nScritto {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
