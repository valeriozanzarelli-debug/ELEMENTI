#!/usr/bin/env python3
"""Analyze coupon-cycle digit stream and cycle lengths; permutation nulls."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("pip install numpy", file=sys.stderr)
    raise SystemExit(1)


def chi2_uniform_digits(d: np.ndarray, n_bins: int = 10) -> float:
    """Pearson chi2 vs discrete uniform on 0..9."""
    counts = np.bincount(d, minlength=n_bins)
    n = int(d.size)
    e = n / n_bins
    return float(((counts.astype(np.float64) - e) ** 2 / e).sum())


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    cdir = root / "data" / "analysis" / "coupon_cycles"
    stream_path = cdir / "concat_digit_stream.txt"
    json_path = cdir / "cycles_completed.json"
    if not stream_path.is_file() or not json_path.is_file():
        print("Esegui prima: python scripts/coupon_cycle_stream.py", file=sys.stderr)
        return 1

    raw = stream_path.read_text(encoding="utf-8").strip()
    digits = np.frombuffer(raw.encode(), dtype=np.uint8) - ord("0")
    if digits.size == 0 or digits.max() > 9 or digits.min() < 0:
        print("Stream cifre non valido", file=sys.stderr)
        return 1

    n = int(digits.size)
    chi2_d = chi2_uniform_digits(digits)
    # lag-1 autocorr on digits as continuous
    if n > 2:
        r1 = float(np.corrcoef(digits[:-1], digits[1:])[0, 1])
    else:
        r1 = 0.0

    half = n // 2
    c1 = np.bincount(digits[:half], minlength=10)
    c2 = np.bincount(digits[half:], minlength=10)
    tbl = np.vstack([c1, c2]).astype(np.float64)
    row_s = tbl.sum(axis=1)
    col_s = tbl.sum(axis=0)
    tot = tbl.sum()
    chi2_2h = 0.0
    for i in range(2):
        for j in range(10):
            e = row_s[i] * col_s[j] / tot
            if e > 0:
                chi2_2h += (tbl[i, j] - e) ** 2 / e

    rng = np.random.default_rng(42)
    # Null per chi2: i.i.d. Uniform(0..9) stessa lunghezza (la permutazione
    # lascerebbe invariato l'istogramma).
    null_chi2 = [
        chi2_uniform_digits(rng.integers(0, 10, size=n, dtype=np.int64))
        for _ in range(499)
    ]
    null_chi2_arr = np.asarray(null_chi2)
    pct_chi2 = float((null_chi2_arr >= chi2_d).mean())
    p_two = min(pct_chi2, 1 - pct_chi2) * 2
    p_two = min(1.0, p_two)

    null_r1 = []
    for _ in range(499):
        sh = rng.permutation(digits)
        null_r1.append(float(np.corrcoef(sh[:-1], sh[1:])[0, 1]))
    null_r1_arr = np.asarray(null_r1)
    pct_r1 = float((np.abs(null_r1_arr) >= abs(r1)).mean())

    data = json.loads(json_path.read_text(encoding="utf-8"))
    cycles = data.get("cycles") or []
    lengths = np.asarray([c["n_draws"] for c in cycles], dtype=np.float64)
    ids = np.arange(len(lengths), dtype=np.float64)
    len_mean = float(lengths.mean()) if lengths.size else 0.0
    len_std = float(lengths.std()) if lengths.size > 1 else 0.0
    if lengths.size > 5:
        rho_time = float(np.corrcoef(ids, lengths)[0, 1])
    else:
        rho_time = 0.0

    # permutation null for cycle lengths autocorr lag1
    if lengths.size > 10:
        dlen = lengths[1:] - lengths[:-1]
        ac_len = float(np.corrcoef(lengths[:-1], lengths[1:])[0, 1])
        null_ac = []
        for _ in range(299):
            sh = rng.permutation(lengths)
            null_ac.append(float(np.corrcoef(sh[:-1], sh[1:])[0, 1]))
        pct_ac = float(np.mean(np.asarray(null_ac) >= ac_len))
    else:
        ac_len = 0.0
        pct_ac = 1.0

    report = {
        "digit_stream": {
            "length": n,
            "chi2_uniform_0_to_9": round(chi2_d, 4),
            "chi2_df": 9,
            "lag1_autocorr": round(r1, 6),
            "two_halves_homogeneity_chi2": round(chi2_2h, 4),
            "two_halves_df": 9,
            "perm_null_chi2_iid_uniform_digits": {
                "reps": 499,
                "fraction_null_ge_observed": round(pct_chi2, 4),
                "two_sided_p_approx": round(p_two, 4),
            },
            "perm_null_lag1_autocorr_abs": {
                "reps": 499,
                "fraction_null_abs_ge_observed": round(pct_r1, 4),
            },
        },
        "cycle_lengths": {
            "n_completed_cycles": int(lengths.size),
            "mean": round(len_mean, 4),
            "std": round(len_std, 4),
            "min": int(lengths.min()) if lengths.size else None,
            "max": int(lengths.max()) if lengths.size else None,
            "corr_cycle_index_vs_length": round(rho_time, 6),
            "lag1_autocorr_lengths": round(ac_len, 6),
            "perm_null_length_lag1_ac": {
                "reps": 299,
                "fraction_null_ge_observed": round(pct_ac, 4),
            },
        },
        "interpretation": (
            "Chi2 vs Uniform(0..9) i.i.d.: confronto con stream simulati; lag1-autocorr reale vs "
            "permutazioni (stesso multiset). Corr(indice_ciclo, lunghezza) negativa spesso riflette "
            "cambi di formato storico (piu numeri per estrazione nel tempo => cicli piu corti), "
            "non un segnale casuale."
        ),
    }

    out = cdir / "coupon_cycle_analysis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nScritto {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
