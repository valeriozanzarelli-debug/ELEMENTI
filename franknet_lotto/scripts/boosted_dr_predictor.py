#!/usr/bin/env python3
"""
Rich-feature gradient boosting on digital roots (one wheel, one target position).

Uses past L draws (all 5 numbers + their digital roots) as features to predict
the target position's digital root on the *next* draw. Chronological split and
optional sparse walk-forward re-fitting.

Install: pip install -r requirements_ml.txt

Does not imply usable edge for betting; compares against strong baselines.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.metrics import accuracy_score, log_loss
except ImportError as e:
    print(
        "Missing ML deps. Run: pip install -r requirements_ml.txt\n",
        e,
        file=sys.stderr,
    )
    raise SystemExit(1)


def digital_root(n: int) -> int:
    while n > 9:
        n = sum(int(c) for c in str(n))
    return n


def load_wheel(
    wide_csv: Path, wheel: str, from_year: int
) -> tuple[list[str], list[list[int]]]:
    dates: list[str] = []
    values: list[list[int]] = []
    cols = [f"{wheel}_{k}" for k in range(1, 6)]
    with wide_csv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row["year"]) < from_year:
                continue
            nums: list[int] = []
            ok = True
            for c in cols:
                v = row[c]
                if v == "":
                    ok = False
                    break
                nums.append(int(v))
            if not ok:
                continue
            dates.append(row["date"])
            values.append(nums)
    return dates, values


def build_matrix(
    values: list[list[int]], target_pos: int, lag_draws: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Rows aligned with draw index i >= lag_draws; y = dr(values[i][target_pos]) - 1 (0..8)."""
    n = len(values)
    dr_all: list[list[int]] = [
        [digital_root(x) for x in row] for row in values
    ]
    feat_names: list[str] = []
    for ell in range(1, lag_draws + 1):
        for p in range(5):
            feat_names.append(f"lag{ell}_p{p+1}_n")
            feat_names.append(f"lag{ell}_p{p+1}_dr")
    rows_x: list[list[float]] = []
    rows_y: list[int] = []
    for i in range(lag_draws, n):
        row: list[float] = []
        for ell in range(1, lag_draws + 1):
            past = values[i - ell]
            past_dr = dr_all[i - ell]
            for p in range(5):
                row.append(float(past[p]))
                row.append(float(past_dr[p]))
        rows_x.append(row)
        rows_y.append(dr_all[i][target_pos] - 1)
    X = np.asarray(rows_x, dtype=np.float64)
    y = np.asarray(rows_y, dtype=np.int64)
    return X, y, feat_names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", default="MILANO")
    ap.add_argument("--target-position", type=int, default=1, help="1..5")
    ap.add_argument("--from-year", type=int, default=1900)
    ap.add_argument("--lag-draws", type=int, default=5, help="How many past draws as features")
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--walkforward-refit",
        type=int,
        default=0,
        help="If >0, on last test segment re-fit every N samples (0=off)",
    )
    args = ap.parse_args()

    if not 1 <= args.target_position <= 5:
        print("target-position must be 1..5", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent.parent
    wide = root / "data" / "draws_wide.csv"
    if not wide.is_file():
        print("Missing data/draws_wide.csv", file=sys.stderr)
        return 1

    dates, values = load_wheel(wide, args.wheel, args.from_year)
    if len(values) < 200:
        print("Too few rows.", file=sys.stderr)
        return 1

    tp = args.target_position - 1
    X, y, feat_names = build_matrix(values, tp, args.lag_draws)
    if len(y) < 100:
        print("Too few samples after lag.", file=sys.stderr)
        return 1

    # Chronological split on X rows
    n_s = len(y)
    split_i = int(n_s * (1.0 - args.test_fraction))
    split_i = max(split_i, 150)

    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    if len(y_test) < 30:
        print("Test set too small; lower --test-fraction", file=sys.stderr)
        return 1

    hgb = HistGradientBoostingClassifier(
        max_depth=8,
        max_iter=400,
        learning_rate=0.06,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=args.random_state,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
    )
    hgb.fit(X_train, y_train)
    proba = hgb.predict_proba(X_test)
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba, labels=np.arange(9))

    rf = RandomForestClassifier(
        n_estimators=240,
        max_depth=16,
        min_samples_leaf=8,
        random_state=args.random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rproba = rf.predict_proba(X_test)
    rpred = np.argmax(rproba, axis=1)
    acc_rf = accuracy_score(y_test, rpred)
    ll_rf = log_loss(y_test, rproba, labels=np.arange(9))
    top_rf = sorted(
        zip(feat_names, rf.feature_importances_.tolist()),
        key=lambda x: -x[1],
    )[:25]

    dummy_mf = DummyClassifier(strategy="most_frequent")
    dummy_mf.fit(X_train, y_train)
    acc_mf = accuracy_score(y_test, dummy_mf.predict(X_test))

    dummy_st = DummyClassifier(strategy="stratified", random_state=args.random_state)
    dummy_st.fit(X_train, y_train)
    acc_d = accuracy_score(y_test, dummy_st.predict(X_test))

    uniform_ll = float(np.log(9.0))
    uniform_proba = np.full((len(y_test), 9), 1.0 / 9.0)
    ll_uni = log_loss(y_test, uniform_proba, labels=np.arange(9))

    imp = getattr(hgb, "feature_importances_", None)
    top_f: list[tuple[str, float]] = []
    if imp is not None:
        imp_flat = np.asarray(imp, dtype=np.float64).ravel()
        nf = min(imp_flat.size, len(feat_names))
        if nf > 0:
            top_f = sorted(
                zip(feat_names[:nf], imp_flat[:nf].tolist()),
                key=lambda x: -x[1],
            )[:25]

    report = {
        "wheel": args.wheel,
        "target_position": args.target_position,
        "from_year": args.from_year,
        "lag_draws": args.lag_draws,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "hist_gradient_boosting": {
            "accuracy": round(float(acc), 6),
            "log_loss": round(float(ll), 6),
        },
        "random_forest": {
            "accuracy": round(float(acc_rf), 6),
            "log_loss": round(float(ll_rf), 6),
            "top_feature_importances": [
                {"feature": a, "importance": round(float(b), 6)} for a, b in top_rf
            ],
        },
        "dummy_most_frequent_accuracy": round(float(acc_mf), 6),
        "dummy_stratified_marginals_accuracy": round(float(acc_d), 6),
        "uniform_proba_log_loss": round(float(ll_uni), 6),
        "uniform_random_guess_log_loss_theory": round(uniform_ll, 6),
        "boosting_log_loss_minus_uniform": round(float(ll - ll_uni), 6),
        "random_forest_log_loss_minus_uniform": round(float(ll_rf - ll_uni), 6),
        "hist_gboost_top_importances": [
            {"feature": a, "importance": round(float(b), 6)} for a, b in top_f
        ],
    }

    wf_extra: dict = {}
    if args.walkforward_refit > 0 and n_s > split_i + 250:
        # Expanding window: train on X[0:t], predict y[t]; refit every refit steps.
        hits = 0
        tot = 0
        ll_sum = 0.0
        model = HistGradientBoostingClassifier(
            max_depth=8,
            max_iter=250,
            learning_rate=0.08,
            min_samples_leaf=25,
            l2_regularization=0.15,
            random_state=args.random_state,
        )
        end = n_s
        refit = args.walkforward_refit
        eval_from = max(200, end - 600)
        last_fit = -1
        for t in range(eval_from, end):
            if last_fit < 0 or (t - last_fit) >= refit:
                model.fit(X[:t], y[:t])
                last_fit = t
            pr = model.predict_proba(X[t : t + 1])
            p0 = np.argmax(pr, axis=1)[0]
            hits += int(p0 == y[t])
            tot += 1
            ll_sum += float(-np.log(max(pr[0, y[t]], 1e-15)))
        if tot > 0:
            wf_extra["walkforward_tail"] = {
                "eval_points": tot,
                "accuracy": round(hits / tot, 6),
                "mean_log_loss": round(ll_sum / tot, 6),
                "refit_every": refit,
                "note": "Tail of full series; each step uses only past rows (no leakage).",
            }
    report.update(wf_extra)

    out_dir = root / "data" / "analysis" / "ml_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.wheel.lower()}_pos{args.target_position}_lag{args.lag_draws}"
    jpath = out_dir / f"{tag}_boosted_report.json"
    jpath.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWritten {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
