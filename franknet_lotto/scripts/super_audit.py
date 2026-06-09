#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Super-audit: orchestra tutta la pipeline di analisi Franknet Lotto in fasi,
con livelli di intensita computazionale, e produce un report master unificato.

NON risolve il lotto: aggrega misure e null per una fotografia completa.

Tier:
  quick   — meno MC, salta mappe ML pesanti
  standard — default bilanciato
  full    — massima intensita (tempi lunghi)

Uso (dalla cartella franknet_lotto o con path assoluto):
  python scripts/super_audit.py --tier standard
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def run(
    py: str,
    script: Path,
    args: list[str],
    cwd: Path,
    timeout: int | None,
) -> tuple[int, float]:
    t0 = time.perf_counter()
    p = subprocess.run(
        [py, str(script)] + args,
        cwd=str(cwd),
        timeout=timeout,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    dt = time.perf_counter() - t0
    return p.returncode, dt


def load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("quick", "standard", "full"), default="standard")
    ap.add_argument("--skip-parse", action="store_true", help="Non rigenerare draws_wide")
    ap.add_argument("--skip-ml", action="store_true", help="Salta boosted_dr_predictor")
    ap.add_argument("--skip-digital-map", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    scripts = root / "scripts"
    py = sys.executable
    wide = root / "data" / "draws_wide.csv"

    tier = args.tier
    sieve_reps = {"quick": 199, "standard": 599, "full": 1999}[tier]
    heavy_tier = tier
    skip_ml = args.skip_ml or tier == "quick"
    skip_dig = args.skip_digital_map or tier == "quick"

    phases: list[dict] = []
    master: dict = {
        "audit_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "tier": tier,
        "root": str(root),
        "phases": phases,
        "artifacts": {},
        "disclaimer": (
            "Report aggregato: nessuna conclusione di prevedibilita senza validazione "
            "prospettica; i p-value sono quelli prodotti dagli script citati."
        ),
    }

    def phase(name: str, script_name: str, argv: list[str], timeout: int | None = None):
        sp = scripts / script_name
        if not sp.is_file():
            phases.append({"name": name, "status": "missing_script", "script": script_name})
            return
        code, dt = run(py, sp, argv, root, timeout)
        phases.append(
            {
                "name": name,
                "script": script_name,
                "args": argv,
                "exit_code": code,
                "seconds": round(dt, 3),
            }
        )
        if code != 0:
            phases[-1]["status"] = "failed"
        else:
            phases[-1]["status"] = "ok"

    # 0 — parse
    if not args.skip_parse or not wide.is_file():
        phase(
            "parse_draws",
            "parse_draws.py",
            [],
            timeout=600,
        )

    if not wide.is_file():
        print("draws_wide.csv mancante dopo parse.", file=sys.stderr)
        return 1

    # 1 — cicli coupon
    phase("coupon_cycle_stream", "coupon_cycle_stream.py", [], timeout=300)
    phase("analyze_coupon_cycles", "analyze_coupon_cycles.py", [], timeout=300)

    # 2 — first hit + chain
    phase("first_hit_chain_analysis", "first_hit_chain_analysis.py", [], timeout=600)

    # 3 — setaccio (numpy)
    phase(
        "structure_sieve_milano_n",
        "structure_sieve.py",
        ["--wheel", "MILANO", "--position", "1", "--mode", "n", "--from-year", "1900", "--null-reps", str(sieve_reps)],
        timeout=1800,
    )

    # 4 — heavy null (il piu pesante dopo sieve)
    phase(
        "heavy_null_engine",
        "heavy_null_engine.py",
        ["--tier", heavy_tier],
        timeout=3600,
    )

    # 5 — single cell + discover
    phase(
        "single_cell_series",
        "single_cell_derived_series.py",
        ["--wheel", "MILANO", "--position", "1", "--from-year", "1900", "--lag", "5"],
        timeout=300,
    )
    phase(
        "discover_structure",
        "discover_sequence_structure.py",
        ["--series-csv", str(root / "data/analysis/single_cell/MILANO_pos1_y1900_series.csv")],
        timeout=600,
    )

    # 6 — longterm windows
    phase("longterm_distribution", "longterm_distribution_windows.py", [], timeout=300)

    # 7 — digital root map (lento)
    if not skip_dig:
        to = {"quick": 600, "standard": 1200, "full": 2400}[tier]
        phase("digital_root_temporal_map", "digital_root_temporal_map.py", ["--wheel", "MILANO", "--from-year", "1900"], timeout=to)

    # 8 — ML boost
    if not skip_ml:
        phase(
            "boosted_dr_predictor",
            "boosted_dr_predictor.py",
            ["--wheel", "MILANO", "--target-position", "1", "--lag-draws", "5"],
            timeout={"standard": 900, "full": 1800}[tier] if tier != "quick" else 900,
        )

    # Raccogli JSON noti
    artifacts = {
        "coupon_cycle_analysis": load_json(root / "data/analysis/coupon_cycles/coupon_cycle_analysis.json"),
        "first_hit_summary": load_json(root / "data/analysis/first_hit_chain/summary.json"),
        "heavy_null": load_json(root / "data/analysis/heavy_null" / f"heavy_null_{heavy_tier}_report.json"),
        "structure_sieve": load_json(
            root / "data/analysis/structure_sieve/milano_pos1_n_y1900_sieve_report.json"
        ),
        "discover_structure": load_json(
            root / "data/analysis/single_cell/MILANO_pos1_y1900_structure_report.json"
        ),
        "ml_boosted": load_json(
            root / "data/analysis/ml_models/milano_pos1_lag5_boosted_report.json"
        ),
        "digital_root_summary": load_json(
            root / "data/analysis/digital_root/milano/summary.json"
        ),
    }
    master["artifacts"] = {k: v for k, v in artifacts.items() if v is not None}
    master["finished_utc"] = datetime.now(timezone.utc).isoformat()
    master["phases_summary"] = {
        "ok": sum(1 for p in phases if p.get("status") == "ok"),
        "failed": sum(1 for p in phases if p.get("status") == "failed"),
        "total_seconds": round(sum(p.get("seconds", 0) for p in phases), 3),
    }

    out_dir = root / "data" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"MASTER_AUDIT_{tier}.json"
    jpath.write_text(json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown sintetico
    lines = [
        f"# Master audit ({tier})",
        "",
        f"- Inizio: {master['started_utc']}",
        f"- Fine: {master['finished_utc']}",
        f"- Fasi OK: {master['phases_summary']['ok']}, fallite: {master['phases_summary']['failed']}",
        f"- Tempo totale fasi: {master['phases_summary']['total_seconds']} s",
        "",
        "## Fasi",
        "",
        "| Fase | Script | Esito | Secondi |",
        "|------|--------|-------|---------|",
    ]
    for p in phases:
        lines.append(
            f"| {p.get('name','')} | {p.get('script','')} | {p.get('status','')} | {p.get('seconds','')} |"
        )
    lines += ["", f"JSON completo: `{jpath}`", "", master["disclaimer"]]
    mpath = out_dir / f"MASTER_AUDIT_{tier}.md"
    mpath.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(master["phases_summary"], indent=2))
    print(f"\nMaster: {jpath}")
    print(f"       {mpath}")

    failed = master["phases_summary"]["failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
