#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cicli "copertura 1..90" su tutte le ruote (tutti i numeri non vuoti di un'estrazione).

Definizione di ciclo: partendo da insieme vuoto, ad ogni estrazione si aggiungono
tutti i numeri usciti su tutte le colonne numeriche del wide CSV; il ciclo termina
la prima volta che sono comparsi almeno una volta tutti i valori in {1..90}.

Regola **uniforme** per trasformare il ciclo in una stringa lunga e deterministica:
  - `canonical_cycle`: per ogni estrazione del ciclo, in ordine cronologico, la
    stringa dei numeri di quell'estrazione ordinati e separati da virgola; estrazioni
    separate da '|'. Poi SHA-256 → esadecimale (come espansione simbolica fissa).
  - `digit_draw`: cifra 0..9 da ogni estrazione: (somma di tutti i numeri dell'estrazione) mod 10.

Nota matematica: questa costruzione non genera un numero "come π" nel senso
trascendente / normale; produce una sequenza **deterministica** dai dati. Se le
estrazioni sono vicine al caso, lo stream assomiglierà al rumore. Il "senso sul
ciclo" è: ogni segmento è delimitato da un evento di copertura ben definito.

Uso: python scripts/coupon_cycle_stream.py
Output: data/analysis/coupon_cycles/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

META_COLS = {"year", "date", "draw_index_year", "num_wheels"}


def row_numbers(row: dict, keys: list[str]) -> list[int]:
    out: list[int] = []
    for k in keys:
        v = row.get(k, "")
        if v == "" or v is None:
            continue
        out.append(int(v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=1871)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    wide = root / "data" / "draws_wide.csv"
    if not wide.is_file():
        print("Manca data/draws_wide.csv", file=sys.stderr)
        return 1

    out_dir = root / "data" / "analysis" / "coupon_cycles"
    out_dir.mkdir(parents=True, exist_ok=True)

    with wide.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        keys = [k for k in r.fieldnames or [] if k not in META_COLS]

        cycles: list[dict] = []
        seen: set[int] = set()
        cycle_draws: list[tuple[str, str, list[int]]] = []
        cycle_id = 0
        digit_lines: list[str] = []
        sha_lines: list[str] = []

        for row in r:
            if int(row["year"]) < args.from_year:
                continue
            date = row["date"]
            idx = row["draw_index_year"]
            nums = row_numbers(row, keys)
            if not nums:
                continue

            parts_for_canon = ",".join(str(x) for x in sorted(nums))
            cycle_draws.append((date, idx, nums))

            for x in nums:
                if 1 <= x <= 90:
                    seen.add(x)

            s = sum(nums)
            digit_lines.append(f"{cycle_id},{date},{idx},{s % 10}")

            if len(seen) == 90:
                canon = "|".join(
                    ",".join(str(x) for x in sorted(ns)) for _, _, ns in cycle_draws
                )
                h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
                first = cycle_draws[0]
                last = cycle_draws[-1]
                cycles.append(
                    {
                        "cycle_id": cycle_id,
                        "n_draws": len(cycle_draws),
                        "start_date": first[0],
                        "start_draw_index": first[1],
                        "end_date": last[0],
                        "end_draw_index": last[1],
                        "sha256_hex": h,
                    }
                )
                sha_lines.append(h)
                digit_lines[-1] += ",1"  # marca chiusura ciclo su ultima riga
                cycle_id += 1
                seen = set()
                cycle_draws = []
            else:
                digit_lines[-1] += ",0"

    # ciclo aperto a fine file
    incomplete = None
    if cycle_draws:
        incomplete = {
            "cycle_id": cycle_id,
            "n_draws": len(cycle_draws),
            "distinct_seen": sorted(seen),
            "missing_count": 90 - len(seen),
        }

    (out_dir / "cycles_completed.json").write_text(
        json.dumps(
            {
                "definition": "Ciclo = tempo fino a che compaiono tutti i numeri 1..90 almeno una volta (pool: tutte le ruote, estrazione per estrazione).",
                "from_year": args.from_year,
                "n_completed_cycles": len(cycles),
                "incomplete_tail": incomplete,
                "cycles": cycles,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with (out_dir / "digit_per_draw.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("cycle_id,date,draw_index_year,sum_mod10,cycle_closed\n")
        f.write("\n".join(digit_lines) + "\n")

    with (out_dir / "cycle_sha256_one_per_line.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(sha_lines) + "\n")

    stream_path = out_dir / "concat_digit_stream.txt"
    digits_only: list[str] = []
    for line in digit_lines:
        parts = line.split(",")
        if len(parts) >= 5:
            digits_only.append(parts[3])
    stream_path.write_text("".join(digits_only), encoding="utf-8")

    print(f"Cicli completati: {len(cycles)}")
    if incomplete:
        print(
            f"Coda incompleta: visti {len(seen)}/90 numeri, "
            f"mancano {incomplete['missing_count']}"
        )
    print(f"Output in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
