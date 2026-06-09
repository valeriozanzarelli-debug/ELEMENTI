#!/usr/bin/env python3
"""
Parse Franknet lotto year HTML files into CSV + JSON for analysis.
Uses only the standard library.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

MONTH_MAP = {
    "gen": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "mag": 5,
    "giu": 6,
    "lug": 7,
    "ago": 8,
    "set": 9,
    "ott": 10,
    "nov": 11,
    "dic": 12,
}

WHEEL_ORDER = [
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


class PreCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_pre = False
        self._buf: list[str] = []
        self.pres: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "pre":
            self._in_pre = True
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "pre" and self._in_pre:
            self.pres.append("".join(self._buf))
            self._in_pre = False

    def handle_data(self, data: str) -> None:
        if self._in_pre:
            self._buf.append(data)


@dataclass
class DrawRow:
    year: int
    date_iso: str
    draw_index_year: int
    num_wheels: int
    wheels: dict[str, list[int | None]]


def extract_pre_blocks(html: str) -> list[str]:
    p = PreCollector()
    p.feed(html)
    return p.pres


def parse_draw_line(line: str, year: int) -> DrawRow | None:
    line = line.strip()
    if not line or line.startswith("</"):
        return None
    parts = line.split()
    if len(parts) < 8:
        return None
    if not parts[0].isdigit():
        return None
    mon = parts[1].lower()
    if mon not in MONTH_MAP:
        return None
    day = int(parts[0])
    month_num = MONTH_MAP[mon]
    rest = parts[2:]
    if not rest:
        return None
    if not rest[-1].isdigit():
        return None
    draw_idx = int(rest[-1])
    nums = rest[:-1]
    if len(nums) % 5 != 0:
        return None
    n_wheels = len(nums) // 5
    if n_wheels < 1 or n_wheels > 11:
        return None
    try:
        d = date(year, month_num, day)
    except ValueError:
        return None

    wheels: dict[str, list[int | None]] = {}
    for i in range(n_wheels):
        name = WHEEL_ORDER[i]
        chunk = nums[i * 5 : (i + 1) * 5]
        col: list[int | None] = []
        for x in chunk:
            if x == "--":
                col.append(None)
            else:
                col.append(int(x))
        wheels[name] = col
    for j in range(n_wheels, 11):
        wheels[WHEEL_ORDER[j]] = [None, None, None, None, None]

    return DrawRow(
        year=year,
        date_iso=d.isoformat(),
        draw_index_year=draw_idx,
        num_wheels=n_wheels,
        wheels=wheels,
    )


def parse_year_file(path: Path) -> list[DrawRow]:
    year = int(path.stem)
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[DrawRow] = []
    seen: set[tuple[str, int]] = set()

    for pre in extract_pre_blocks(text):
        for raw_line in pre.splitlines():
            row = parse_draw_line(raw_line, year)
            if row is None:
                continue
            key = (row.date_iso, row.draw_index_year)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    rows.sort(key=lambda r: (r.date_iso, r.draw_index_year))
    return rows


def row_to_wide_dict(r: DrawRow) -> dict:
    out: dict = {
        "year": r.year,
        "date": r.date_iso,
        "draw_index_year": r.draw_index_year,
        "num_wheels": r.num_wheels,
    }
    for w in WHEEL_ORDER:
        nums = r.wheels[w]
        for k in range(5):
            out[f"{w}_{k + 1}"] = nums[k] if nums[k] is not None else ""
    return out


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    years_dir = root / "years"
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not years_dir.is_dir():
        print("Missing years/ folder", file=sys.stderr)
        return 1

    all_rows: list[DrawRow] = []
    files = sorted(years_dir.glob("*.HTM"), key=lambda p: int(p.stem))
    for yf in files:
        all_rows.extend(parse_year_file(yf))

    wide_path = data_dir / "draws_wide.csv"
    long_path = data_dir / "draws_long.csv"
    json_path = data_dir / "draws.json"

    wide_fields = (
        ["year", "date", "draw_index_year", "num_wheels"]
        + [f"{w}_{i}" for w in WHEEL_ORDER for i in range(1, 6)]
    )

    with wide_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=wide_fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(row_to_wide_dict(r))

    with long_path.open("w", newline="", encoding="utf-8") as f:
        lw = csv.writer(f)
        lw.writerow(
            ["year", "date", "draw_index_year", "wheel", "n1", "n2", "n3", "n4", "n5"]
        )
        for r in all_rows:
            for wheel in WHEEL_ORDER:
                nums = r.wheels[wheel]
                if all(x is None for x in nums):
                    continue
                lw.writerow(
                    [
                        r.year,
                        r.date_iso,
                        r.draw_index_year,
                        wheel,
                        *["" if x is None else x for x in nums],
                    ]
                )

    json_rows = []
    for r in all_rows:
        d = {
            "year": r.year,
            "date": r.date_iso,
            "draw_index_year": r.draw_index_year,
            "num_wheels": r.num_wheels,
            "wheels": {
                k: v for k, v in r.wheels.items() if not all(x is None for x in v)
            },
        }
        json_rows.append(d)

    json_path.write_text(
        json.dumps(json_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "source": "https://www.franknet.altervista.org/lotto/page.php",
        "draw_rows": len(all_rows),
        "year_files": len(files),
        "wide_csv": "draws_wide.csv",
        "long_csv": "draws_long.csv",
        "json": "draws.json",
        "wheel_order": WHEEL_ORDER,
    }
    (data_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Parsed {len(all_rows)} draws from {len(files)} year files.")
    print(f"Written: {wide_path.name}, {long_path.name}, {json_path.name}, meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
