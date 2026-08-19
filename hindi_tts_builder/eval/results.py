"""Append-only benchmark ledger at ``eval/results.csv``.

Deliberately small. The point is not a results database — it is that every
evaluated model lands a row *automatically*, because the alternative is what
happened here: three models trained (VITS from scratch, Parler 2h, Parler 10h),
a results.csv with exactly one row, and model comparisons carried in memory.

Columns are fixed and append-only. A new metric gets a new column at the end;
existing rows keep whatever they had.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

COLUMNS = [
    "model", "backend", "data_hours", "steps", "mean_cer", "rtf",
    "manual_mos", "long_form_stable", "date", "notes",
]


def default_results_path() -> Path:
    """`eval/results.csv` at the repo root."""
    return Path(__file__).resolve().parents[2] / "eval" / "results.csv"


def append_result(
    *,
    model: str,
    backend: str,
    data_hours: float | str = "",
    steps: int | str = "",
    mean_cer: float | str = "",
    rtf: float | str = "",
    manual_mos: float | str = "",
    long_form_stable: str = "",
    notes: str = "",
    on: date | str | None = None,
    path: Path | None = None,
) -> Path:
    """Append one benchmark row, creating the file with a header if needed.

    Returns the path written. Existing rows are never rewritten.
    """
    p = path or default_results_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists() and p.stat().st_size > 0

    row = {
        "model": model,
        "backend": backend,
        "data_hours": data_hours,
        "steps": steps,
        "mean_cer": mean_cer,
        "rtf": rtf,
        "manual_mos": manual_mos,
        "long_form_stable": long_form_stable,
        "date": str(on) if on else date.today().isoformat(),
        "notes": notes,
    }
    with p.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)
    return p


def read_results(path: Path | None = None) -> list[dict]:
    """Read the ledger. Returns [] when it does not exist yet."""
    p = path or default_results_path()
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
