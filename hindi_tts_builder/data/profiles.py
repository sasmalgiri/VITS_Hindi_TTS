"""Staged dataset profiles.

From one full training_set/{train,val,test}.csv, produce subsampled
copies named like `train_30h.csv`, `val_30h.csv`, `test_30h.csv` for
controlled fine-tuning experiments.

Why this matters: the best-sounding model is often 30 h or 60 h, not
100 h. Same model fine-tuned on 30 h of clean data routinely beats the
same model on 100 h of mixed quality. So we want to compare apples to
apples on the same corpus, just at different sizes.

Sampling strategy:
  - random.sample(seed=...) for reproducibility
  - source-stratified: try to draw proportionally from each source_id
    so smaller profiles still cover the diversity of the corpus
  - hours-targeted: keep adding clips until cumulative duration ≥ target
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv
import random


PROFILES = {
    "2h_clean":   2.0,
    "10h_clean":  10.0,
    "30h_clean":  30.0,
    "60h_clean":  60.0,
    "100h_clean": 100.0,
}


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="|")
        rows = list(r)
        return r.fieldnames or [], rows


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter="|")
        w.writeheader()
        w.writerows(rows)


def _stratified_sample(rows: list[dict], target_hours: float, seed: int) -> list[dict]:
    """Draw rows source-stratified until cumulative duration ≥ target_hours.

    Each source contributes proportionally to its share of the full corpus.
    If a source runs out before its quota is met, the deficit is taken from
    other sources (round-robin).
    """
    target_seconds = target_hours * 3600.0
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r.get("source_id", "")].append(r)

    rng = random.Random(seed)
    for src_rows in by_src.values():
        rng.shuffle(src_rows)

    total_rows = sum(len(v) for v in by_src.values())
    if total_rows == 0:
        return []

    src_quotas: dict[str, float] = {}
    for src, src_rows in by_src.items():
        src_quotas[src] = (len(src_rows) / total_rows) * target_seconds

    picked: list[dict] = []
    used_seconds = 0.0
    src_pointers: dict[str, int] = {s: 0 for s in by_src}
    src_used: dict[str, float] = {s: 0.0 for s in by_src}

    # Pass 1: respect per-source quotas.
    for src, src_rows in by_src.items():
        while src_pointers[src] < len(src_rows) and src_used[src] < src_quotas[src]:
            row = src_rows[src_pointers[src]]
            try:
                d = float(row.get("duration", 0.0) or 0.0)
            except ValueError:
                d = 0.0
            picked.append(row)
            used_seconds += d
            src_used[src] += d
            src_pointers[src] += 1

    # Pass 2: top up from any source that still has rows, until target met.
    src_keys = list(by_src.keys())
    rr = 0
    safety_iters = total_rows * 2
    while used_seconds < target_seconds and safety_iters > 0:
        src = src_keys[rr % len(src_keys)]
        rr += 1
        safety_iters -= 1
        if src_pointers[src] >= len(by_src[src]):
            continue
        row = by_src[src][src_pointers[src]]
        try:
            d = float(row.get("duration", 0.0) or 0.0)
        except ValueError:
            d = 0.0
        picked.append(row)
        used_seconds += d
        src_pointers[src] += 1

    return picked


def make_profile(
    project_root: Path,
    profile_name: str,
    *,
    seed: int = 1234,
) -> dict:
    """Materialise one profile as `train_<profile>.csv`, etc.

    Returns a summary dict with hours achieved per split.
    Raises KeyError if profile_name is not in PROFILES.
    """
    if profile_name not in PROFILES:
        raise KeyError(f"Unknown profile {profile_name!r}. Known: {list(PROFILES)}")
    target_hours = PROFILES[profile_name]
    ts = project_root / "training_set"
    summary = {"profile": profile_name, "target_hours": target_hours, "splits": {}}

    for split in ("train", "val", "test"):
        src = ts / f"{split}.csv"
        if not src.exists():
            summary["splits"][split] = {"hours": 0.0, "rows": 0, "skipped": "missing source CSV"}
            continue
        header, rows = _read_csv(src)

        # Scale val/test proportionally to train target so the held-out sets
        # don't get bigger than the train set on small profiles.
        full_train_path = ts / "train.csv"
        if split == "train":
            split_target = target_hours
        else:
            full_train_hours = sum(float(r.get("duration", 0.0) or 0.0) for r in _read_csv(full_train_path)[1]) / 3600.0
            ratio = target_hours / full_train_hours if full_train_hours > 0 else 1.0
            full_split_hours = sum(float(r.get("duration", 0.0) or 0.0) for r in rows) / 3600.0
            split_target = min(full_split_hours, full_split_hours * ratio)

        sampled = _stratified_sample(rows, split_target, seed=seed + hash(split) % 1000)
        out = ts / f"{split}_{profile_name}.csv"
        _write_csv(out, header, sampled)
        achieved_hours = sum(float(r.get("duration", 0.0) or 0.0) for r in sampled) / 3600.0
        summary["splits"][split] = {
            "rows": len(sampled),
            "hours": round(achieved_hours, 3),
            "out": str(out.relative_to(project_root)),
        }
    return summary


def list_profiles() -> dict[str, float]:
    """Public read of the profile registry."""
    return dict(PROFILES)
