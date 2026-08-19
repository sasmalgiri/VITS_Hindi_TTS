"""Dry-run the timeline analysis + sentence merge over real SRTs. Cuts no audio.

This is the go/no-go before re-segmenting a corpus: it shows what the new
segmentation would produce, per source, without spending hours in ffmpeg.

    python scripts/srt_dry_merge.py <srt-dir-or-file> [--max-seconds 15] [--no-close-gaps]

Reports per file: the timeline verdict, then the before/after sentence-terminator
ratio — the single number that says whether clips end where sentences end.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hindi_tts_builder.data.cue_merge import (  # noqa: E402
    merge_cues_to_sentences,
    terminator_ratio,
    units_from_cues,
)
from hindi_tts_builder.data.srt_health import analyze_timeline, close_gaps  # noqa: E402
from hindi_tts_builder.utils.srt import parse_srt  # noqa: E402


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def summarize(name: str, units, label: str) -> dict:
    durs = [u.duration for u in units]
    if not durs:
        print(f"  {label:<10} (no units)")
        return {}
    over = sum(1 for d in durs if d > 15.0)
    print(
        f"  {label:<10} units={len(units):<6} terminated={pct(terminator_ratio(units))}"
        f"  dur med={median(durs):5.2f}s max={max(durs):6.2f}s  >15s={over}"
        f"  hours={sum(durs) / 3600:.2f}"
    )
    return {"units": len(units), "term": terminator_ratio(units), "hours": sum(durs) / 3600}


def run(path: Path, *, max_seconds: float, min_seconds: float, do_close: bool) -> dict:
    cues = parse_srt(path)
    print(f"\n=== {path.name}")
    rep = analyze_timeline(cues)
    for line in rep.describe().splitlines():
        print(f"  {line}")

    before = summarize(path.name, units_from_cues(cues), "BEFORE")

    work = cues
    if do_close and (rep.gaps_uniform or rep.gap_fraction > 0.05):
        work, reclaimed = close_gaps(cues)
        print(f"  close_gaps: reclaimed {reclaimed / 60:.1f} min of discarded audio")

    units, stats = merge_cues_to_sentences(
        work, min_seconds=min_seconds, max_seconds=max_seconds
    )
    after = summarize(path.name, units, "AFTER")
    print(
        f"             flushed_gap={stats['flushed_by_gap']} flushed_len={stats['flushed_by_length']}"
        f" split={stats['split_overlong']} oversized={stats['oversized_single']}"
        f" short_attached={stats['attached_short']} short_dropped={stats['dropped_short']}"
    )
    return {"before": before, "after": after}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--no-close-gaps", action="store_true")
    a = ap.parse_args()

    files = sorted(a.target.glob("*.srt")) if a.target.is_dir() else [a.target]
    if not files:
        print(f"no .srt found under {a.target}")
        raise SystemExit(1)

    results = [
        run(f, max_seconds=a.max_seconds, min_seconds=a.min_seconds, do_close=not a.no_close_gaps)
        for f in files
    ]

    ok = [r for r in results if r["before"] and r["after"]]
    if ok:
        bt = sum(r["before"]["units"] * r["before"]["term"] for r in ok) / sum(r["before"]["units"] for r in ok)
        at = sum(r["after"]["units"] * r["after"]["term"] for r in ok) / sum(r["after"]["units"] for r in ok)
        print("\n" + "=" * 66)
        print(f"CORPUS  sentence-terminated clips:  {pct(bt)}  ->  {pct(at)}")
        print(f"CORPUS  trainable hours:            {sum(r['before']['hours'] for r in ok):.2f}"
              f"  ->  {sum(r['after']['hours'] for r in ok):.2f}")


if __name__ == "__main__":
    main()
