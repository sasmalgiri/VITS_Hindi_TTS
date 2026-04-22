"""Stage 5: Build train/val/test splits from QC-passed clips.

Reads `training_set/qc_report.csv`, takes only passed clips, runs each clip's
transcript through the HindiFrontend, and writes three CSVs under
`training_set/`:

    train.csv   (95%)
    val.csv     (4%)
    test.csv    (1%)

Each row has:
    audio_path | raw_text | processed_text | duration | source_id

`processed_text` is what the training loop feeds to the model tokenizer.
`raw_text` is kept for diagnostics so you can always recover the original.

Splits are stable across runs: a fixed hash of clip_id decides which split a
clip falls into. Adding new clips never moves existing clips between splits.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json

from hindi_tts_builder.frontend.pipeline import HindiFrontend
from hindi_tts_builder.utils import get_logger
from hindi_tts_builder.utils.project import ProjectPaths


@dataclass
class DatasetRow:
    audio_path: str
    raw_text: str
    processed_text: str
    duration: float
    source_id: str


def _split_for(clip_id: str, val_pct: float, test_pct: float) -> str:
    """Deterministic split assignment based on hash of clip_id."""
    h = hashlib.sha1(clip_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF  # 0..1
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


def build_training_set(
    paths: ProjectPaths,
    *,
    frontend: HindiFrontend | None = None,
    val_pct: float = 0.04,
    test_pct: float = 0.01,
    logger=None,
) -> dict:
    """Build train/val/test CSVs. Returns count summary."""
    log = logger or get_logger("data.dataset", paths.logs / "dataset.log")

    qc_report = paths.training_set / "qc_report.csv"
    if not qc_report.exists():
        raise FileNotFoundError(
            f"QC report not found at {qc_report}. Run `quality_filter` first."
        )

    frontend = frontend or HindiFrontend()

    splits: dict[str, list[DatasetRow]] = {"train": [], "val": [], "test": []}

    with qc_report.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["passed"] != "1":
                continue
            clip_id = row["clip_id"]
            source_id = row["source_id"]
            clip_wav = paths.aligned / source_id / f"{clip_id}.wav"
            clip_txt = paths.aligned / source_id / f"{clip_id}.txt"
            if not clip_wav.exists() or not clip_txt.exists():
                log.warning(f"[skip] {clip_id}: missing wav or txt file")
                continue

            raw_text = clip_txt.read_text(encoding="utf-8").strip()
            processed = frontend(raw_text)
            if not processed:
                continue

            rel_wav = str(clip_wav.relative_to(paths.root))
            split = _split_for(clip_id, val_pct, test_pct)
            splits[split].append(DatasetRow(
                audio_path=rel_wav,
                raw_text=raw_text,
                processed_text=processed,
                duration=float(row["duration"]),
                source_id=source_id,
            ))

    # Write CSVs
    for name, rows in splits.items():
        out = paths.training_set / f"{name}.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="|")
            w.writerow(["audio_path", "raw_text", "processed_text", "duration", "source_id"])
            for r in rows:
                w.writerow([r.audio_path, r.raw_text, r.processed_text, f"{r.duration:.3f}", r.source_id])

    # Write vocabulary snapshot — all unique tokens seen in processed_text
    vocab: set[str] = set()
    for rows in splits.values():
        for r in rows:
            vocab.update(r.processed_text.split())
    vocab_path = paths.training_set / "vocabulary.json"
    vocab_path.write_text(
        json.dumps(sorted(vocab), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {k: len(v) for k, v in splits.items()}
    summary["total"] = sum(summary.values())
    summary["vocab_size"] = len(vocab)
    total_hours = sum(
        sum(r.duration for r in rows) for rows in splits.values()
    ) / 3600.0
    summary["total_hours"] = round(total_hours, 2)

    log.info(
        f"dataset built: train={summary['train']} val={summary['val']} "
        f"test={summary['test']} total={summary['total']} "
        f"({total_hours:.2f}h, {summary['vocab_size']} vocab tokens)"
    )
    return summary
