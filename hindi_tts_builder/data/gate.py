"""Pre-train gate: refuse to spend GPU-days on a corpus with a known defect.

Every blocker here corresponds to something that actually happened on this
project and cost real time:

* **QC never ran.** All 10,280 rows of ``qc_report.csv`` read ``passed=1`` with
  ``snr_db=0.0`` and no CER, because ``--skip-qc`` wrote a passthrough report
  that was indistinguishable from a real one. Configured thresholds
  (``min_snr_db: 18.0``) were never enforced.
* **Clips do not end where sentences end.** 7.3% sentence-terminated. Two models
  trained on it, both dropping words; the Parler fine-tune scored worse CER than
  its own base model.
* **A corpus cut two different ways.** Nothing recorded which policy produced
  which clips, so mixing was undetectable after the fact.

The gate reports rather than raises, so a caller can print everything wrong at
once. ``force=True`` demotes blockers to warnings for a deliberate override.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from hindi_tts_builder.data.manifest import Manifest
from hindi_tts_builder.utils.project import ProjectPaths
from hindi_tts_builder.utils.text_compat import untrainable_chars

QC_META_FILENAME = "qc_report_meta.json"


@dataclass
class GateResult:
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    forced: bool = False

    @property
    def ok(self) -> bool:
        return not self.blockers

    def describe(self) -> str:
        lines = []
        if self.stats:
            lines.append("corpus:")
            for k, v in self.stats.items():
                lines.append(f"    {k}: {v}")
        for b in self.blockers:
            lines.append(f"  BLOCKER  {b}")
        for w in self.warnings:
            lines.append(f"  warning  {w}")
        if self.ok:
            lines.append("  gate PASSED" + (" (forced)" if self.forced else ""))
        else:
            lines.append(f"  gate FAILED — {len(self.blockers)} blocker(s)")
        return "\n".join(lines)


def read_qc_mode(paths: ProjectPaths) -> str:
    """Determine whether QC actually scored anything.

    Prefers the sidecar written alongside the report. Falls back to the legacy
    signature: a passthrough report has every row's reason set to ``qc_skipped``.
    Returns "full", "no_whisper", "skipped", or "missing".
    """
    meta = paths.training_set / QC_META_FILENAME
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get("qc_mode", "missing")
        except Exception:
            pass

    report = paths.training_set / "qc_report.csv"
    if not report.exists():
        return "missing"
    with report.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "missing"
    if all((r.get("reason") or "") == "qc_skipped" for r in rows):
        return "skipped"
    # A report where no row carries an SNR reading never measured anything.
    if all(not (r.get("snr_db") or "").strip() or r.get("snr_db") == "0.00" for r in rows):
        return "skipped"
    if all(not (r.get("whisper_cer") or "").strip() for r in rows):
        return "no_whisper"
    return "full"


def iter_data_rows(train_csv: Path):
    """Yield data rows from a pipe-delimited training CSV, skipping the header.

    `dataset.py` writes a header row (`audio_path|raw_text|processed_text|...`)
    and `train/dataset.py` skips it. Anything auditing the same file must skip it
    too, or the header's own text is scored as if it were a training sample.
    """
    with train_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="|")
        first = next(reader, None)
        if first is None:
            return
        if first and first[0].strip() != "audio_path":
            yield first  # headerless file: the first row is data
        for row in reader:
            yield row


def _terminator_ratio_from_csv(train_csv: Path) -> tuple[float, int, float]:
    """Sentence-terminated share, row count, and total hours from a train CSV."""
    from hindi_tts_builder.data.cue_merge import ends_with_terminator

    n = term = 0
    hours = 0.0
    for row in iter_data_rows(train_csv):
        if len(row) < 4:
            continue
        n += 1
        if ends_with_terminator(row[1]):
            term += 1
        try:
            hours += float(row[3])
        except ValueError:
            pass
    return (term / n if n else 0.0), n, hours / 3600.0


def check_corpus(
    project_root: Path,
    *,
    min_sentence_terminator_ratio: float = 0.60,
    require_real_qc: bool = True,
    allow_no_whisper_qc: bool = False,
    min_hours: float = 1.0,
    profile: str | None = None,
    force: bool = False,
) -> GateResult:
    """Inspect a prepared corpus and report everything that should block training."""
    paths = ProjectPaths(project_root)
    res = GateResult(forced=force)

    name = f"train_{profile}.csv" if profile else "train.csv"
    train_csv = paths.training_set / name
    if not train_csv.exists():
        res.blockers.append(f"no training CSV at training_set/{name} — run `prepare` first")
        return _finalize(res, force)

    ratio, n_rows, hours = _terminator_ratio_from_csv(train_csv)
    res.stats["csv"] = name
    res.stats["clips"] = n_rows
    res.stats["hours"] = f"{hours:.2f}"
    res.stats["sentence_terminated"] = f"{100 * ratio:.1f}%"

    if n_rows == 0:
        res.blockers.append(f"training_set/{name} has no usable rows")

    if hours < min_hours:
        res.blockers.append(f"only {hours:.2f}h of training audio (need >= {min_hours}h)")

    # --- QC honesty -------------------------------------------------------
    qc_mode = read_qc_mode(paths)
    res.stats["qc_mode"] = qc_mode
    if qc_mode in ("skipped", "missing") and require_real_qc:
        res.blockers.append(
            f"QC did not run (qc_mode={qc_mode}). Every clip is marked passed without "
            f"being scored — SNR, silence ratio and CER were never measured. Re-run "
            f"`prepare` without --skip-qc, loosening thresholds in config.yaml if needed."
        )
    elif qc_mode == "no_whisper":
        # Previously only a warning. That let `require_real_qc: true` pass a corpus
        # where text/audio agreement — the single thing that broke the last two
        # models — was never measured on a single clip. SNR and silence cannot
        # detect misalignment; only CER can.
        msg = (
            "QC ran WITHOUT Whisper CER, so text/audio agreement was never measured on "
            "any clip and max_cer_vs_whisper was never applied. SNR and silence checks "
            "cannot detect misalignment, which is the defect that broke the previous two "
            "models. Re-run `prepare` without --no-whisper-qc."
        )
        if require_real_qc and not allow_no_whisper_qc:
            res.blockers.append(msg)
        else:
            res.warnings.append(msg)

    # --- sentence alignment ----------------------------------------------
    if ratio < min_sentence_terminator_ratio:
        msg = (
            f"only {100 * ratio:.1f}% of clips end on a sentence terminator "
            f"(need >= {100 * min_sentence_terminator_ratio:.0f}%). Clips that start and "
            f"end mid-phrase teach the model that utterances have no boundaries, which "
            f"shows up as word-dropping and truncation at inference."
        )
        if ratio < min_sentence_terminator_ratio / 2:
            res.blockers.append(msg)
        else:
            res.warnings.append(msg)

    # --- segmentation provenance -----------------------------------------
    manifest = Manifest(paths.sources / "manifest.json")
    policies = {s.segmentation_policy for s in manifest if s.status.segmented}
    conflicted = [s.id for s in manifest if s.segmentation_state == "conflict"]
    res.stats["segmentation_policies"] = sorted(p or "unknown" for p in policies) or ["none"]

    if conflicted:
        res.blockers.append(
            f"{len(conflicted)} source(s) flagged as policy conflicts: {conflicted[:5]}. "
            f"Their clips were cut under different settings than currently configured."
        )
    if len({p for p in policies if p}) > 1:
        res.blockers.append(
            f"corpus mixes segmentation policies {sorted(p for p in policies if p)} — "
            f"clips were cut two different ways. Re-run `resegment` to unify."
        )
    if policies == {None}:
        res.warnings.append(
            "no segmentation provenance recorded (corpus predates policy tracking); "
            "cannot verify all clips were cut the same way"
        )

    # --- dead references and untrainable text -----------------------------
    missing_audio = 0
    checked_audio = 0
    bad_chars: set[str] = set()
    for row in iter_data_rows(train_csv):
        if len(row) < 3:
            continue
        if checked_audio < 400:  # sampled: a full stat() sweep over 10k clips is slow
            checked_audio += 1
            if not (paths.root / row[0]).exists():
                missing_audio += 1
        bad_chars |= untrainable_chars(row[2])
    if missing_audio:
        res.blockers.append(
            f"{missing_audio} of {checked_audio} sampled clip paths do not exist on disk — "
            f"the CSV references audio that was moved or deleted"
        )
    if bad_chars:
        res.stats["untrainable_chars"] = "".join(sorted(bad_chars))
        res.blockers.append(
            f"processed_text contains {len(bad_chars)} character(s) outside the trainer "
            f"vocabulary: {sorted(bad_chars)!r}. Training would abort at the pre-flight "
            f"check. Fold them in the frontend first."
        )

    return _finalize(res, force)


def _finalize(res: GateResult, force: bool) -> GateResult:
    if force and res.blockers:
        res.warnings.extend(f"(forced past) {b}" for b in res.blockers)
        res.blockers = []
    return res
