# Segmentation, timeline health, and the pre-train gate

Added 2026-08-18 after diagnosing why two trained models both dropped words.

## The short version

Clip boundaries were wrong, and nothing measured it. Three defects compounded:

1. **The SRT timelines were fabricated.** Integer-second timestamps, an exactly
   1.000s gap between all 11,737 cue transitions, and audio energy in those gaps
   within 0.3 dB of the speech. The narration is continuous; the timeline was
   generated on a fixed grid. Cutting on it discarded **3.26 hours** of speech
   and left every clip's text describing ~1s more audio than the clip contained.
2. **Clips did not end where sentences end** — 7.3% sentence-terminated. But not
   for the obvious reason: sentences average **~3.0s** while cues are **7.0s**,
   so each cue held **2.09–2.29** terminators. 12,202 sentence boundaries sat
   *inside* cues, unreachable at cue level.
3. **QC never ran.** `--skip-qc` wrote a report with `passed=1` and `snr_db=0.00`
   on all 10,280 rows, indistinguishable from a real one. Configured thresholds
   (`min_snr_db: 18.0`) were never applied.

Together these explain the symptoms: word-dropping, truncation, the Parler 2h
fine-tune scoring worse CER (0.347) than its own base (0.231), and VITS
plateauing at step 4,900 and never improving across 52,600 further steps.

## Where timed words come from — this is the decision that matters

Two ways to get word-level timings, and the wrong one is much worse than useless.
Measured on this corpus:

| source of timed text | sentence-terminated | speed |
|---|---|---|
| original SRT cue boundaries | 7.3% | — |
| Whisper large-v3 re-transcription | **0.0%** | 2.4× realtime |
| **forced alignment of the existing SRT text** | **85.9%** | **86.6× realtime** |

Whisper transcribing this Hindi narration emitted **no sentence punctuation at
all**, so the splitter had nothing to split on. The SRTs already carried 15–18
terminators per 1000 characters. The text was never the problem — only the
timings were.

**Rule: if a transcript exists, force-align it. Only use ASR when there is no
text at all.** `prepare` now does this automatically: Stage 1.5a force-aligns
sources that have transcripts, Stage 1.5b runs ASR on those that don't.

```bash
hindi-tts-builder force-align <name>     # recover timings for existing text
hindi-tts-builder transcribe <name>      # ASR, only when there is no text
```

## Check before you cut

```bash
hindi-tts-builder srt-health projects/<name>/sources/transcripts/
python scripts/srt_dry_merge.py projects/<name>/sources/transcripts/
```

`srt-health` exits non-zero on a timeline that should not be trusted, and now
also reports **punctuation density**, which predicts segmentability before any
GPU time is spent:

```
punctuation: 0.01 terminators/1k chars, 0.00 per cue
  -> mode: none (unpunctuated - needs punctuation restoration or exclusion)
```

Read it like this:

| terminators / 1k chars | meaning |
|---|---|
| < 2 | **unsegmentable** — no mode helps; restore punctuation or exclude |
| 15–18 | healthy Hindi narration |

| terminators per cue | mode to use |
|---|---|
| > 1.2 | `aligned_words` — sentences are shorter than cues |
| ≤ 1.2 | `sentence` — cue merging can reach the boundaries |

This check costs milliseconds and would have predicted, upfront, that
`src_6C-6RvEkqxg` (1 terminator in 1,285 cues) could never be sentence-segmented.

To drop a source without hand-editing the manifest:

```bash
hindi-tts-builder sources <name>                        # list, with health per source
hindi-tts-builder sources <name> --exclude src_XXXX     # skip it everywhere
hindi-tts-builder sources <name> --include src_XXXX     # put it back
```

Excluding sets a flag; audio and word timings stay on disk, so it is reversible.

## One writer at a time

`prepare` takes a lock on the project (`.pipeline.lock`). Two concurrent
pipelines interleave ffmpeg writes on the same clip paths and produce corrupt
WAVs — and because `skip_existing` treats a corrupt file as done, the corruption
survives every subsequent resume. This happened; the only clean recovery was
deleting every clip and recutting. A lock whose owning process is gone is treated
as stale and taken over, so a crash does not wedge the project.

## The three modes

| Mode | Cuts at | Use when |
|---|---|---|
| `cue` | every SRT cue (legacy default) | reproducing existing behaviour |
| `sentence` | merged cue runs ending on a terminator | sentences are **longer** than cues |
| `aligned_words` | ASR word timings | sentences are **shorter** than cues, or there is no SRT |

Set in `config.yaml` under `segmentation.mode`, or per-run with `prepare --mode`.

`cue` stays the default so no existing project silently changes shape. In `cue`
mode clip filenames keep using the SRT's own cue index, not a list position — a
non-sequential SRT would otherwise rename an entire corpus.

## No transcript at all

```bash
hindi-tts-builder new my_voice
hindi-tts-builder add-url my_voice "https://youtu.be/VIDEO_ID"
hindi-tts-builder prepare my_voice --mode aligned_words
hindi-tts-builder gate my_voice
```

`prepare` runs Stage 1.5 (Whisper with `word_timestamps=True`) automatically when
a source has no SRT or the mode needs word timings. Timings come from the
waveform, so there is no fabricated timeline to inherit.

Two guards run at transcription time, because the transcript becomes the training
target and its errors are permanent:

- Typographic characters Whisper emits (`”`, `—`, `…`) are **folded** to the
  trainer's vocabulary. Without this, `_preflight_text_compat` raises hours into
  a run.
- Latin-script output is **measured and reported, never dropped**. Whisper writes
  English words in Latin for Hinglish audio; those need transliteration, not
  deletion. Over 2% triggers a warning.

## Reclaiming fabricated gaps

`close_gaps()` extends each cue's end to the next cue's start, recovering audio a
naive cut would discard. It runs automatically in `sentence` mode when the
timeline looks gridded. Gaps wider than `max_close_seconds` (default 2.0s) are
left alone — those are plausibly real pauses, and swallowing them would attach
unlabelled audio to a clip.

On this corpus it reclaimed 195.5 minutes, taking trainable audio 22.70h → 25.95h.

## Provenance

Each source records `segmentation_policy`, `segmentation_fingerprint` (a hash over
every cut-determining parameter), `transcript_origin`, and `qc_mode`. Re-running
with different settings against existing clips flags the source as a **conflict**
and skips it rather than mixing two policies in one corpus. It never aborts the
run — that would make `corpus-ingest --watch` loop forever.

To recut deliberately:

```bash
hindi-tts-builder resegment <name> --mode aligned_words --force
```

`--force` deletes each source's existing clips first. Without it, conflicted
sources are skipped and reported.

## The gate

```bash
hindi-tts-builder gate <name> [--profile 30h_clean]
```

`train` runs this automatically and refuses to start on a blocker. Override with
`train --skip-gate`, which is a deliberate choice to spend GPU time on a corpus
with a known defect.

Blockers: QC never ran · sentence-terminated ratio below half the threshold ·
mixed segmentation policies · flagged conflicts · missing/empty training CSV ·
below `min_hours` · dead clip paths · characters outside the trainer vocabulary.

Warnings: QC ran without Whisper CER · terminator ratio merely low · no
provenance recorded (a corpus predating this tracking).

Tuning lives under `gate:` in `config.yaml`
(`min_sentence_terminator_ratio`, `require_real_qc`, `min_hours`).

## QC honesty

A passthrough report now leaves metric columns **empty** rather than writing
`0.00`, sets `reason=qc_skipped`, adds a `qc_mode` column, and writes
`training_set/qc_report_meta.json`. The gate reads that sidecar, and falls back to
the legacy signature (every row `reason=qc_skipped`) for older corpora.

Per ROADMAP Rule 1: for known-clean audio, **loosen the QC thresholds — do not
disable QC.** Disabling it is what made a corpus with three measurable defects
look ready to train.
