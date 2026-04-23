# Smoke Test — real-hardware end-to-end check

[`scripts/smoke_test.sh`](../scripts/smoke_test.sh) exercises every stage
of the pipeline against a tiny dataset so you know the stack works on
your machine before committing 5+ days to a real training run.

## What it tests

| Stage | What actually runs | Typical time on RTX 3060 12GB |
|---|---|---|
| `doctor` | env + CUDA + ffmpeg + yt-dlp + coqui-tts check | <5 s |
| `new` + `add-sources` | project creation, manifest write | <1 s |
| `prepare` — download | yt-dlp pulling audio from YouTube | 1–3 min per clip |
| `prepare` — align | WhisperX refines SRT timestamps on GPU | 30–90 s per clip |
| `prepare` — segment | ffmpeg per-cue clip cut + loudness-norm | ~10 s per clip |
| `prepare` — QC | Whisper round-trip CER filter | 1–2 min per clip |
| `prepare` — training set | tokenizer fit + train/val/test split | <1 s |
| `train` (1000 steps) | full Coqui VITS fit on GPU | 3–8 min |
| `export` | checkpoint + tokenizer + frontend → `engine/` | <1 s |
| `speak` | single-utterance generation + save WAV | 1–3 s |
| `serve` + `/health` | FastAPI boots and responds | <3 s |

**Total wall time for 2–3 short (3–5 min) clips: 15–30 min.**

## Prerequisites

Inside WSL Ubuntu-22.04 (or the Windows venv, with CUDA torch installed):

```bash
source venv/bin/activate
hindi-tts-builder doctor   # must show all ✓
```

You also need **at least one short Hindi YouTube video with a matching
SRT transcript**. Short (3–5 minutes) is much better than long — the
whole point is fast feedback.

## Running it

```bash
./scripts/smoke_test.sh \
    --urls  my_smoke_urls.txt \
    --srts  my_smoke_srts/
```

**Required args:**

- `--urls FILE` — one YouTube URL per line (comments and blank lines ok)
- `--srts DIR` — directory with matching `.srt` files, in the same sort
  order as the URLs

**Optional:**

- `--project NAME` — project name (default: `smoke_test`)
- `--steps N` — training steps cap (default: 1000). Drop to 100 for a
  really fast sanity check; raise to 5000 if you want to hear partial
  convergence.
- `--skip-train` — stop after `prepare`. Useful when you only want to
  check the data pipeline, not GPU training.
- `--skip-whisperx` — use SRT timestamps verbatim (faster; less
  accurate). Good for the very first attempt if WhisperX is flaky on
  your box.
- `--port PORT` — port for the `/health` probe (default: 8780)
- `--keep` — keep the project dir even on success (default behavior
  preserves it on failure for debugging and deletes on success)

**Exit codes:**

- `0` every stage passed
- `1` argument / setup error (before any stage ran)
- `2` a pipeline stage failed — **project dir is preserved** at
  `projects/<name>/` so you can inspect logs and intermediate files

## How to pick smoke test inputs

Good candidates for your URL + SRT pair:

- **Your own dubbing project source** — if you already have a
  3–5-minute clip with an accurate Hindi SRT, use it. The
  characteristics of your real data are what matter.
- **A short Hindi news/vlog clip** with community-generated Hindi
  captions (check "CC → Hindi" on the video).
- A trailer, interview excerpt, or podcast clip under 5 minutes.

Avoid:

- Long videos (wasted download time)
- Videos with auto-generated (ASR) English captions rather than Hindi
  SRTs (won't match the audio)
- Age-gated or region-locked content (yt-dlp will fail without a
  cookies file)

## What the output looks like

Successful run tail:

```
=== [17:38:02] 8/8 speak + serve probe ===
✓ generated projects/smoke_test/smoke_test.wav (48314 bytes)
✓ /health returned 200
SMOKE TEST PASSED — the full pipeline works on this machine.
✓ cleaned up projects/smoke_test
```

A failure looks like (stage name will differ):

```
=== [17:28:14] 6/8 train (1000 steps on GPU) ===
…Coqui stderr…
✗ train failed
project preserved at projects/smoke_test for debugging
```

Then `projects/smoke_test/logs/train.log` has the actual error.

## Known failure modes & where to look

| Symptom | Where to check |
|---|---|
| `doctor` shows ✗ for torch+CUDA | CPU-only torch installed. `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| `doctor` shows ✗ for coqui-tts | `transformers` skew. `pip install 'transformers>=4.57,<5'` |
| `yt-dlp` HTTP 403 / 429 | YouTube rate-limited this IP, or the video is geo/age-locked. Pick a different URL, or supply cookies. |
| WhisperX import error in `prepare` | Re-run with `--skip-whisperx` first; open [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) for install notes. |
| CUDA OOM during `train` | Reduce `training_config.yaml → batch_size` (try 8 or 4); retry. |
| `train` crashes at step 0 with ValueError about vocab | The tokenizer fit returned an empty vocab. Check that QC didn't filter out **every** clip — look at `projects/<name>/training_set/ready.json`. |
| `/health` returns non-200 at the end | Port collision. Re-run with `--port 8781` (or pick any free port). |

Any new failure you hit that isn't on this list — file an entry under
[docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) per rule #6 in TASKS.md so
the next person doesn't lose a day on it.

## When to run this

- **Before your first real training run** — mandatory.
- **After every fresh clone or venv rebuild** — catches install drift.
- **After bumping `torch`, `coqui-tts`, or `transformers`** in
  `requirements.txt` — dep upgrades are the #1 way this pipeline
  silently breaks.
- **After changes to any stage of the data pipeline** (`hindi_tts_builder/data/`) or `train/trainer.py`.
