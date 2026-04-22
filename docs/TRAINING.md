# Training Guide

End-to-end guide for training a Hindi TTS from YouTube URLs + SRT transcripts.

## Prerequisites

- WSL2 setup complete (see [SETUP.md](SETUP.md))
- 50+ hours of single-narrator Hindi YouTube videos with matching SRT files
- ~200 GB free disk space

## 1. Create a project

```bash
hindi-tts-builder new my_voice
```

This creates `./projects/my_voice/` with the standard layout and a default
config. You can edit `projects/my_voice/config.yaml` to change things like
clip length thresholds — but defaults are tuned for VITS on 12GB.

## 2. Prepare your sources

Two inputs are required:

**`urls.txt`** — one YouTube URL per line:

```
https://www.youtube.com/watch?v=aaa11111111
https://www.youtube.com/watch?v=bbb22222222
# comments are allowed with #
https://youtu.be/ccc33333333
```

**`transcripts/`** — a directory of `.srt` files, one per URL, in sorted order
by filename:

```
transcripts/
├── 01_episode_one.srt     ↔  first URL
├── 02_episode_two.srt     ↔  second URL
├── 03_episode_three.srt   ↔  third URL
```

**Critical:** the Nth line of `urls.txt` must correspond to the Nth SRT file
when the transcripts folder is sorted alphabetically. Name your SRT files
with leading zeros to guarantee the order.

Register them with the project:

```bash
hindi-tts-builder add-sources my_voice \
    --urls urls.txt \
    --transcripts ./transcripts/
```

The SRT files are copied into the project, so you can safely delete the
originals afterward.

## 3. Run the data pipeline

```bash
hindi-tts-builder prepare my_voice
```

This runs five stages in order:

| Stage | What it does | Expected time (50h) |
|---|---|---|
| Download | `yt-dlp` extracts audio from each URL | 4–8 hours (network-bound) |
| Align | `WhisperX` refines SRT timestamps | 2–3 hours |
| Segment | `ffmpeg` cuts per-cue clips | 45 minutes |
| QC | SNR / silence / Whisper-CER filter | 2–3 hours |
| Dataset | Build `train.csv`, `val.csv`, `test.csv` | <1 minute |

Every stage is **resumable**. If you Ctrl+C or lose power:

```bash
hindi-tts-builder prepare my_voice    # picks up where it stopped
```

Skip slow stages on first iteration:

```bash
# Fast iteration path: skip WhisperX alignment and Whisper QC
hindi-tts-builder prepare my_voice --no-whisperx --no-whisper-qc
```

Output locations:

- `projects/my_voice/audio/raw/` — downloaded WAVs
- `projects/my_voice/aligned/<source_id>/` — per-cue clips (WAV + TXT)
- `projects/my_voice/training_set/train.csv` — final training set
- `projects/my_voice/training_set/qc_report.csv` — pass/fail per clip
- `projects/my_voice/training_set/vocabulary.json` — character set

## 4. Train

```bash
hindi-tts-builder train my_voice
```

On first run:

- Fits the tokenizer on your training set
- Saves `projects/my_voice/checkpoints/tokenizer.json`
- Saves `projects/my_voice/training_config.yaml` (you can edit it)
- Starts VITS training via Coqui TTS

Expected wall-clock on 12GB GPU: **~5 days for 500,000 steps**.

Dry-run / validation without training:

```bash
hindi-tts-builder train my_voice --prepare-only
```

Training is resumable. `latest.pt` is auto-detected on restart:

```bash
# After interruption
hindi-tts-builder train my_voice    # picks up from latest checkpoint
```

### Monitoring progress

Training logs go to `projects/my_voice/logs/train.log`. Coqui TTS also writes
TensorBoard logs under `projects/my_voice/checkpoints/`:

```bash
tensorboard --logdir projects/my_voice/checkpoints/
```

Key metrics to watch:

- `loss_gen_total` — should decrease steadily in first 50k steps
- `loss_disc_real` and `loss_disc_fake` — should hover near 0.5 each (they're adversarial)
- Audio samples auto-generated every 25k steps; listen to them

### Health checks

The trainer auto-checkpoints every 10,000 steps and keeps the last 5 plus
step-0. If loss diverges (NaN, sudden spike), training halts and the last
good checkpoint is preserved.

## 5. Export the engine

```bash
hindi-tts-builder export my_voice
```

Creates a self-contained `projects/my_voice/engine/` folder:

```
engine/
├── manifest.json           # version, sample rate, frontend config
├── model.pt                # trained VITS weights
├── tokenizer.json          # vocab
├── pronunciation_dict.json # optional custom pronunciations
└── training_config.yaml    # snapshot
```

This folder is portable. Move it to another machine with the same package
version and it works identically.

## 6. Use the engine

See [INFERENCE.md](INFERENCE.md).

## Common issues

**CUDA out of memory during training**
→ Reduce `training.batch_size` in `projects/my_voice/training_config.yaml`.
→ VITS at batch_size 8 still trains fine on 12GB, just slower.

**Training loss plateaus early**
→ Usually a data quality issue. Inspect `qc_report.csv` — is the Whisper CER
  column showing many high values? That indicates training transcripts don't
  match the audio accurately. Fix source SRTs or drop affected sources.

**Ctrl+C corrupted a checkpoint**
→ The atomic-write pattern (tmp → rename) prevents this, but if it happens,
  delete the most recent `ckpt_step_*.pt` and restart. `latest.pt` will
  auto-repoint to the previous good one.

**WhisperX import errors**
→ WhisperX has strict `pyannote.audio` version requirements. Workaround:
  `hindi-tts-builder prepare --no-whisperx my_voice` — quality drops
  slightly but pipeline works.

## Tuning for quality vs speed

The defaults (batch_size=16, 500k steps) target MOS 4.4+. For faster iteration
while prototyping:

```yaml
# projects/my_voice/training_config.yaml
batch_size: 32              # if you have more VRAM
max_steps: 200000           # faster but lower quality ceiling
```

For higher quality at the cost of time:

```yaml
batch_size: 8               # smaller but more stable
max_steps: 800000           # diminishing returns past 500k for 50h data
```
