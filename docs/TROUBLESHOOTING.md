# Troubleshooting

Common problems and fixes, organized by which stage of the pipeline they
appear in.

## Environment & Setup

### `nvidia-smi` not found inside WSL2

Your Windows NVIDIA driver is either too old or not installed. Update to the
latest Game Ready / Studio driver from [nvidia.com](https://www.nvidia.com).
Any driver from 2022+ supports WSL2 GPU access.

### CUDA reports `False` in Python

Installed CPU-only torch by accident.

```bash
pip uninstall -y torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### `ffmpeg: command not found`

```bash
sudo apt update && sudo apt install -y ffmpeg
```

### `coqui-tts` install fails

Coqui's installer pins tight versions. Install in a fresh venv:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install coqui-tts
```

If still failing, install `numpy<2.0` first (Coqui incompatible with NumPy 2):

```bash
pip install "numpy<2.0"
pip install coqui-tts
```

## Download stage

### `yt-dlp` returns HTTP 403 / age-restricted errors

Some YouTube videos require cookies. Export your browser cookies and pass via
`yt-dlp`:

```bash
yt-dlp --cookies-from-browser chrome -x --audio-format wav <URL>
```

We don't expose this in `hindi-tts-builder` CLI yet — for now, manually
download the problem videos into `projects/<n>/audio/raw/src_<id>.wav`
and re-run `prepare`. It'll skip the download step.

### Download is very slow

Usually rate limiting from YouTube. yt-dlp handles this automatically but it
slows things down. Let it run — it's still faster than re-running.

### Downloaded WAV is 0 bytes or corrupt

Rerun `prepare`: the download stage has a `size > 0` check and will redownload.

## Alignment stage

### WhisperX fails to import

WhisperX has strict `pyannote.audio` version pins. Common workaround:

```bash
pip install whisperx --force-reinstall
```

If that doesn't help, use the SRT-only fallback:

```bash
hindi-tts-builder prepare my_voice --no-whisperx
```

Quality drops slightly but training still works.

### Alignment takes forever

WhisperX runs Whisper-large-v3 over every video. Expected time is real-time
÷ (your GPU speed). On a 12GB card that's roughly 1 hour of audio per 20–30
minutes of wall clock. If your dataset is 50 hours, plan for 2–3 hours of
alignment.

## Segmentation stage

### Almost every clip gets rejected at QC

Usually means either:
- Your SRT timestamps are badly misaligned (try `--no-whisperx` disabled so
  WhisperX corrects them)
- Audio has a lot of noise → raise `qc.min_snr_db` threshold in config.yaml

Inspect `projects/<n>/training_set/qc_report.csv`:

```bash
head -20 projects/my_voice/training_set/qc_report.csv
```

The `reason` column tells you why each clip failed.

### Clips have wrong durations

Your SRT timestamps might be in the wrong format. Check one by hand:

```
1
00:00:01,000 --> 00:00:03,500      ← HH:MM:SS,mmm (comma, not period)
नमस्ते दुनिया
```

Common mistake: using `.` instead of `,` for milliseconds. Our parser
accepts both, but some tools export `.` and some export `,`.

## Training

### CUDA out of memory

Lower the batch size in `projects/<n>/training_config.yaml`:

```yaml
batch_size: 8       # was 16
```

Or enable gradient checkpointing (already on by default for VITS).

### Training loss diverges / NaN

This almost always means a problem with the training data, not the model:

1. Inspect `qc_report.csv` for anomalously long or short clips that slipped
   through
2. Listen to a few random clips from `aligned/<source>/` — are they
   pure voice?
3. Reduce learning rate: `training_config.yaml` → `optim.learning_rate_gen: 1e-4`
4. Reduce batch size — sometimes a single bad clip poisons a small batch

### Training is slow — why?

Check your data loader isn't the bottleneck:

```bash
nvidia-smi dmon -s u                # GPU utilization live
```

If GPU utilization is <70%, data loading is the bottleneck. Increase
`num_workers` in training_config.yaml (`num_workers: 4` on WSL2/Linux).

On Windows native, `num_workers` > 0 uses `spawn()` and is slow. This is why
we recommend WSL2.

### Resuming training resets step counter

If `latest.pt` is pointing at the wrong file or is a stale symlink, reset it:

```bash
ls projects/my_voice/checkpoints/ckpt_step_*.pt | tail -1
ln -sf <that-file-name> projects/my_voice/checkpoints/latest.pt
```

## Inference

### First speak() is slow (~5s)

That's model loading + warmup. Keep the `TTSEngine` object around — later
calls are fast.

### Output is silent or noise

Almost always a tokenizer mismatch. The engine expects the tokenizer it was
trained with. If you re-exported from a different training run, the vocab
may not match the weights. Fix: re-export from the same training run and
reload.

### Round-trip validation always fails

Two cases:
1. **Your model is undertrained.** Validation is catching real problems.
   Train longer or inspect samples.
2. **Whisper can't understand your narrator.** If your narrator has a strong
   regional accent and Whisper was only trained on "standard" Hindi, CER will
   be high even on perfect output. Raise the CER threshold:

    ```python
    engine = TTSEngine.load(engine_dir, cer_threshold=0.10)
    ```

### Python `ImportError: cannot import 'TTSEngine'`

You probably haven't `pip install -e .`'d the project. From the repo root:

```bash
pip install -e .
```

Or run as a module:

```bash
python -m hindi_tts_builder.cli.main speak my_voice --text "..." --out ...
```

## Project maintenance

### Where do all my files go?

```
projects/my_voice/
├── config.yaml                     # main settings
├── training_config.yaml            # training knobs
├── sources/                        # urls, transcripts, manifest
├── audio/raw/                      # downloaded WAVs (BIG — ~GB per hour)
├── audio/resampled/                # 24kHz mono (unused if ffmpeg-resamples in segment)
├── aligned/                        # per-cue clips + metadata
├── training_set/                   # train.csv, val.csv, qc_report.csv
├── checkpoints/                    # model checkpoints (BIG — ~500MB each)
├── engine/                         # final deployable package
└── logs/                           # everything logged
```

You can safely delete `audio/raw/` after segmentation (stage 3 finished).
Clips under `aligned/` become the actual training data. Keep `engine/`
forever — it's your trained model.

### Disk usage getting large

Per-stage approximate sizes for a 50-hour project:

- `audio/raw/` — 25–40 GB
- `aligned/<sources>/` — 15–25 GB
- `checkpoints/` — 5–10 GB (last N checkpoints + step 0)
- `engine/` — 400 MB

Total: ~50–80 GB. Delete `audio/raw/` after segmentation if tight on space.
