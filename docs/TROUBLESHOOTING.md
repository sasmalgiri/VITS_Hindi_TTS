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

### `Could not load library libcudnn_ops_infer.so.8` during `prepare` (WhisperX)

**Symptom:** `hindi-tts-builder prepare` aborts in Stage 2 (align) with
a core dump and the message:

```
Could not load library libcudnn_ops_infer.so.8. Error:
libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory
```

**Cause:** WhisperX uses `faster-whisper`, which uses `ctranslate2`, which
dynamically links against **cuDNN 8**. Meanwhile torch 2.5.1+cu121 bundles
**cuDNN 9** — so the venv has v9 but ctranslate2 can't find v8.

**Fix — keep both side-by-side:**

```bash
# Stage cuDNN 8 in a separate directory so it doesn't clobber torch's v9
mkdir -p /opt/cudnn8
pip install --target /opt/cudnn8 nvidia-cudnn-cu12==8.9.7.29

# Prepend both dirs to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cudnn8/nvidia/cudnn/lib:/root/hindi-tts/venv/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

Append the `export` line to `~/.bashrc` (or your distro's shell-init file)
so every new shell gets it. `Start Studio.bat` handles this automatically.

**Why not just `pip install nvidia-cudnn-cu12==8.9.7.29`?**
That overwrites torch's v9 and breaks torch with
`ImportError: libcudnn.so.9: cannot open shared object file`. They must
coexist, not replace.

### `CUDA failed with error out of memory` during Stage 2 (align)

**Symptom:** WhisperX loads, starts transcribing, then aborts with:

```
2026-xx-xx ... [ERROR] data.pipeline: [fail] src_... : CUDA failed with error out of memory
```

**Cause:** the default Whisper model used for timestamp alignment
(`large-v3` at float16, batch_size 16) needs ~10-11 GB VRAM. On a 12 GB
card sharing VRAM with a Windows desktop (~4 GB reserved by the
compositor + browser GPU accel), only ~7-8 GB is actually free.

**Fix:** `align.py` reads three env vars at alignment time. Defaults are
already tuned for 12 GB-with-desktop (`medium`, batch 4, `int8_float16`).
If you still OOM, step further down:

```bash
export HTTS_WHISPERX_MODEL=small     # or base, tiny
export HTTS_WHISPERX_BATCH=2
export HTTS_WHISPERX_COMPUTE=int8    # pure int8, smallest memory
```

On OOM the loader automatically falls back to smaller models before
giving up. If every model fails, the pipeline falls back to using the
SRT timestamps as-is (equivalent to `--no-whisperx`). You'll see one of:

```
[whisperx] medium OOM or CUDA error; trying smaller model
[whisperx] src_...: fell back to SRT timestamps
```

**Free up VRAM without reducing model size:** close the browser, disable
hardware acceleration in Chrome/Edge, or set `HTTS_WHISPERX_MODEL=base`
temporarily during alignment. VRAM is only needed for the ~1-2 min of
per-source alignment; you can turn it back on once `prepare` completes.

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
