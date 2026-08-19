# Hindi TTS Builder

Private end-to-end pipeline that turns YouTube videos + Hindi SRT transcripts into a trained Hindi text-to-speech model. Single command from raw URLs to a synthesised wav.

**Status (2026-04-26):** v4 VITS-from-scratch run live (PID 77436), step ~11,700 of 150,000 (~3h09m elapsed, 56.6 steps/min, GPU 100% / 10.2 GB, last BEST 4,900). Resumed from `best_model_7350.pth` after v3 NaN-collapsed under fp16. Trainer ships with NaN-guard + AMP-off default. Studio (PID 85150) serves the live Training Health panel.

**Canonical plan: see [ROADMAP.md](ROADMAP.md).** This file describes how the system is built; the roadmap describes what to do next (staged datasets, backend pivot to Indic-Parler-TTS, benchmark pack, retry, dict-central, long-form eval).

---

## 1. Quick start

```bash
# Windows: double-click `Start Studio.bat` in repo root.
# Or from any shell:
hindi-tts-builder studio       # opens http://127.0.0.1:8770
```

In the browser:
1. Click **+ New voice**, give it a name (e.g. `divya`).
2. Paste YouTube URL + matching `.srt` filename pairs (Nth URL ↔ Nth SRT in sorted order).
3. (Optional) Drop one or more anime/donghua portraits via **🖼 Upload avatars** — they cycle as the model matures.
4. Click **▶ Start training**. The page streams logs, shows per-stage progress, model maturity %, the **Training Health panel** (status pill + 6 sparklines + alarms), and a procedural anime character that grows new layers as training advances.

CLI equivalent for power users:
```bash
hindi-tts-builder new divya
hindi-tts-builder add-sources divya --urls urls.txt --transcripts ./transcripts/
hindi-tts-builder prepare divya [--trust-srt] [--skip-qc]
hindi-tts-builder audit  divya                       # pre-train sanity check
hindi-tts-builder make-profile divya 30h_clean       # subsample staged dataset
hindi-tts-builder train  divya --backend coqui-vits  # or --backend indic-parler
hindi-tts-builder export divya
hindi-tts-builder speak  divya "नमस्ते दुनिया।" --out hello.wav
hindi-tts-builder eval-long divya --srt scene.srt --out-dir reports/  # long-form audit
```

---

## 2. What's in the repo

```
hindi-tts/
├── PROJECT.md                       ← this file (architecture + status)
├── ROADMAP.md                       canonical next-steps plan (Indic-Parler pinnacle sprint)
├── README.md                        short intro
├── TASKS.md                         long-running task list (history of decisions)
├── CHANGELOG.md
├── Start Studio.bat                 probe-first launcher (kills stale Win listener on :8770)
├── pyproject.toml                   installs `hindi-tts-builder` CLI entry point
├── requirements.txt                 coqui side: torch, coqui-tts, whisperx, fastapi, indic-nlp, …
├── requirements-indic-parler.txt    parler side: parler-tts (git), accelerate, datasets, jiwer, …
├── eval/
│   ├── hindi_dubbing_100.jsonl      benchmark pack — 30 sentences across 10 categories
│   └── results.csv                  comparison-table scaffold (model/data/steps/CER/RTF/MOS/...)
├── docs/
│   ├── ARCHITECTURE.md  SETUP.md  TRAINING.md  INFERENCE.md
│   ├── SMOKE_TEST.md  TROUBLESHOOTING.md  STATUS.md
├── tests/                           pytest, ~22 modules covering every layer
└── hindi_tts_builder/               the package
    ├── data/
    │   ├── pipeline.py              orchestrates the 5 data stages, idempotent
    │   ├── manifest.py              per-source state machine
    │   ├── download.py              yt-dlp wrapper, m4a → 24 kHz mono wav
    │   ├── align.py                 WhisperX forced alignment (with OOM-fallback chain)
    │   ├── segment.py               cuts cues from SRT; new `trust_srt` mode (50/100ms pad)
    │   ├── qc.py                    SNR + silence ratio + Whisper CER vs ref text
    │   ├── dataset.py               writes train/val/test pipe-CSVs + ready.json
    │   ├── audit.py                 NEW: pre-training dataset audit (hours, p95, char-cov, sp-consistency)
    │   └── profiles.py              NEW: staged subsamplers (2h/10h/30h/60h/100h_clean)
    ├── exporters/                   NEW: per-backend dataset exporters
    │   ├── __init__.py
    │   └── indic_parler.py          pipe-CSV → train.jsonl + clips_44k/ + HF dataset script
    ├── frontend/                    Hindi text → token-stream pipeline (NFC, numbers, schwa, prosody)
    │   ├── pipeline.py  normalizer.py  numbers.py  hindi_num.py
    │   ├── transliterate.py  schwa.py  prosody.py
    ├── train/
    │   ├── trainer.py               drives Coqui's VITS — heart of training
    │   ├── config.py                TrainingConfig (default mixed_precision="none" — see lesson #5)
    │   ├── tokenizer.py             HindiTokenizer pre-seeded U+0900–U+097F
    │   ├── checkpoint.py            resumability + Coqui best_model_*.pth fallback
    │   ├── dataset.py               CSV reader (pipe-delimited)
    │   └── backends/                NEW: trainer adapter layer
    │       ├── base.py              TrainerBackend ABC: prepare/train/export_engine
    │       ├── __init__.py          registry: get_backend("coqui-vits"|"indic-parler"|"f5-hindi")
    │       ├── coqui_vits.py        wraps existing Trainer (current production)
    │       ├── indic_parler.py      Parler fine-tune driver (cross-venv subprocess, real)
    │       └── f5_hindi.py          stub (parked per ROADMAP.md)
    ├── inference/
    │   ├── engine.py                TTSEngine: load model + tokenizer + frontend; tactic-based retry
    │   ├── manifest.py              EngineManifest + `pronunciation_dict_required` (fail-closed)
    │   ├── retry_tactics.py         NEW: seed_only → extra_normalize → apply_dict → shorten_long
    │   ├── srt_renderer.py          speak each cue, gap-pad, mux to one wav
    │   ├── roundtrip.py             text → wav → ASR → CER (regression detector)
    │   └── backends/                NEW: per-engine inference (parler runs in parler venv only)
    │       ├── __init__.py
    │       └── indic_parler.py      ParlerTTSForConditionalGeneration two-tokenizer pattern
    ├── eval/
    │   ├── runner.py  test_set.py  metrics.py
    ├── utils/
    │   ├── project.py  audio.py  srt.py
    ├── cli/
    │   ├── main.py                  click root: new/add-sources/prepare/audit/make-profile/train/
    │   │                            export/speak/render-srt/eval-long/serve/studio/doctor
    │   │                            + 4 Parler-specific: export-/train-/export-engine-/speak-indic-parler
    │   └── server.py                FastAPI inference server (separate from studio)
    └── web/                         the Training Studio
        ├── app.py                   FastAPI: SSE log stream, multi-avatar, NEW /api/projects/{name}/health
        ├── health.py                NEW: log-tail analyzer (status pill, sparklines, NaN counter, alarms)
        ├── jobs.py                  JobRegistry + reattach_orphans
        └── templates/index.html     SPA: voice gallery, 6-stage strip, full-body anime SVG,
                                          loss-convergence grid, NEW Training Health panel
```

---

## 3. Two-venv setup (required)

`parler-tts 0.2.2` hard-pins `transformers==4.46.1`. `coqui-tts 0.27.5` requires `transformers>=4.57`. They cannot coexist in one venv. The pipeline runs across **two isolated environments** (both share the same CUDA torch wheel; only one talks to the GPU at a time):

| Venv | Purpose | Key pins |
|------|---------|----------|
| `/root/hindi-tts/venv` | Data pipeline + Coqui-VITS train/inference + dataset exporters | `transformers==4.57.6`, `coqui-tts==0.27.5`, `torch==2.5.1+cu121` |
| `/root/parler-venv` | Parler-TTS train subprocess + Parler inference (`speak-indic-parler`) | `transformers==4.46.1`, `parler_tts==0.2.2`, `torch==2.5.1+cu121` |

`IndicParlerBackend` lives in the coqui venv (its `prepare()` only needs the exporter — no parler deps). When it calls `accelerate launch`, it spawns the subprocess with the parler venv's Python, resolved in this order:
1. `IndicParlerBackendConfig.python_executable` (explicit)
2. `HINDI_TTS_PARLER_PYTHON` env var
3. Default: `/root/parler-venv/bin/python3`

Setup recipe in [ROADMAP.md](ROADMAP.md#dual-venv-setup-required-as-of-2026-04-26).

---

## 4. The five data stages

Each stage is **idempotent**. State lives in `projects/<name>/sources/manifest.json`.

| # | Stage | Inputs | Outputs | Module |
|---|-------|--------|---------|--------|
| 1 | **Download** | YouTube URL | 24 kHz mono wav | `data/download.py` |
| 2 | **Align** | wav + SRT | per-cue char-level timing JSON | `data/align.py` (WhisperX) |
| 3 | **Segment** | wav + cues | per-utterance LUFS-normalised wav | `data/segment.py` |
| 4 | **QC** | clip + ref text | pass/fail per clip | `data/qc.py` |
| 5 | **Build training set** | passing clips | `train.csv`, `val.csv`, `test.csv` | `data/dataset.py` |

Flags:
- `--skip-qc` bypasses Stage 4 (every clip passes; **quick experiments only** — for final 100h build, loosen QC thresholds instead per ROADMAP rule 1)
- `--trust-srt` skips Stage 2 (alignment) entirely and cuts directly from the user-supplied SRT with a 50ms-left + 100ms-right pad. Use when SRT timing is verified word-to-word and you don't want WhisperX to introduce drift.

CSV format (pipe-delimited, UTF-8):
```
audio_path|raw_text|processed_text|duration|source_id
clips/src_xxx/c0001.wav|नमस्ते।|namaste.|1.42|src_xxx
```

---

## 5. Trainer architecture (Coqui-VITS = current baseline; Indic-Parler = pinnacle)

`hindi_tts_builder.train.backends` is the adapter layer. CLI: `train --backend {coqui-vits | indic-parler | f5-hindi}`.

### CoquiVitsBackend (current, in production for v4)

Wraps Coqui-TTS's `Vits`. Critical design choices baked in (each one a real bug we hit):

1. **`CharactersConfig` set explicitly.** Coqui defaults to a 67-char English alphabet that silently `<unk>`-discards every Devanagari char. Run 1 wasted 86k steps before this fix. Now: full Devanagari block (U+0900–U+097F) + ASCII digits + curated punctuation.
2. **Pre-flight text-compat check.** Before model init, every char in `train.csv` is checked against `CharactersConfig`. Missing char → run dies in seconds with a precise error.
3. **Custom CSV formatter** registered in Coqui (`hindi_csv`) — Coqui's built-ins assume comma + LJSpeech.
4. **`use_sdp=False` (DDP, not SDP).** Coqui's Stochastic Duration Predictor crashed on short-token batches at step ~5,400 (run 1) and ~12,250 (run 2). DDP is rock-stable.
5. **`mixed_precision: "none"` default.** Run 3 NaN-collapsed at step 8,350 under fp16: GradScaler kept halving on `loss_kl` overflow → scale ~5e-44 → 16h of GPU spinning on garbage. fp32 is ~25% slower but stable.
6. **NaN-guard on `model.train_step`.** Defense-in-depth: even if AMP is re-enabled, the wrapper replaces NaN/Inf losses with zero, logs a warning, and **kills the run after 5 consecutive bad steps**.
7. **`restore_path`-only resume.** Coqui's `get_last_checkpoint(continue_path)` only scans one level deep, but Coqui writes nested. Init weights from BEST + start fresh run dir, also avoids re-loading a poisoned optimizer state.
8. **`latest_checkpoint()` finds `best_model_*.pth` recursively** and prefers BEST over later `checkpoint_*.pth`. Skips `_archive*` subtrees.

### IndicParlerBackend (v5 pinnacle target — fully wired, awaiting first run)

- Real backend, not a stub. Production wrapper around the official Parler-TTS training script.
- `prepare()` calls `IndicParlerExporter` to convert pipe-CSV → HF dataset (`train.jsonl` + clips_44k/ + dataset script). Devanagari raw_text only (refuses Coqui-romanized text); resamples to 44.1 kHz; pace-tagged description per clip.
- `train()` auto-clones `huggingface/parler-tts` into `.external/`, writes `parler_training_config.json` with RTX 3060 12 GB-safe defaults (batch=1, grad-accum=16, fp16, gradient-checkpointing, configurable `max_steps`/`lr`/`dtype`). Spawns `accelerate launch ./training/run_parler_tts_training.py` in the parler venv.
- `export_engine()` bundles checkpoint + tokenizer + pronunciation_dict.json + manifest into `engines/indic_parler/<profile>/`.
- `--smoke` mode caps to 64 train / 16 eval samples for 5-minute wiring validation.
- 4 dedicated CLI commands: `export-indic-parler`, `train-indic-parler`, `export-indic-parler-engine`, `speak-indic-parler`. Generic `train --backend indic-parler` route also goes through the same backend (env vars `HINDI_TTS_PROFILE`, `HINDI_TTS_SMOKE`).

### F5HindiBackend (stub by design)

Per ROADMAP: ignore F5/Kokoro until Indic-Parler is exhausted.

### TrainingConfig (defaults, in `train/config.py`)

| Knob | Default | Why |
|------|---------|-----|
| `batch_size` | 20 | bumped from 16; 24 risks OOM with desktop sharing the 12 GB |
| `grad_accum_steps` | 1 | was 2; effective batch = batch_size now |
| `max_steps` | 150_000 | ~42 h on RTX 3060 to a usable model |
| `mixed_precision` | `"none"` | fp32 — see lesson #5 |
| `num_workers` | 8 | user has 12 CPUs |
| `checkpoint_every_steps` | 10_000 | |
| `eval_every_steps` | 20_000 | BEST still saves on improvement |
| `sample_every_steps` | 100_000 | less synth-overhead during training |
| `max_audio_length_sec` | 10.0 | skip clips longer than this |
| `lr_gen`/`lr_disc` | 2e-4 | Coqui VITS standard |
| `grad_clip_norm` | 5.0 | |
| `warmup_steps` | 4000 | |

Per-project YAML at `projects/<name>/training_config.yaml` overrides these.

---

## 6. Studio (web UI) — `http://127.0.0.1:8770`

FastAPI + single-page HTML. **No build step.**

### Endpoints (`hindi_tts_builder/web/app.py`)
- `GET  /` — the SPA
- `GET  /api/projects` / `POST /api/projects`
- `GET/POST /api/projects/{name}` / `POST /api/projects/{name}/start` / `POST /api/projects/{name}/stop`
- `GET  /api/projects/{name}/status`
- `GET  /api/projects/{name}/logs?tail=200` — Server-Sent Events live stream
- **`GET  /api/projects/{name}/health`** — JSON snapshot powering the Training Health panel
- `POST/GET/DELETE /api/projects/{name}/avatar` — multi-file PNG/JPG/WEBP upload

### JobRegistry (`web/jobs.py`)
- `start_pipeline()` builds chain `prepare && train` as one subprocess
- **`reattach_orphans(projects_root)`** scans `/proc` on studio start and re-registers any orphaned `cli.main (prepare|train) <project>` subprocesses, so studio restart doesn't lose track of running training (verified twice this session — both restarts re-attached v4 PID 77436 cleanly)

### Training Health panel (NEW)

Polls `/health` every 5 seconds. Renders:
- **Status pill**: 🟢 healthy / 🔵 starting / 🟡 STALLED / 🔴 NaN COLLAPSE / ⚪ done
- **6 stat cards**: Last step, Steps/min, Last BEST, NaN steps (last 100), NaN-guard fires, Step samples
- **Alarms** (orange callout) — surfaces things like `avg_loss_duration > 5× target floor`
- **6 sparklines** — one per VITS loss component (mel, kl, gen, disc, feat, duration). Each shows a 60-sample SVG line + dashed target-floor line + plain-language label ("KL divergence · target ≤ 1.5 · latent-space alignment")
- **Reading guide** — bold-key paragraph explaining what green/yellow/red mean, what BEST is, what NaN counter means, what steps/min tells you

Backed by `web/health.py`: tail-parses studio_run.log (5 MB window), slices to the most recent run-restart marker (so v3's NaN-collapse era doesn't poison v4's stats), detects status from consecutive-NaN count, stall threshold (3 min), eval-average comparison vs target floors. Bullet-proof against in-progress eval blocks (walks backwards until it finds one with values).

### UI features (templates/index.html)
- URL ↔ SRT pair-row form (numbered, add/remove)
- Voice gallery with avatar tiles
- **Procedural full-body anime SVG** with maturity-driven layers (head → eyes → hair → jacket → headphones → mic → music notes → aura) in magenta/navy/gold palette
- Composite maturity %: `0.3 × step% + 0.7 × loss-convergence%`
- Multi-avatar cycling (upload N images, they cycle through stages by maturity)
- 6-stage pipeline strip with per-stage status icons
- Live SSE log tail with last-200 backfill

---

## 7. Hindi text frontend

`HindiFrontend` (`frontend/pipeline.py`) — the canonical text → token-stream pipeline. Used both at training-data build time AND at inference time so the model sees identical input.

Stages:
1. **Normalize** — NFC, strip dotted circles, collapse whitespace
2. **Number expansion** — Indic system: 1,00,000 → "एक लाख". ASCII-only word boundaries (Python's `\b`+`\w` includes Devanagari, breaks matches).
3. **Transliterate** — Latin-script Hindi back to Devanagari (configurable)
4. **Schwa deletion** — drop final inherent schwa
5. **Prosody tagging** — `<falling>`, `<p_short>` tokens. **Disabled in v1** because Coqui's char-level Graphemes tokenizer would split `<falling>` into 9 chars. Re-enable with an atomic-aware tokenizer subclass.

`HindiTokenizer` is pre-seeded with full Devanagari + ASCII digits + basic punctuation in `__init__`.

---

## 8. Inference

```python
from hindi_tts_builder import TTSEngine
eng = TTSEngine.load("projects/divya/engine")
result = eng.speak("नमस्ते दुनिया।")        # GenerationResult(audio, sr, retries, manual_review, tactics_tried)
eng.speak_to_file("…", "out.wav")
eng.render_srt("subs.srt", "out.wav")
```

Engine bundle = `model.pt` + `tokenizer.json` + optional `pronunciation_dict.json` + `training_config.yaml` + `manifest.json`.

### Tactic-based retry (NEW)

When validation fails, `engine.speak()` walks a tactic chain:
1. `seed_only` — re-roll with a different RNG seed
2. `extra_normalize` — strip stray glyphs, collapse repeated punctuation
3. `apply_dict_aggressive` — force every pronunciation-dict entry, case-insensitive
4. `shorten_long` — last-resort: keep only the first sentence

Each tactic is logged. If all fail, the result is flagged `manual_review=True` with `tactics_tried=[…]` for the caller to persist.

### Pronunciation dict promoted to first-class

Manifest field `pronunciation_dict_required: bool` (default False for back-compat). When True, `TTSEngine.load()` raises if the dict is missing — no silent fallback. Indic-Parler engines ship with this set to True.

### Long-form eval (NEW)

```bash
hindi-tts-builder eval-long divya --srt scene.srt --out-dir reports/
```
Renders an SRT through the engine, then audits per cue. Outputs `final_mix.wav` + `failed_cues.csv` + `duration_mismatch.csv` + `cer_report.csv`. A model that sounds great on 10s clips can still fall apart in a 30-min video.

---

## 9. Lessons learned (the painful ones)

| Run | Outcome | Root cause | Fix |
|-----|---------|------------|-----|
| **Run 1** (Apr 24, 86k steps) | Wasted | `CharactersConfig=None` → English default → all Hindi `<unk>` | Explicit Devanagari `CharactersConfig` + pre-flight text-compat check |
| **Run 2** (Apr 24, crashed @ 12,250) | SDP empty-tensor crash on short batches | `use_sdp=False` (Deterministic Duration Predictor) |
| **Run 3** (Apr 25–26, NaN-collapsed @ 8,350, ran 16 h producing nothing) | fp16 GradScaler underflowed on `loss_kl` overflow → optimizer stepped on NaN forever | `mixed_precision="none"` default + NaN-guard wrapper + BEST-aware `latest_checkpoint()` + `restore_path`-only resume + `/health` panel so silent collapse is visible |
| **Run 4** (Apr 26, in progress) | All losses finite at step 11,700; resumed from BEST 7,350; new BEST 4,900 written; nan-guard never fired | (current) |

Other production-grade fixes already shipped:
- WhisperX cuDNN-8 side-load (`/opt/cudnn8`) so it coexists with torch's cuDNN-9
- WhisperX OOM auto-fallback chain (large-v3 → medium → small, batch 16 → 4 → 1, fp16 → int8_float16)
- Stale Windows :8770 listener auto-killed by `Start Studio.bat` (probe-first)
- WSL2 localhost forwarding reset via `wsl --shutdown` on cold start
- `from __future__ import annotations` removed from `web/app.py` (broke FastAPI's `TypeAdapter` for `UploadFile` ForwardRef)
- bash `||` swallowing exit codes under `set -e` → replaced with explicit `if` blocks
- Git Bash `$?` mangling through `wsl.exe` → switched to `bash -s <<EOF` heredocs
- Health analyzer slices to current-run before counting NaN/throughput (otherwise v3 era pollutes v4 stats)
- `_BEST_RE` uses positive `BEST MODEL :` marker (filters out resume-source mentions)
- Exporter no longer silently falls back to bare `train.csv` when a profile is missing — raises with a list of available profiles

---

## 10. Roadmap (canonical: [ROADMAP.md](ROADMAP.md))

### v4 — current VITS-from-scratch run (baseline only)
- Let it cook to step ~30,000; run benchmark on `eval/hindi_dubbing_100.jsonl`; lock in a baseline row in `eval/results.csv`
- DO NOT perfect VITS forever — it's a baseline, not a target

### v5 PINNACLE — Indic-Parler-TTS fine-tune
1. Set up dual venv (done)
2. Build staged datasets (`make-profile` 2h_clean / 10h_clean / 30h_clean / 60h_clean / 100h_clean — only 2h_clean exists currently)
3. Smoke train: `train-indic-parler divya --profile 2h_clean --smoke --max-steps 50` (~5 min)
4. 2h real → 10h → 30h → 60h → 100h ladder, with sample-listening gate after each
5. Pick the best checkpoint by 20-min real SRT, NOT by loss

### Skipped per ROADMAP
- ❌ XTTS v2 — CPML non-commercial + Coqui dead Jan 2024
- ❌ MMS-TTS-hin — CC-BY-NC, mechanical quality
- ❌ OpenVoice v2 / MeloTTS — no Hindi
- ❌ StyleTTS 2 — no Indic checkpoint
- ❌ F5-Hindi — secondary; only after Indic-Parler is exhausted

### Backlog
- Drop `HindiTokenizer`, use Coqui's `Graphemes` everywhere (one less moving part)
- Atomic-aware tokenizer subclass to re-enable prosody tokens (`<falling>`, `<p_short>`)
- `--mixed-precision fp16` re-enable path with NaN-guard proven over 50k+ steps
- `hindi-tts-builder dict add/remove/list` CLI for managing pronunciation dictionaries

---

## 11. Troubleshooting (top 8)

| Symptom | Probable cause | Fix |
|---------|----------------|-----|
| Studio "Loading…" forever | JS SyntaxError in `index.html` | `node --check` (extract `<script>` body first) |
| `ERR_CONNECTION_REFUSED` on :8770 | Stale Windows-side listener; WSL2 localhost broken | `Start Studio.bat` auto-handles |
| `character X is not in vocabulary` warning during prepare | Char missing from `CharactersConfig` | Add to `_PUNCT_FOR_COQUI` in `trainer.py` |
| Train aborts: `No models found in continue path` | Coqui's nested run-dir issue | Already fixed — `restore_path`-only resume |
| Training step counter frozen, all losses NaN | AMP underflow | Already fixed — fp32 default + NaN-guard. Health panel surfaces in red. |
| WhisperX OOM at large-v3 | Too-aggressive defaults | Auto-fallback chain; or `HTTS_WHISPERX_MODEL=medium HTTS_WHISPERX_BATCH=4 HTTS_WHISPERX_COMPUTE=int8_float16` |
| `transformers` version conflict between coqui and parler | Same venv | Use the documented dual-venv setup |
| Parler train: "Could not find train/val CSVs for profile X" | Profile not yet built | `hindi-tts-builder make-profile <project> X` |

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for the long list.

---

## 12. Hardware target

- **GPU:** RTX 3060 12 GB (training works on anything ≥ 8 GB at smaller batch)
- **CPU:** 12-core (8 dataloader workers × ~1 step/sec throughput on the coqui side; ~57 step/min currently observed for v4)
- **RAM:** 32 GB recommended (16 GB OK with swap)
- **Disk:** ~50 GB free per project (raw audio + segmented clips + checkpoints + Parler vectorized cache)
- **OS:** Windows 11 + WSL2 Ubuntu-22.04 + CUDA 12 (the supported path); native Linux works too

---

## 13. License

Private personal project. Code by Shirshendu (`sasmalgiri@gmail.com`). Coqui-TTS itself is MPL-2.0. Hindi NLP libs (indic-nlp-library, num2words) MIT/BSD. WhisperX BSD-2.

If pivoting to Indic-Parler-TTS (Apache 2.0) for v5 — and it's the recommended path — the resulting fine-tuned model can be released commercially with attribution. Avoid XTTS / MMS bases if commercial use matters.
