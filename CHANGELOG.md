# Changelog

## v1.0.0 — Feature complete

All v1 subsystems built, tested, documented.

**New in this release:**

### Inference subsystem
- `inference/manifest.py` — versioned engine manifest with compatibility checking
- `inference/roundtrip.py` — Whisper-based round-trip validator (faster-whisper preferred, openai-whisper fallback, graceful skip when neither available)
- `inference/engine.py` — `TTSEngine` with `load()`, `speak()`, `speak_many()`, lazy model loading, frontend caching, seeded regeneration on validation failure
- `inference/srt_renderer.py` — `SRTRenderer` with `fit_to_cue` and `natural` modes, ffmpeg atempo time-stretching with numpy resample fallback

### CLI + HTTP server
- `cli/main.py` — 9 Click sub-commands (`new`, `add-sources`, `prepare`, `train`, `export`, `speak`, `render-srt`, `serve`, `doctor`)
- `cli/server.py` — FastAPI app with `/speak`, `/render-srt`, `/health`, `/info` endpoints, auto-generated OpenAPI docs

### Evaluation harness
- `eval/test_set.py` — 5-category locked test set (narration/dialogue/numeric/pronunciation/long), deterministic IDs
- `eval/metrics.py` — CER, WER, UTMOS (optional), RTF, per-category aggregation
- `eval/runner.py` — runs engine over test set, writes `report.csv` + `report.json`

### Documentation
- `docs/TRAINING.md` — end-to-end training guide with timing estimates, tuning, common issues
- `docs/INFERENCE.md` — Python API + CLI + HTTP reference
- `docs/TROUBLESHOOTING.md` — 25+ common issues organized by pipeline stage
- Updated `docs/STATUS.md` and `README.md` to reflect feature-complete state

**Tests: 211 passing** (up from 163 at v0.3.0).

---

## v0.3.0 — Training milestone

Adds the complete training subsystem: config, tokenizer, PyTorch dataset,
checkpoint management, and Coqui-TTS-driven VITS trainer.

**New modules (`hindi_tts_builder/train/`):**
- `config.py` — `TrainingConfig` dataclass with `ModelConfig` and `OptimConfig` nested. YAML round-trip. Sized for 12GB VRAM.
- `tokenizer.py` — `HindiTokenizer`: character-level + prosody-token-aware. Atomic `<p_short>` etc. handling. Closed vocab for stability. Save/load as JSON.
- `dataset.py` — `TTSDataset` (lazy torch imports) reading train/val CSVs. Computes mel spectrograms with standard VITS hyperparameters. `collate_batch` handles padding.
- `checkpoint.py` — List, prune, load, save. Atomic writes. Symlink-with-copy-fallback for Windows. Preserves step-0 checkpoint.
- `trainer.py` — `Trainer` class. `prepare()` validates data, fits tokenizer, saves configs (no GPU needed). `train()` drives Coqui TTS VITS. `export_engine()` bundles model + tokenizer + frontend state for inference.

**New tests (all passing):**
- `tests/test_tokenizer.py` — 13 tests for split, fit, encode/decode, persistence
- `tests/test_training_config.py` — 5 tests for defaults and YAML round-trip
- `tests/test_checkpoint.py` — 6 tests for listing, sorting, filtering

**Total tests: 163 passing.**

**Design notes:**
- Coqui TTS (`coqui-tts` pip package) is used for the actual VITS training loop. Re-implementing VITS from scratch is a multi-month research project; Coqui's is battle-tested.
- All torch imports are lazy. `import hindi_tts_builder` does not pull torch.
- `Trainer.prepare()` runs without torch or a GPU, so you can validate data on a laptop before going to the training box.
- Resumability: `train()` auto-detects `latest.pt` and continues. Safe to Ctrl+C.

---

## v0.2.0 — Data pipeline milestone

Adds the complete data pipeline: YouTube audio download, WhisperX alignment,
clip segmentation, quality filtering, and training-set assembly.

**New modules (`hindi_tts_builder/data/`):**
- `manifest.py` — Source tracking with resumable state (per-source status flags). Deterministic IDs from YouTube video IDs.
- `download.py` — yt-dlp wrapper with per-source crash recovery. Writes 48kHz mono WAV. Skips sources already downloaded.
- `align.py` — WhisperX forced alignment against user SRTs, snapping cues to word boundaries. Graceful fallback to raw SRT timestamps when WhisperX unavailable.
- `segment.py` — ffmpeg-driven clip extraction with loudness normalization, silence trimming, and duration filtering.
- `qc.py` — Quality filtering: SNR threshold, silence ratio, Whisper round-trip CER, duration bounds. Produces `qc_report.csv`.
- `dataset.py` — Assembles train/val/test splits with deterministic hash-based assignment. Writes `train.csv`, `val.csv`, `test.csv`, `vocabulary.json`.
- `pipeline.py` — Top-level orchestrator (`run_pipeline()`).

**New tests (all passing):**
- `tests/test_manifest.py` — 9 tests covering stable IDs, round-trip, duplicate handling
- `tests/test_data_logic.py` — 11 tests covering split determinism, distribution, CER computation
- `tests/test_project.py` — 9 tests for ProjectPaths and config management

**Total tests: 138 passing.**

**Resumability:** Every stage is idempotent. Re-running after interruption skips already-processed work using manifest status flags and filesystem checks. Safe to Ctrl+C any stage.

**CER (Whisper round-trip) in QC:** Catches mispronunciations, wrong-speaker clips, and alignment failures at dataset-build time — before they pollute training.

---

## v0.1.0 — Frontend milestone

Initial release with complete Hindi text frontend and project scaffolding.

**Modules:**
- `frontend/normalizer.py` — Unicode NFC, nukta consolidation, invisible-char removal
- `frontend/hindi_num.py` — Self-contained Hindi cardinal number writer (Indian numbering: thousand/lakh/crore/arab)
- `frontend/numbers.py` — Digits, decimals, times, dates, currency, percentages → Devanagari
- `frontend/transliterate.py` — Dictionary-first Latin → Devanagari with rule-based fallback
- `frontend/schwa.py` — Rule-based Hindi schwa deletion via halant insertion
- `frontend/prosody.py` — Punctuation → prosody tokens with strict preservation regex
- `frontend/pipeline.py` — End-to-end `HindiFrontend` orchestrator
- `utils/audio.py` — Resampling, loudness, SNR, silence metrics
- `utils/srt.py` — SRT parser/writer with BOM and CRLF tolerance
- `utils/project.py` — Project layout and config management

**Tests: 109 passing.**
