# Project Status

Current version: **v1.0.0 — feature complete**

All planned subsystems for v1 are built, tested, and documented.

## What's in v1

### Hindi text frontend — ✅ complete
- Unicode NFC normalization, nukta consolidation, invisible-char stripping
- Hindi cardinal numbers (Indian numbering: thousand/lakh/crore/arab)
- Digits, decimals, dates, times, currency, percentages → Devanagari
- Latin → Devanagari transliteration with custom pronunciation dictionary
- Schwa deletion via halant insertion
- Punctuation → prosody tokens (`<p_short>`, `<falling>`, `<rising>`, etc.)
- End-to-end `HindiFrontend` orchestrator

### Utilities — ✅ complete
- Audio helpers (resample, loudness, SNR, silence trimming)
- SRT parser/writer with BOM and CRLF tolerance
- Project folder layout & config management
- Structured logging

### Data pipeline — ✅ complete
- YouTube download (`yt-dlp`) — resumable, per-source error recovery
- WhisperX forced alignment with graceful SRT-only fallback
- ffmpeg-based per-cue segmentation with loudness normalization
- QC filtering (SNR + silence ratio + Whisper round-trip CER)
- Deterministic train/val/test split builder
- Top-level pipeline orchestrator (`run_pipeline`)

### Training — ✅ complete
- TrainingConfig dataclass with YAML round-trip
- HindiTokenizer (character + prosody-token aware, closed vocab)
- TTSDataset (PyTorch, lazy torch imports)
- Resumable checkpoint management with atomic writes
- `Trainer` class driving Coqui TTS VITS under the hood
- Engine export bundles model + tokenizer + frontend state

### Inference — ✅ complete
- `TTSEngine.load()` — version-checked engine loading
- Round-trip Whisper validator for zero-omission guarantee
- SRT renderer with `fit_to_cue` and `natural` modes
- Per-sentence seed-based regeneration on validation failure
- Lazy model loading, frontend output caching

### CLI + HTTP API — ✅ complete
- `hindi-tts-builder new` — create project
- `hindi-tts-builder add-sources` — register URLs + SRTs
- `hindi-tts-builder prepare` — run full data pipeline
- `hindi-tts-builder train` — run VITS training
- `hindi-tts-builder export` — bundle engine for inference
- `hindi-tts-builder speak` — generate single WAV
- `hindi-tts-builder render-srt` — SRT → timed WAV
- `hindi-tts-builder serve` — FastAPI HTTP server (OpenAPI docs at /docs)
- `hindi-tts-builder doctor` — environment diagnostics

### Evaluation — ✅ complete
- TestSet builder with 5 fixed categories
- CER / WER / RTF / UTMOS metrics
- Per-category and overall aggregation
- Runner that evaluates a TTSEngine against a locked test set

### Documentation — ✅ complete
- `README.md` — overview, quick start, pipeline stages
- `docs/SETUP.md` — WSL2 install, GPU driver, Python env
- `docs/TRAINING.md` — end-to-end training guide with tuning
- `docs/INFERENCE.md` — Python API + CLI + HTTP API reference
- `docs/TROUBLESHOOTING.md` — common issues by pipeline stage
- `docs/ARCHITECTURE.md` — design decisions for your future self
- `CHANGELOG.md` — milestone history

### Test suite — 211 passing tests
- Frontend coverage: every rule, every number range, token preservation
- Data pipeline: manifest, split determinism, CER computation
- Training: tokenizer, config round-trip, checkpoint management
- Inference: manifest, roundtrip logic, SRT renderer helpers
- CLI: every subcommand's help + project creation + source registration
- Evaluation: test-set construction, metrics, aggregation

Run with:
```bash
pytest tests/ -v
```

## What comes after v1

See `CHANGELOG.md` for the future roadmap. Summary:

- **v2** — StyleTTS 2 backend swap (higher quality, emotion-via-reference-audio)
- **v3** — ONNX export, inference speed optimizations, Docker image
