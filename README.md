# Hindi TTS Builder

Build a private Hindi text-to-speech engine from YouTube videos + SRT transcripts.

Give it a folder of YouTube URLs and matching SRT files. It downloads, aligns,
segments, trains a VITS model end-to-end, and exports a reusable TTS engine
that you can call from Python, the command line, or HTTP.

**Designed for:** one narrator, one language (Hindi), private personal use, ship-grade code.

**Platform:** Windows 11 + WSL2 (Ubuntu 22.04), 12GB NVIDIA GPU.

**Status:** v1.0.0 — feature complete. 211 passing tests.

---

## Quick start

```bash
# Setup (one-time, ~30 minutes — see docs/SETUP.md)
wsl --install -d Ubuntu-22.04
# (then inside WSL2)
sudo apt install -y python3.11 python3.11-venv ffmpeg
git clone <this-repo> ~/hindi-tts && cd ~/hindi-tts
python3.11 -m venv venv && source venv/bin/activate
pip install -e . && pip install -r requirements.txt

# Verify your environment
hindi-tts-builder doctor

# Create a new voice project
hindi-tts-builder new my_voice

# Register your YouTube URLs and SRT transcripts
hindi-tts-builder add-sources my_voice \
    --urls urls.txt \
    --transcripts ./transcripts/

# Full data pipeline: download → align → segment → QC → build training set
hindi-tts-builder prepare my_voice

# Train VITS (~5 days continuous on 12GB GPU, resumable)
hindi-tts-builder train my_voice

# Package the trained model
hindi-tts-builder export my_voice

# Use the trained engine
hindi-tts-builder speak my_voice --text "नमस्ते दुनिया" --out hello.wav
hindi-tts-builder render-srt my_voice --srt episode.srt --out episode.wav

# Or serve as HTTP API
hindi-tts-builder serve my_voice --host 127.0.0.1 --port 8765
```

From Python:

```python
from hindi_tts_builder import TTSEngine

engine = TTSEngine.load("projects/my_voice/engine")
result = engine.speak("यह एक परीक्षण है।", output="out.wav")

# With Whisper round-trip validation (catches word omissions)
result = engine.speak("लंबा और जटिल वाक्य।")
if result.validation and result.validation.passed:
    print(f"CER = {result.validation.cer:.3f}, retries used: {result.retries}")
```

## What it does, exactly

1. **Download** — `yt-dlp` pulls audio from YouTube URLs (48kHz mono WAV, resumable)
2. **Align** — WhisperX refines your SRT timestamps to audio-precise word boundaries (SRT-only fallback when WhisperX unavailable)
3. **Segment** — ffmpeg cuts per-cue clips with loudness normalization
4. **QC filter** — rejects clips with low SNR, high silence ratio, or high Whisper round-trip CER
5. **Frontend** — Unicode normalization, Hindi cardinal numbers, schwa deletion, prosody tokens
6. **Train** — VITS from scratch via Coqui TTS, ~5 days on 12GB GPU
7. **Validate** — Whisper round-trip check catches any residual word omissions
8. **Export** — self-contained engine folder, Python + CLI + HTTP ready

## Why VITS

VITS is a non-autoregressive, duration-based TTS model. It explicitly predicts
a duration for every input token and generates audio for all of them in
parallel. This **architecturally cannot drop a word** — no attention collapse,
no truncation. Combined with the Whisper round-trip validator, you get
effectively zero-omission output.

v2 will add a StyleTTS 2 backend option for higher quality with emotion control.

## Project layout

```
hindi-tts/
├── README.md, SETUP.md, TRAINING.md, INFERENCE.md, TROUBLESHOOTING.md, ARCHITECTURE.md
├── hindi_tts_builder/
│   ├── frontend/       — Hindi text processing (7 modules)
│   ├── utils/          — audio, SRT, project paths, logging
│   ├── data/           — YouTube download, alignment, segmentation, QC
│   ├── train/          — VITS training driver, tokenizer, dataset, checkpoints
│   ├── inference/      — TTSEngine, round-trip validator, SRT renderer
│   ├── eval/           — test-set builder, metrics, runner
│   └── cli/            — Click CLI + FastAPI server
└── tests/              — 211 passing tests, <1s total
```

## Quality guarantees

- **Zero word omissions** — architectural (VITS duration prediction) + Whisper round-trip validation
- **Faithful punctuation** — deterministic prosody tokens baked into training and inference
- **Faithful pronunciation** — custom pronunciation dictionary overrides rule-based transliteration
- **Voice consistency** — single-speaker model, zero drift across unlimited generation
- **Reproducible output** — deterministic seeding means identical input → identical audio

## Documentation

| Guide | What's in it |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | WSL2 install, GPU driver, Python environment, verification |
| [docs/TRAINING.md](docs/TRAINING.md) | End-to-end training guide, expected timing, tuning, monitoring |
| [docs/INFERENCE.md](docs/INFERENCE.md) | Python API, CLI, HTTP API with examples |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 25+ common issues organized by stage |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Design decisions recorded for your future self |
| [CHANGELOG.md](CHANGELOG.md) | Milestone history |

## Development

```bash
# Run full test suite (<1 second)
pytest tests/ -v

# Check code quality
pytest tests/ --cov=hindi_tts_builder --cov-report=term-missing
```

## License

Private / personal use.
