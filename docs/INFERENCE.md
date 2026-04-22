# Inference Guide

Using a trained engine to generate audio — from Python, the CLI, or an HTTP
server.

## Python API

### Basic usage

```python
from hindi_tts_builder import TTSEngine

engine = TTSEngine.load("projects/my_voice/engine")

# Generate and save
result = engine.speak("नमस्ते, आज आपका दिन कैसा है?", output="hello.wav")
print(result.sample_rate)       # 24000
print(len(result.audio))        # number of float32 samples

# Generate and use in memory (no file written)
result = engine.speak("यह एक परीक्षण है।")
audio = result.audio             # numpy float32 array, shape [T]
```

### Zero-omission validation

Validation is on by default. The engine runs Whisper on the generated audio
and compares to the input; if the CER exceeds threshold, it regenerates with
a new seed up to `roundtrip_retries` times.

```python
engine = TTSEngine.load(
    "projects/my_voice/engine",
    enable_roundtrip=True,      # default True
    cer_threshold=0.05,         # default 5%
    roundtrip_retries=2,        # default 2
)

result = engine.speak("एक लंबा वाक्य जो मॉडल के लिए कठिन हो सकता है।")
if result.validation:
    print("passed:", result.validation.passed)
    print("cer:", result.validation.cer)
    print("retries used:", result.retries)
```

Disable validation for maximum speed:

```python
result = engine.speak(text, validate=False)
```

### Reproducible generation

```python
# Same seed → identical audio
a = engine.speak("नमस्ते", seed=42)
b = engine.speak("नमस्ते", seed=42)
# a.audio == b.audio element-wise
```

### Rendering an SRT file

```python
from hindi_tts_builder import TTSEngine
from hindi_tts_builder.inference import SRTRenderer

engine = TTSEngine.load("projects/my_voice/engine")

# Mode 1: fit audio into each cue's allotted time (default)
#   Good for dubbing where cue timings are authoritative.
renderer = SRTRenderer(engine, mode="fit_to_cue")
summary = renderer.render("episode1.srt", "episode1.wav")

# Mode 2: natural pace with silence gaps between cues
#   Good when cue timings are loose; narration reads cleanly.
renderer = SRTRenderer(engine, mode="natural", gap_ms_between_cues=300)
summary = renderer.render("episode1.srt", "episode1.wav")
```

### Custom pronunciation for proper nouns

Edit `projects/my_voice/engine/pronunciation_dict.json` to add entries:

```json
{
  "sung_jin_woo": "सुंग जिन वू",
  "solo_leveling": "सोलो लेवलिंग",
  "mana": "माना"
}
```

Keys are lowercase. On the next `TTSEngine.load()` these override the
built-in Latin-to-Devanagari transliteration rules.

### Batch generation

```python
texts = [
    "पहला वाक्य।",
    "दूसरा वाक्य।",
    "तीसरा वाक्य।",
]
results = engine.speak_many(texts)
for r in results:
    print(r.validation.cer if r.validation else "no validation")
```

## Command line

```bash
# Single sentence
hindi-tts-builder speak my_voice \
    --text "नमस्ते दुनिया" \
    --out hello.wav

# Skip validation (faster)
hindi-tts-builder speak my_voice \
    --text "नमस्ते" \
    --out hello.wav \
    --no-validate

# Reproducible
hindi-tts-builder speak my_voice \
    --text "नमस्ते" \
    --out hello.wav \
    --seed 42

# Render an SRT file
hindi-tts-builder render-srt my_voice \
    --srt episode.srt \
    --out episode.wav \
    --mode fit_to_cue          # or "natural"
```

## HTTP server

Start the server:

```bash
hindi-tts-builder serve my_voice --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765/docs for auto-generated OpenAPI documentation.

### Endpoints

**GET /health** — liveness check

```bash
curl http://127.0.0.1:8765/health
# {"status": "ok"}
```

**GET /info** — engine metadata

```bash
curl http://127.0.0.1:8765/info
```

**POST /speak** — text → audio

```bash
curl -X POST http://127.0.0.1:8765/speak \
    -H "Content-Type: application/json" \
    -d '{"text": "नमस्ते दुनिया", "validate": true}' \
    --output hello.wav
```

Response is `audio/wav` with metadata in response headers:

- `X-Sample-Rate`: sample rate
- `X-Retries`: retries used
- `X-Validation-Passed`: `true` / `false`
- `X-Validation-CER`: CER if validation ran

**POST /render-srt** — upload SRT → timed audio

```bash
curl -X POST http://127.0.0.1:8765/render-srt \
    -F "srt=@episode.srt" \
    -F "mode=fit_to_cue" \
    --output episode.wav
```

## Performance tuning

### Speed

Inference speed on 12GB GPU with VITS:

| Setup | RTF (lower is faster) |
|---|---|
| Default (fp32, no compile) | 0.35× |
| fp16 | 0.25× |
| fp16 + torch.compile after warmup | 0.20× |

For your Story Recap YouTube pipeline, that means a 20-minute episode renders
in 4–7 minutes depending on settings.

### Memory

Base inference uses ~2GB VRAM. Multiple concurrent requests to the HTTP server
share one loaded model — no extra VRAM per request.

### Disable validation in production

Validation adds ~1s per sentence because Whisper large-v3 is big. If your
production is latency-sensitive and you trust the model at this point, disable
it:

```python
engine = TTSEngine.load(engine_dir, enable_roundtrip=False)
```

Or use a lighter Whisper model for validation:

```python
# (future work — not in v1)
```

## Troubleshooting

**`TTSEngine.load` raises FileNotFoundError**
→ You haven't exported yet. Run `hindi-tts-builder export my_voice`.

**Engine version mismatch**
→ You trained with one package version and are loading with another. Either
  retrain with the current version or downgrade to match the engine.

**Audio is silent or garbled**
→ Usually a tokenizer mismatch. The engine's `tokenizer.json` must match
  what the model was trained with. Re-export from the same training run.

**Words omitted in output despite validation**
→ Edge case: validation passed but the model still dropped something. File
  a bug with the problematic text. Workaround: split long sentences at
  commas/dandas.

**Slow on first speak() call**
→ First call loads the model (~5s) and warms up any compiled kernels.
  Subsequent calls are fast. Keep the engine object around.
