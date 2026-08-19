# Voice Library — multi-speaker TTS via LoRA adapters

Train one LoRA adapter per speaker on top of base Indic-Parler-TTS. Each adapter is **~50 MB** instead of **~3.5 GB** (full fine-tune per voice). Hot-swap voices at inference time.

```
voice_library/
├── adapters/
│   ├── male_narrator_16h/                # ~50 MB
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── voice_card.json
│   ├── female_dramatic_150h/             # ~50 MB
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── voice_card.json
│   └── ...
```

Storage cost for 10 voices = base (3.5 GB) + 10 × 50 MB = **4 GB total** (vs 35 GB if every voice was a full fine-tune).

## Workflow

### 1. Train a LoRA adapter per voice

For each speaker, you need a project with that speaker's audio + SRTs ingested (use `corpus-ingest` to do this in bulk).

```bash
# Voice A: male narrator from your existing 16h corpus
hindi-tts-builder train-indic-parler-lora h_tts_1 \
    --profile 10h_clean \
    --adapter-name male_narrator_16h \
    --max-steps 30000 \
    --rank 32 --alpha 64 --lr 1e-4

# Voice B: female narrator from a separate 150h corpus
hindi-tts-builder train-indic-parler-lora h_tts_150h_female \
    --profile 100h_clean \
    --adapter-name female_dramatic_150h \
    --max-steps 50000 \
    --rank 32 --alpha 64 --lr 1e-4
```

Notes:
- LoRA needs more steps than full FT (typical 20k-50k). Default 30,000.
- LoRA tolerates higher LR (1e-4 vs full FT's 5e-6). Default 1e-4.
- `--rank 32` is a good starting point. Lower (16) = smaller adapter + faster but weaker; higher (64) = stronger but bigger.

### 2. Bundle into the per-project voice library

```bash
hindi-tts-builder voice-export h_tts_1 --adapter-name male_narrator_16h
# → projects/h_tts_1/voice_library/adapters/male_narrator_16h/
```

### 3. Register in the global voice library

```bash
hindi-tts-builder voice-add \
    projects/h_tts_1/voice_library/adapters/male_narrator_16h \
    --library-dir /path/to/global/voice_library
```

Repeat for each voice. The global library is just a directory you point at — keep it wherever convenient.

### 4. List + remove

```bash
hindi-tts-builder voice-list --library-dir /path/to/global/voice_library
# → male_narrator_16h    rank=32 steps=30000 profile=10h_clean
# → female_dramatic_150h rank=32 steps=50000 profile=100h_clean

hindi-tts-builder voice-remove female_dramatic_150h \
    --library-dir /path/to/global/voice_library
```

### 5. Synthesize with a specific voice

```bash
hindi-tts-builder speak-voice "नमस्ते दुनिया।" \
    --voice male_narrator_16h \
    --out hello_male.wav \
    --library-dir /path/to/global/voice_library

hindi-tts-builder speak-voice "नमस्ते दुनिया।" \
    --voice female_dramatic_150h \
    --out hello_female.wav \
    --library-dir /path/to/global/voice_library
```

**Voice swap is fast** — base Parler is loaded once into VRAM, adapter swap takes ~100ms (no model reload). Render multiple voices in one process by reusing the same `VoiceLibrary` instance from Python.

## Python API

```python
from hindi_tts_builder.inference.backends.voice_library import VoiceLibrary

lib = VoiceLibrary(library_dir="/path/to/voice_library")
print(lib.list_voices())  # ['male_narrator_16h', 'female_dramatic_150h']

# Render same text in two voices
audio_m, sr = lib.synthesize("यह एक परीक्षण है।", voice="male_narrator_16h")
audio_f, sr = lib.synthesize("यह एक परीक्षण है।", voice="female_dramatic_150h")

# Or to file
lib.speak_to_file("...", "out.wav", voice="male_narrator_16h")

# Custom description (overrides voice card default)
lib.synthesize(
    "...",
    voice="male_narrator_16h",
    description="A male Hindi speaker reads slowly with a soft, calming tone.",
)
```

## When to use LoRA vs full fine-tune

| Use case | Pick |
|---|---|
| **One voice, best possible quality** | Full FT (`train-indic-parler`) |
| **Multiple voices** | LoRA library |
| **Fast iteration on data/hyperparams** | LoRA (50 MB checkpoints, easy A/B) |
| **Want to keep base Parler's emotional range** | LoRA |
| **Worried about catastrophic forgetting** | LoRA |
| **Have 100h+ of one speaker, want max quality** | Full FT |

## What goes in a `voice_card.json`

```json
{
  "adapter_name": "male_narrator_16h",
  "speaker_name": "Divya",
  "default_description": "Divya speaks Hindi in a clear, close-sounding narrator voice...",
  "base_model": "ai4bharat/indic-parler-tts",
  "lora_rank": 32,
  "lora_alpha": 64,
  "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
  "training_profile": "10h_clean",
  "training_steps": 30000,
  "learning_rate": 1e-4,
  "sample_rate": 44100,
  "language": "hi"
}
```

The voice card is what `speak-voice` reads to pick the right description prompt + sample rate per voice.

## Hard constraints / things to know

1. **Run inference from the parler venv** — the same as regular `speak-indic-parler`. The voice library imports `parler_tts` and `peft`, which only live in `/root/parler-venv/`.
2. **Adapter is tied to base model** — adapters trained against `ai4bharat/indic-parler-tts` won't load against the pre-trained variant. Don't mix.
3. **Cross-voice consistency** — voice-card descriptions matter. Even with the same adapter, a different description string changes prosody/emotion. Keep descriptions consistent within an adapter unless you're deliberately exploring style.
4. **First adapter swap is slower** (~1s) — peft wraps the base model on first load. Subsequent swaps are ~100ms.
5. **rank tradeoff** — rank=16 saves disk but loses speaker similarity vs rank=32. rank=64+ has diminishing returns on speakers; useful for harder voice characteristics (very different accent/timbre from base).
