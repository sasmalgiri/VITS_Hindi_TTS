# Roadmap — Hindi TTS Builder

Authoritative plan as of **2026-04-26**, replacing the looser "v5 / v6" sketch in [PROJECT.md](PROJECT.md). The goal is a **serious Hindi narrator model** suitable for full-length YouTube dubbing — not a demo.

The pipeline is the asset. Don't rebuild it. Use it to produce clean staged datasets, freeze Coqui-VITS as baseline, pivot fine-tuning to Indic-Parler-TTS, and benchmark every model on the same Hindi-dubbing test set.

---

## Immediate operating rules

### Rule 1 — Don't `--skip-qc` on final builds

`--skip-qc` is a quick-experiment escape hatch. The final 100-h build must run full QC because clean SRT cannot detect:
- clipped first/last syllables
- long internal silence
- corrupt audio chunks
- sudden volume jumps
- bad yt-dlp extractions
- broken SRT cue timing
- accidental non-speech segments

For known-clean SRTs, **loosen QC thresholds, don't disable QC**. Knobs in `projects/<name>/config.yaml`:
```yaml
qc:
  min_snr_db: 10           # was 15 — loosen for narration
  max_cer_vs_whisper: 0.35 # was 0.25 — loosen for accent/Whisper-Hindi noise
  max_silence_ratio: 0.5
```

### Rule 2 — Trust SRT timing, but trim silence

When SRTs are word-to-word and verified, **don't run WhisperX alignment** — it can introduce drift on already-perfect timing. Use the new `--trust-srt` flag (see segment changes below):
- Cut directly from SRT timestamps
- Pad **50 ms left, 100 ms right** to capture leading consonant attacks and trailing breath
- Trim silence *after* cutting (not before — preserves the pad)

`--trust-srt` skips Stage 2 (align) entirely and feeds the raw SRT directly into Stage 3 (segment) with the padded-cut policy.

### Rule 3 — Audit before you train

Add a `hindi-tts-builder audit <name>` command that prints:
- total_raw_hours / total_clean_hours / total_clips
- accepted_clips / rejected_clips (with reason histogram)
- mean_duration / p95_duration / clips_under_1s / clips_over_10s
- loudness_min / loudness_max
- text_char_coverage (vs `CharactersConfig`)
- speaker_consistency_score (mean cosine-sim of x-vectors across N random clips — proxy for "is this really one voice")

This catches the silent-collapse bugs cheaper than a 16-hour training run.

### Rule 4 — Build staged datasets, not one giant dataset

From the same 100-h corpus, produce profiles by sampling:
- `2h_clean`   — sanity check + tokenizer fit
- `10h_clean`  — minimum viable Hindi voice
- `30h_clean`  — common sweet-spot for fine-tunes
- `60h_clean`  — diminishing returns above this
- `100h_clean` — full corpus

Selection logic: random-sample without replacement, stratified by source so each profile gets diversity. Reproducible via `seed`.

Why: the best-sounding model is often **30 h or 60 h, not 100 h**. The same model fine-tuned on 30 h of clean data routinely beats the same model on 100 h with mixed quality.

---

## Dual-venv setup (required as of 2026-04-26)

`parler-tts 0.2.2` hard-pins `transformers==4.46.1`. `coqui-tts 0.27.5` requires `transformers>=4.57`. They are mutually exclusive in one venv. The pipeline runs across **two isolated environments**:

| Venv | Python | Purpose | Key pins |
|------|--------|---------|----------|
| `/root/hindi-tts/venv` (the existing one) | `python3.10` | All data-pipeline work + Coqui-VITS training/inference + dataset exporters (the exporter only needs soundfile/torchaudio, which are framework-agnostic) | `transformers==4.57.6`, `coqui-tts==0.27.5`, `torch==2.5.1+cu121` |
| `/root/parler-venv` (new) | `python3.10` | Parler-TTS training subprocess + Parler inference (`speak-indic-parler`) | `transformers==4.46.1`, `parler_tts==0.2.2`, `torch==2.5.1+cu121` |

Both venvs share the same CUDA torch wheel so they can run side-by-side on the same RTX 3060 (only one at a time uses the GPU).

`IndicParlerBackend` runs in the **coqui** venv (its `prepare()` only needs the exporter). When it calls `accelerate launch ./training/run_parler_tts_training.py`, it spawns the subprocess with the **parler** venv's Python, resolved in this order:
1. `IndicParlerBackendConfig.python_executable` (explicit)
2. `HINDI_TTS_PARLER_PYTHON` env var
3. Default: `/root/parler-venv/bin/python3`

For Parler **inference** (`speak-indic-parler`), the CLI must be invoked from the parler venv directly:
```bash
/root/parler-venv/bin/python3 -m hindi_tts_builder.cli.main speak-indic-parler ...
```
The CLI handler detects the wrong-venv case and prints the corrected command.

To set up the parler venv from scratch:
```bash
python3 -m venv /root/parler-venv
/root/parler-venv/bin/pip install --upgrade pip wheel
/root/parler-venv/bin/pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
/root/parler-venv/bin/pip install -r /root/hindi-tts/requirements-indic-parler.txt
/root/parler-venv/bin/pip install -e /root/hindi-tts   # so it can find hindi_tts_builder.*
```

---

## Backend adapter layer (the architectural change)

The trainer is currently Coqui-VITS-locked. Refactor to an adapter pattern:

```
hindi_tts_builder/train/backends/
├── __init__.py            registry: BACKENDS = {"coqui-vits": ..., "indic-parler": ..., "f5-hindi": ...}
├── base.py                BaseTrainerBackend ABC: prepare(), train(), export(), name, supports_resume
├── coqui_vits.py          current trainer.py logic, lifted in
├── indic_parler.py        AI4Bharat Indic-Parler-TTS fine-tune
├── f5_hindi.py            SPRINGLab F5-Hindi-24KHz fine-tune
└── kokoro.py              (placeholder, low priority)
```

CLI becomes:
```bash
hindi-tts-builder train divya --backend coqui-vits     # current default
hindi-tts-builder train divya --backend indic-parler   # v5 primary
hindi-tts-builder train divya --backend f5-hindi       # v5 secondary
```

The dataset format also generalizes — the data pipeline becomes a dataset factory:
```
projects/<name>/training_set/
├── clips_24k/             current
├── clips_48k/             added when --resample-48k given
├── train.csv  val.csv  test.csv          (Coqui)
├── train.jsonl val.jsonl test.jsonl      (Parler / F5 — adds `description` for Parler)
└── metadata.parquet                       (one source-of-truth, the others derived)
```

---

## v4 — current VITS-from-scratch run (baseline)

Don't blindly let it run to 150 k. Sample-check at milestones:

### At step ≈ 30,000
Synthesize the Hindi-dubbing benchmark pack (`eval/hindi_dubbing_100.jsonl` — 10 categories): 8 held-out sentences, 1 short SRT scene, 1 emotional paragraph, 1 Hinglish paragraph, 1 names-heavy paragraph.

If output is robotic noise → **stop and pivot** to the adapter layer + Parler track.

### At step ≈ 60,000
Render a 3-minute narration. Check:
- pronunciation
- speaker stability
- duration drift
- background noise / artifacts
- word skipping
- Hindi clarity (especially conjuncts and aspirated stops)

### At step 100,000–150,000
Only keep cooking if 60 k samples are clearly promising. Otherwise pivot.

### Freeze v4 as baseline
```bash
hindi-tts-builder export divya
hindi-tts-builder speak divya "जियांग यूएछू ने पेई चांगछिंग की ओर देखा।" --out test_name.wav
hindi-tts-builder speak divya "उसके पास बारह करोड़ पैंतालीस लाख रुपये थे।" --out test_number.wav
```
These can sound bad. They're a baseline.

---

## v5 — pivot to a stronger Hindi base model

### Primary: Indic-Parler-TTS (AI4Bharat, 0.9 B, Apache 2.0)
- Hindi-targeted training (1,806 h Indic data including IndicTTS, LIMMITS, Rasa)
- Apache 2.0 — commercial use clean
- Single-speaker fine-tune recipe documented
- Tight on 12 GB VRAM — needs bf16 + grad_accum=8 at batch=2–4

### Secondary: F5-Hindi-24KHz (SPRINGLab IIT-Madras, 151 M, CC-BY-4.0)
- Native 24 kHz Hindi
- Smaller, faster iteration on RTX 3060
- License caveat: training data (IndicVoices-R) terms must be verified before any commercial release

### Skipped
- ❌ XTTS v2 — CPML non-commercial + Coqui dead Jan 2024
- ❌ MMS-TTS-hin — CC-BY-NC, mechanical quality
- ❌ OpenVoice v2 / MeloTTS — no Hindi
- ❌ StyleTTS 2 — no Indic checkpoint, heavy lift

---

## The Hindi-dubbing benchmark pack

Single source of truth for **every** model comparison. Lives at `eval/hindi_dubbing_100.jsonl`. Each entry:
```json
{"id": "n01", "category": "fantasy_names", "text": "जियांग यूएछू ने पेई चांगछिंग की ओर देखा।", "notes": "Chinese names transliterated"}
```

Categories (10):
1. `pure_hindi` — simple narration
2. `hinglish` — Hindi+English mix common in YouTube
3. `numbers` — lakh/crore, dates, times, prices (with `₹`)
4. `fantasy_names` — Chinese/Korean/Japanese names transliterated to Devanagari (donghua use case)
5. `suspense` — slow narration, pauses
6. `fear` — high arousal, breathy
7. `anger` — high arousal, low pitch
8. `soft` — calm, ASMR-ish
9. `long_form` — 200+ word paragraphs (tests duration drift)
10. `srt_short` — 5-cue SRT scene

Every future model commit must run against this pack and append a row to `eval/results.csv`:

| Model | Data (h) | Steps | mean_CER | RTF | manual_MOS | long_form_stable | notes |
|-------|----------|-------|----------|-----|-----------|------------------|-------|
| coqui_vits | 16 | 30k | … | … | … | … | baseline |
| indic_parler | 10 | 5k | … | … | … | … | |

---

## Inference improvements (v5+)

### Failed-cue retry logic
For YouTube dubbing, inference cannot just emit audio. It must self-validate:

```python
def speak_with_retry(engine, text, max_retries=3):
    out = engine.speak(text)
    if out.cer > CER_THRESHOLD:
        out = engine.speak(normalize(text))            # retry 1: aggressive normalize
    if out.duration > expected * 1.4:
        out = engine.speak(rewrite_shorter(text))      # retry 2: shrink
    if name_dropped(text, out):
        out = engine.speak(apply_pronunciation(text))  # retry 3: dict lookup
    if out.failed():
        mark_for_manual_review(text, out)
    return out
```

### Pronunciation dictionary (promoted to first-class)
Currently optional in the engine bundle. Make it central:
```json
{
  "Jiang Yuechu": "जियांग यूएछू",
  "Pei Changqing": "पेई चांगछिंग",
  "Xiao Yan": "शियाओ यान",
  "Tang San": "तांग सान",
  "system": "सिस्टम",
  "activate": "एक्टिवेट"
}
```
Frontend auto-loads `engine/pronunciation_dict.json` if present. Fail closed if the manifest declares one and the file is missing — silent fallback is what got us into the v3 NaN-collapse pattern.

### Long-form eval command
```bash
hindi-tts-builder eval-long divya --srt test_20min.srt
```
Outputs:
- `failed_cues.csv` — cue_index, text, reason, cer, duration_diff
- `duration_mismatch.csv` — cues whose synth duration differs from cue duration by > 30%
- `cer_report.csv` — per-cue CER
- `final_mix.wav` — full mux for ear inspection

A model that sounds great on 10-second clips can still fall apart in a 30-minute video.

---

## What we are NOT doing

- **Not** rebuilding the data pipeline. It works.
- **Not** chasing more random base models. Indic-Parler + F5-Hindi cover the design space.
- **Not** training only from scratch. VITS-from-scratch is the baseline, not the goal.
- **Not** adding more UI features until the model side is solid. The studio is good enough.

---

## Verdict

With the user's claimed dataset condition (100 h, single speaker, word-to-word SRT, no background music), the realistic target is a **serious Hindi narrator model**, not a demo. The path:

1. Freeze Coqui-VITS v4 as baseline.
2. Build the staged datasets (2/10/30/60/100 h).
3. Pivot fine-tuning to Indic-Parler-TTS as v5 primary, F5-Hindi as quality benchmark.
4. Evaluate everything on `eval/hindi_dubbing_100.jsonl`.
5. Ship the winner with retry logic + pronunciation dict + long-form eval.
