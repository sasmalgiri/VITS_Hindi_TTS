# TASKS.md — Improvement Backlog

Explicit specifications for each planned improvement to hindi-tts-builder.
Each task is independent and has its own acceptance criteria. Tasks are
ordered by priority — do them top-down unless there's a reason to skip.

Estimates assume Claude Code working with full filesystem/bash access on a
Windows 11 + WSL2 Ubuntu 22.04 setup with a 12GB NVIDIA GPU.

---

## T1: Warm-start training from MMS-TTS Hindi

**Status: DEFERRED (2026-04-23).** The spec assumes a clean
same-architecture warm-start, but MMS-TTS Hindi (Meta's MMS variant) and
Coqui VITS have incompatible tensor names and shapes across the text
encoder and duration predictor. The token vocabularies also don't line up
(our Devanagari chars vs MMS's uroman/IPA). Net useful weight transfer
is small and the vocoder transfer risks acoustic-domain pollution
(MMS-TTS was trained on Bible recordings). The projected MOS gain
(4.2 → 4.4) assumes a clean transfer that isn't available with these two
models as-is.

Revisit only if (a) a Coqui-native pretrained Hindi VITS checkpoint
becomes available, or (b) the T2 smoke test surfaces a real reason to
accelerate convergence. Meanwhile the trainer continues to initialize
from scratch — proven, deterministic, and risk-free.

**Priority:** P0 (do first)
**Estimate:** ~1 day of iteration + real-hardware testing
**Why:** The current trainer initializes VITS from scratch. Warm-starting
from a Hindi-pretrained base model is the single biggest quality-per-hour
improvement available. MOS ceiling rises from ~4.2 to ~4.4+ and training
time drops from 5–7 days to 1.5–2 days.

### Deliverables

1. New module: `hindi_tts_builder/train/warmstart.py`
2. Modified: `hindi_tts_builder/train/config.py` — adds `warm_start` field
3. Modified: `hindi_tts_builder/train/trainer.py` — loads warm-start weights
4. Modified: `hindi_tts_builder/cli/main.py` — adds flags
5. New tests: `tests/test_warmstart.py` — adapter logic without GPU
6. Updated: `docs/TRAINING.md` — warm-start section
7. Updated: `CHANGELOG.md` — v1.1.0 entry

### Specification

**`warmstart.py` public API:**

```python
def download_base_model(name: str, cache_dir: Path) -> Path:
    """Download a base model from Hugging Face. Returns local path.

    Supported names (initially):
        'mms-tts-hi' → facebook/mms-tts-hin

    Idempotent: if already cached, returns cached path.
    """

def adapt_checkpoint(
    *,
    base_checkpoint: Path,
    our_tokenizer,                 # HindiTokenizer
    single_speaker_mode: bool = True,
) -> dict:
    """Load a base VITS checkpoint and adapt to our tokenizer + speaker setup.

    Specifically:
      - Remap text embedding matrix: for each token in our vocab, copy the
        corresponding row from the base if it exists, otherwise initialize
        randomly (standard normal * 0.1)
      - If single_speaker_mode: replace multi-speaker embedding with single
        learned embedding initialized to the mean of base speaker embeddings
      - Keep all other weights (encoder, decoder, flow, vocoder) as-is

    Returns state_dict compatible with our Coqui VITS model.
    """

def apply_freeze_schedule(
    model,
    step: int,
    *,
    freeze_text_encoder_until: int = 20_000,
) -> None:
    """Freeze/unfreeze components based on current step.

    Called at each training step. Freezes text encoder for first N steps
    so warm-start weights stabilize before drifting.
    """
```

**`TrainingConfig` additions:**

```python
warm_start: str | None = "mms-tts-hi"   # or None / "none" for from-scratch
freeze_text_encoder_steps: int = 20_000
```

**`Trainer.train()` changes:**

1. Before calling `Vits.init_from_config(config)`:
   - If `warm_start` is set: download base model, adapt checkpoint,
     load state_dict into the Vits model
2. Register a training-step callback that calls `apply_freeze_schedule`
3. Log to `logs/train.log` which mode is being used and the base model name

**CLI flags:**

```
hindi-tts-builder train my_voice --warm-start mms-tts-hi  # default
hindi-tts-builder train my_voice --from-scratch           # opt-out
hindi-tts-builder train my_voice --warm-start-from <path> # custom checkpoint
```

### Acceptance criteria

- [ ] `pytest tests/` passes (all 208 existing tests still green)
- [ ] New tests in `test_warmstart.py` cover:
  - download_base_model uses cache (no double download)
  - adapt_checkpoint handles vocab size mismatch correctly
  - adapt_checkpoint handles missing tokens (random init)
  - apply_freeze_schedule actually freezes and unfreezes
- [ ] `hindi-tts-builder doctor` confirms `huggingface_hub` is installed
- [ ] Running `hindi-tts-builder train --prepare-only` in a smoke-test
      project logs "warm-start: mms-tts-hi, downloaded to: ..."
- [ ] `docs/TRAINING.md` has a "Warm-start vs from-scratch" section
- [ ] `CHANGELOG.md` has a v1.1.0 entry

### Testing plan

1. Unit tests (no GPU needed): adapter logic with fake state_dicts
2. Integration test: actually download MMS-TTS Hindi once, confirm
   adapter produces a loadable state_dict
3. Smoke test: train 500 steps with warm-start on a tiny dataset (10
   clips), confirm loss decreases and no NaN

### Known risks

- MMS-TTS checkpoints may have different tensor shapes than Coqui VITS
  expects. If so, write a shape-translation layer. Do not try to force
  incompatible shapes; prefer partial warm-start (just the text embedding
  + encoder) over a broken full warm-start.
- Hugging Face model downloads can be large (~600MB). Cache under
  `~/.cache/huggingface/` not inside the project.

---

## T2: Real-hardware end-to-end smoke test

**Priority:** P0 (do second)
**Estimate:** 3–5 hours
**Why:** Everything in the Claude.ai chat sandbox was Linux-only and
GPU-free. There will be Windows/WSL2-specific bugs that only surface on
real hardware. Catching them with a tiny dataset is cheap; catching them
mid-50h-training is expensive.

### Deliverables

1. New script: `scripts/smoke_test.sh` (or `.ps1` — pick one)
2. New doc: `docs/SMOKE_TEST.md` with exact steps

### Specification

Create a fresh throwaway project with 2–3 public domain Hindi YouTube URLs
(pick short ones, 3–5 minutes each) and their SRT transcripts. Run the
full pipeline end-to-end:

```bash
hindi-tts-builder doctor
hindi-tts-builder new smoke_test
hindi-tts-builder add-sources smoke_test --urls smoke_urls.txt --transcripts smoke_srts/
hindi-tts-builder prepare smoke_test
hindi-tts-builder train smoke_test  # edit training_config.yaml first: max_steps=1000
hindi-tts-builder export smoke_test
hindi-tts-builder speak smoke_test --text "यह एक परीक्षण है।" --out test.wav
```

Budget: 1–2 hours for download, ~30 minutes for 1000-step training on
12GB GPU, then ~5 minutes for inference.

### Acceptance criteria

- [ ] Every stage completes without errors
- [ ] `test.wav` is generated and playable
- [ ] Round-trip validation runs (even if it fails at 1000 steps — the
      point is that it runs, not that output is good yet)
- [ ] `hindi-tts-builder serve smoke_test` starts and `/health` returns 200
- [ ] Any bugs discovered get fixed AND added to `docs/TROUBLESHOOTING.md`

### What failure looks like and how to respond

- **`yt-dlp` errors:** usually geo/age-restriction; note URL in troubleshooting
- **`ffmpeg` path errors:** update docs; consider bundling an ffmpeg check in `doctor`
- **CUDA OOM at batch_size=16:** reduce default batch_size; update TRAINING.md
- **WhisperX import failures:** improve the fallback message; confirm `--no-whisperx` path works
- **Filesystem permission errors:** likely a Windows/WSL2 mount issue; document the fix

---

## T3: Frontend regression tests with real SRT samples

**Priority:** P1
**Estimate:** 2–3 hours
**Why:** The current frontend tests use synthetic examples. Real translated
Hindi SRT content has edge cases (mixed scripts, quirky punctuation from
translators, dubbing-specific formatting) that synthetic tests miss.

### Deliverables

1. New folder: `tests/fixtures/real_srts/` with 5–10 anonymized SRT cues
2. New test: `tests/test_frontend_real_samples.py`

### Specification

Collect 20–30 real Hindi SRT lines from diverse sources (your own dubbing
projects, anonymized). For each:

1. Hand-write the expected frontend output
2. Add a test that runs the line through `HindiFrontend` and checks the
   output matches

These become regression tests — if a frontend change ever breaks
real-world behavior, these tests catch it immediately.

### Acceptance criteria

- [ ] At least 20 real-world test cases
- [ ] Test cases cover: numeric content, mixed Latin/Devanagari, unusual
      punctuation, long sentences, questions, exclamations
- [ ] All tests pass with current frontend (i.e., frontend doesn't need
      changes to accommodate them — if it does, that's a bug to fix)

---

## T4: Inference speed optimizations

**Priority:** P1
**Estimate:** 1 day
**Why:** Default inference runs at RTF ~0.3 on 12GB GPU. For a 20-minute
YouTube episode that's 6 minutes of rendering. Can reduce to 2–3 minutes
with optimizations already sketched in `docs/INFERENCE.md`.

### Deliverables

1. Modified: `hindi_tts_builder/inference/engine.py`
2. New module: `hindi_tts_builder/inference/optimize.py`
3. New tests: `tests/test_optimize.py`

### Specification

Add three opt-in optimizations to `TTSEngine`:

**FP16 inference:**
```python
engine = TTSEngine.load(engine_dir, precision="fp16")
```
Cast model to half precision after load. Expect ~30% speedup, no quality
loss on VITS.

**torch.compile():**
```python
engine = TTSEngine.load(engine_dir, use_torch_compile=True)
```
Call `torch.compile()` on the model. First generation is slow (~30s
warmup), subsequent generations are 20% faster. Skip on PyTorch <2.0.

**Batch inference:**
```python
results = engine.speak_batch(["sent1", "sent2", "sent3"])
```
Process multiple texts in a single forward pass. Padding-aware.

### Acceptance criteria

- [ ] All existing inference tests still pass
- [ ] FP16 mode measurably faster and produces audio of similar quality
      (audio correlation >0.99 with FP32 reference)
- [ ] `torch.compile()` mode works on 2.0+ and falls back cleanly on older
- [ ] `speak_batch()` correctly handles padding and produces same per-item
      output as individual `speak()` calls

---

## T5: StyleTTS 2 backend (v2 milestone)

**Priority:** P2 (do after v1.1 proves out)
**Estimate:** 1–2 weeks
**Why:** The roadmap's v2. Higher quality ceiling (~MOS 4.55+) and natural
support for emotion-via-reference-audio.

### Deliverables

1. New package: `hindi_tts_builder/train_styletts2/` (parallel to `train/`)
2. New inference: `hindi_tts_builder/inference/styletts2_engine.py`
3. CLI flag: `hindi-tts-builder train --model styletts2`
4. Engine manifest supports `model_type: styletts2`
5. Engine loader dispatches correctly based on manifest
6. Tests for all new modules

### Specification (high level — expand when starting)

StyleTTS 2 ([Li et al. 2023](https://arxiv.org/abs/2306.07691)) is a
diffusion-based TTS with better prosody than VITS. The reference
implementation lives at `yl4579/StyleTTS2` on GitHub.

Port the same data pipeline to feed StyleTTS 2 training. Keep all frontend,
tokenizer, and data-pipeline code identical — only the acoustic model
changes. The `TTSEngine` interface stays the same; users shouldn't need to
know which model is under the hood.

Add emotion control:
```python
engine.speak("दुख से बोला।", reference_audio="sad_voice_sample.wav")
```

### Acceptance criteria

- [ ] `hindi-tts-builder train my_voice --model styletts2` works
- [ ] `hindi-tts-builder speak my_voice` works identically for either
      model type (no API change for basic use)
- [ ] Engine manifests from v1 VITS models still load correctly
- [ ] Reference-audio emotion control documented and tested
- [ ] A/B comparison test against VITS baseline on the same training
      data shows measurable MOS improvement

---

## T6: Evaluation against commercial baselines

**Priority:** P2
**Estimate:** 1 day (after you have a trained model)
**Why:** The project's goal is "beat all others." Without measured
comparisons, you don't know if you did.

### Deliverables

1. Extended: `hindi_tts_builder/eval/baselines.py` — new module
2. CLI: `hindi-tts-builder eval my_voice --vs google,elevenlabs,indic-parler`
3. New doc: `docs/EVALUATION.md`

### Specification

Build runners for each baseline system:

- **Google Cloud TTS** (free tier, `hi-IN-Neural2-B` voice)
- **ElevenLabs Multilingual v2** (free tier, Hindi voice)
- **Sarvam Bulbul** (free credits)
- **Indic-Parler-TTS** (local, free)
- **MMS-TTS Hindi** (local, free)

For each baseline:
1. Generate audio for every item in the locked TestSet
2. Transcribe with Whisper-large-v3
3. Compute CER, WER, UTMOS
4. Save per-system report

Comparison report shows your model's scores next to each baseline, with
per-category and overall aggregates.

### Acceptance criteria

- [ ] Handles missing API keys gracefully (skips that baseline with note)
- [ ] Respects free tier limits (checks rate-limit headers)
- [ ] Generates a final markdown report comparing all systems
- [ ] `docs/EVALUATION.md` documents how to set up each API key

---

## T7: Docker image (optional, for distribution)

**Priority:** P3 (only if you ever want to share the tool)
**Estimate:** 3–4 hours
**Why:** One-command deploy for anyone who wants to try the tool.

### Deliverables

1. `Dockerfile` at project root
2. `docker-compose.yml` for GPU passthrough
3. `docs/DOCKER.md`

### Specification

CUDA 12.1 base image, installs all dependencies, mounts a volume for
projects. Single command:

```bash
docker-compose run tts-builder doctor
docker-compose run tts-builder new my_voice
# ...etc
```

### Acceptance criteria

- [ ] Image builds in <10 minutes from scratch
- [ ] GPU is accessible inside container (test with `doctor`)
- [ ] Projects persist across container runs (volume mount)
- [ ] Image size under 8 GB

---

## Ground rules for all tasks

These apply to every task above:

1. **Tests stay green.** `pytest tests/ -q` must pass before and after
   every task. Run it in a tight loop during development.

2. **No scope creep.** If you discover something that should be fixed but
   isn't part of the current task, add it to this file as a new task and
   stay focused.

3. **Docs are code.** If behavior changes, docs change in the same commit.

4. **CHANGELOG is source of truth.** Every user-visible change gets a
   CHANGELOG entry before the task is considered done.

5. **No placeholder code.** If you can't fully implement something,
   leave it out entirely and document why in the task.

6. **Real-hardware bugs are gold.** When you find one, fix it AND add an
   entry to `docs/TROUBLESHOOTING.md` so the next person doesn't lose
   a day on it.

7. **Ask if unclear.** Better to confirm intent than ship the wrong fix.

## Task status

Update this table as tasks are completed:

| Task | Status | Completed on | Notes |
|---|---|---|---|
| T1: Warm-start | 🚫 Deferred | 2026-04-23 | MMS-TTS/Coqui VITS arch mismatch — see T1 note |
| T2: Smoke test | ✅ Complete | 2026-04-23 | Caught + fixed: cuDNN 8/9 coexistence, WhisperX OOM, stale Windows listener, URL↔SRT pair UI. Real model training blocked on matching-language SRT (data, not code). |
| T3: Real SRT tests | ⏳ Pending | — | — |
| T4: Inference speed | ⏳ Pending | — | — |
| T5: StyleTTS 2 | ⏳ Pending | — | — |
| T6: Baseline eval | ⏳ Pending | — | — |
| T7: Docker | ⏳ Pending | — | — |
