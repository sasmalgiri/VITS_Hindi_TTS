# Architecture Notes

This document records WHY key design decisions were made. Useful for your
future self when you forget why something was built a specific way.

## Why VITS for v1, StyleTTS 2 for v2

VITS is a non-autoregressive, duration-based TTS model. For each input token,
it predicts an explicit duration and generates audio for all tokens in
parallel. This architecture **cannot structurally drop a word** — every token
gets a guaranteed slot in the output.

Autoregressive models (XTTS, Tortoise, Bark, Tacotron) generate audio
token-by-token with attention over the input. Attention can collapse, causing
chunks of text to be silenced or skipped. Known failure mode.

Given the hard requirement "never miss or omit a word," autoregressive models
are ruled out. Among non-autoregressive choices:

- **VITS**: simpler, more battle-tested, end-to-end (text + vocoder in one model), 83M params, fits comfortably on 12GB VRAM, ~3× realtime inference.
- **FastSpeech2 + vocoder**: fastest but flattest prosody.
- **StyleTTS 2**: 200M params, highest quality ceiling with diffusion-based prosody, but more complex to train and slightly slower inference.

v1 = VITS gets us to working product in 5 days of training.
v2 = StyleTTS 2 gets us higher naturalness later, reusing all data and frontend.

## Why a custom Hindi cardinal writer instead of num2words

`num2words` does not implement Hindi and raises `NotImplementedError`. We
implement Indian-numbering-system cardinals directly: thousand, lakh, crore,
arab. The `frontend.hindi_num` module is pure Python, no dependencies,
deterministic.

## Why rule-based schwa deletion instead of a neural model

Hindi schwa deletion has good linguistic rules that get ~95% accuracy. The
remaining 5% are irregular forms (Sanskrit loanwords, names) that a neural
model wouldn't reliably fix anyway — and would add a training dependency.

If v2 needs higher accuracy, plug in a ByT5-small fine-tuned on the
IIT-KGP schwa deletion dataset. The frontend interface is stable: it takes
Devanagari in and returns Devanagari + halants. Internals can be swapped.

## Why prosody tokens instead of letting the model learn from raw punctuation

Two reasons:

1. **Deterministic behavior**: the same punctuation always produces the same token, so acoustic output is reproducible across runs, models, and checkpoints.
2. **Explicit control**: you can manually inject `<emphasis>` tokens mid-sentence for deliberate dramatic effect without needing to rewrite the punctuation.

The tokens are plain ASCII so they survive every pipeline step unchanged
(assuming we don't accidentally strip `<`, `>`, or `_` — which we test for
explicitly).

## Why round-trip Whisper validation at inference time

Even with a non-autoregressive model (architectural omission guarantee), edge
cases exist: a rare conjunct the model has never seen, a novel proper noun,
a number that's at the edge of the training distribution. Round-trip
validation catches these by running Whisper-large-v3 on the generated audio
and comparing the transcription to the input. If CER > 2%, regenerate with a
different random seed. If still failing after N retries, flag for manual
review.

This is cheap (Whisper-large-v3 on a single 10-second clip runs in ~1 second
on a 12GB GPU) and is the final guarantee of the "zero missed words"
requirement.

## Why WSL2 not native Windows

Not a quality difference — GPU compute is identical. It's about training
wall-clock:

- Native Windows `num_workers > 0` uses `multiprocessing.spawn()` which
  re-imports the whole process per worker. Slow startup, slow data loading.
- Linux (including WSL2) uses `fork()` which is near-instant. `num_workers=4`
  works without issues.
- NTFS is 2–5× slower than ext4 for the tens of thousands of small files we
  produce during segmentation.

Practical impact on 50 hours of training data: WSL2 saves ~2 days of
training time.

## Why no emotion-via-reference-audio in v1

VITS doesn't have clean style conditioning. Bolting on a reference encoder
would compromise training quality for a feature that works naturally in
StyleTTS 2. Defer to v2.

## Why export the frontend state with the model

The frontend is part of the "model" from the user's perspective — changing
frontend rules between training and inference degrades quality. The
exported engine folder therefore contains:
- The model checkpoint
- The pronunciation dictionary JSON
- The frontend version and feature flags
- A manifest describing vocabulary, tokens, sample rate, etc.

At load time, the engine verifies the frontend matches what the model was
trained with.
