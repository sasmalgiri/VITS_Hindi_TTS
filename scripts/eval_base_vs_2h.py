"""Direct A/B: base AI4Bharat Indic-Parler-TTS vs the 2h fine-tune.

Renders the same 5 Hindi sentences through both models with identical
generation settings, so you can A/B-listen and answer:

  1. Speaker identity — does 2h sound like the target speaker (Divya)?
     If base sounds like a different speaker, the fine-tune did its job.
  2. Prosody — does 2h have the warm-narrator pacing the training data
     had? Or does it sound like generic TTS?
  3. Word completeness — base is the canonical reference; if base also
     drops words on the same sentences, the omissions are baked into the
     architecture, not a 2h-fine-tune defect.

Output: <OUT_ROOT>/sNN_base.wav and sNN_2h.wav side by side per sentence,
plus a single full_compare_base.wav and full_compare_2h.wav for end-to-end.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from hindi_tts_builder.inference.backends.indic_parler import IndicParlerEngine

# Pick 5 representative sentences across the long-form set:
#   short factual, mid dialogue, long narrative, dramatic exclamation, dialogue with names
SENTENCE_INDICES = [0, 5, 9, 14, 23]  # 0-indexed lines from long_form_3min.txt

ENGINE_2H = Path("/root/hindi-tts/projects/h_tts_1/engines/indic_parler/2h_clean")
TEXT_FILE = Path("/root/hindi-tts/eval/long_form_3min.txt")
OUT_ROOT = Path("/root/hindi-tts/projects/h_tts_1/base_vs_2h_compare")
GAP_MS = 300

DEFAULT_DESCRIPTION = (
    "A Hindi speaker speaks clearly in a close-sounding recording with "
    "very clear audio and no background noise."
)


def render_with_base_model(sentences: list[str], description: str):
    """Load base AI4Bharat model and render. Mirrors the IndicParlerEngine
    generation path but without any engine.json/manifest/pron-dict layer."""
    from parler_tts import ParlerTTSForConditionalGeneration  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    print("  loading base ai4bharat/indic-parler-tts (HF cache)...")
    t0 = time.time()
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts",
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    prompt_tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = int(model.config.sampling_rate)
    print(f"  loaded in {time.time() - t0:.1f}s, sr={sr}")

    d = desc_tok(description, return_tensors="pt").to(device)
    outputs: list[np.ndarray] = []
    for s in sentences:
        p = prompt_tok(s, return_tensors="pt").to(device)
        with torch.inference_mode():
            gen = model.generate(
                input_ids=d.input_ids,
                attention_mask=d.attention_mask,
                prompt_input_ids=p.input_ids,
                prompt_attention_mask=p.attention_mask,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=2400,
            )
        arr = gen.detach().to(torch.float32).cpu().numpy()
        if arr.ndim >= 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim >= 2 and arr.shape[0] == 1:
            arr = arr[0]
        outputs.append(np.asarray(arr, dtype=np.float32))
    return outputs, sr


def main():
    all_lines = [
        line.strip() for line in TEXT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    sentences = [all_lines[i] for i in SENTENCE_INDICES]
    print(f"selected {len(sentences)} sentences for A/B:")
    for i, s in enumerate(sentences):
        print(f"  s{i:02d}: {s[:60]}{'...' if len(s) > 60 else ''}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --- Render through base ---
    print("\n=== BASE: ai4bharat/indic-parler-tts (no fine-tune) ===")
    base_outputs, base_sr = render_with_base_model(sentences, DEFAULT_DESCRIPTION)

    # --- Render through 2h fine-tune ---
    print("\n=== 2H FINE-TUNE: 2h_clean engine ===")
    t_load = time.time()
    eng = IndicParlerEngine.load(ENGINE_2H, dtype="auto", compile_model=False)
    fine_sr = int(eng.base_model.config.sampling_rate) if hasattr(eng, "base_model") else base_sr
    print(f"  loaded in {time.time() - t_load:.1f}s")

    fine_outputs = []
    for s in sentences:
        audio, _ = eng.synthesize(s, temperature=0.7, do_sample=True)
        fine_outputs.append(audio)

    # --- Save side-by-side ---
    print(f"\n=== writing pairs to {OUT_ROOT} ===")
    gap = np.zeros(int(base_sr * GAP_MS / 1000), dtype=np.float32)

    base_chunks = []
    fine_chunks = []
    for i, (b, f) in enumerate(zip(base_outputs, fine_outputs)):
        sf.write(str(OUT_ROOT / f"s{i:02d}_base.wav"), b, base_sr, subtype="PCM_16")
        sf.write(str(OUT_ROOT / f"s{i:02d}_2h.wav"),  f, fine_sr, subtype="PCM_16")
        base_chunks.extend([b, gap])
        fine_chunks.extend([f, gap])
        b_dur = len(b) / base_sr
        f_dur = len(f) / fine_sr
        print(f"  s{i:02d}: base={b_dur:.2f}s  2h={f_dur:.2f}s  text='{sentences[i][:50]}...'")

    if base_chunks:
        base_chunks = base_chunks[:-1]
        fine_chunks = fine_chunks[:-1]
        sf.write(str(OUT_ROOT / "full_compare_base.wav"),
                 np.concatenate(base_chunks), base_sr, subtype="PCM_16")
        sf.write(str(OUT_ROOT / "full_compare_2h.wav"),
                 np.concatenate(fine_chunks), fine_sr, subtype="PCM_16")
    print(f"\nALL DONE: {OUT_ROOT}")


if __name__ == "__main__":
    main()
