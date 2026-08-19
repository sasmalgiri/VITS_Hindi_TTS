"""Render the full 54-sentence long-form text through BASE ai4bharat
(no fine-tune), so we can run CER evaluation against the same reference
text for an apples-to-apples comparison vs the 2h fine-tune.

Output:
  /root/hindi-tts/projects/h_tts_1/longform_ab_BASE/
    long_form_full.wav   — 300ms-gapped concatenation
    s000.wav .. s053.wav — per-sentence WAVs (used by CER eval)
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

TEXT_FILE = Path("/root/hindi-tts/eval/long_form_3min.txt")
OUT_ROOT = Path("/root/hindi-tts/projects/h_tts_1/longform_ab_BASE")
GAP_MS = 300

DESCRIPTION = (
    "A Hindi speaker speaks clearly in a close-sounding recording with "
    "very clear audio and no background noise."
)


def main():
    sentences = [
        line.strip() for line in TEXT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    print(f"loaded {len(sentences)} sentences")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    from parler_tts import ParlerTTSForConditionalGeneration  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    print("\nloading base ai4bharat/indic-parler-tts (HF cache)...")
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

    d = desc_tok(DESCRIPTION, return_tensors="pt").to(device)

    print(f"\nrendering {len(sentences)} sentences through base...")
    t0 = time.time()
    audios = []
    for i, s in enumerate(sentences):
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
        arr = np.asarray(arr, dtype=np.float32)
        audios.append(arr)
        sf.write(str(OUT_ROOT / f"s{i:03d}.wav"), arr, sr, subtype="PCM_16")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sentences)} done")
    synth_time = time.time() - t0

    # full concatenation
    gap = np.zeros(int(sr * GAP_MS / 1000), dtype=np.float32)
    full_pieces = []
    total_audio = 0.0
    for a in audios:
        full_pieces.append(a)
        full_pieces.append(gap)
        total_audio += len(a) / sr
    if full_pieces:
        full_pieces = full_pieces[:-1]
    full = np.concatenate(full_pieces)
    full_path = OUT_ROOT / "long_form_full.wav"
    sf.write(str(full_path), full, sr, subtype="PCM_16")

    rtf = synth_time / max(total_audio, 1e-6)
    print(f"\nrendered {len(sentences)} sentences in {synth_time:.1f}s ({synth_time/60:.1f} min)")
    print(f"total audio: {total_audio:.1f}s ({total_audio/60:.1f} min)")
    print(f"RTF: {rtf:.2f}")
    print(f"-> {full_path}")


if __name__ == "__main__":
    main()
