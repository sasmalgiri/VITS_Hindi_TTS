"""Long-form A/B render: 2h Parler vs 10h Parler.

Reads eval/long_form_3min.txt (one sentence per line), renders each
sentence through both engines using the batched API, then concatenates
each engine's output into ONE continuous WAV with 300ms silence between
sentences (mimicking natural narration pacing).

Result: two ~3-minute WAVs you can A/B-listen end-to-end on real
narrative content (not 5-second benchmark snippets).
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from hindi_tts_builder.inference.backends.indic_parler import IndicParlerEngine

ENGINES = {
    "parler_2h":  Path("/root/hindi-tts/projects/h_tts_1/engines/indic_parler/2h_clean"),
    "parler_10h": Path("/root/hindi-tts/projects/h_tts_1/engines/indic_parler/10h_clean"),
}
TEXT_FILE = Path("/root/hindi-tts/eval/long_form_3min.txt")
OUT_ROOT = Path("/root/hindi-tts/projects/h_tts_1/longform_ab_FIXED")
GAP_MS = 300  # natural narration pause between sentences


def main():
    sentences = [
        line.strip() for line in TEXT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    print(f"loaded {len(sentences)} sentences from {TEXT_FILE.name}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for engine_name, engine_dir in ENGINES.items():
        if not engine_dir.exists():
            print(f"SKIP {engine_name}: engine dir missing {engine_dir}")
            continue
        print(f"\n=== rendering {engine_name} ===")
        t_load = time.time()
        eng = IndicParlerEngine.load(engine_dir, dtype="auto", compile_model=False)
        sr = int(eng.base_model.config.sampling_rate) if hasattr(eng, "base_model") else 44100
        print(f"  loaded in {time.time() - t_load:.1f}s, sr={sr}")

        # Render in batches of 4 (matches the batched-bench pattern)
        t0 = time.time()
        results = eng.synthesize_batch(
            sentences, batch_size=4, do_sample=True, temperature=0.7,
        )
        synth_time = time.time() - t0

        # Concatenate with GAP_MS silence between sentences
        sr_engine = results[0][1] if results else sr
        gap = np.zeros(int(sr_engine * GAP_MS / 1000), dtype=np.float32)
        chunks = []
        total_audio = 0.0
        for audio, _ in results:
            if audio.size == 0:
                continue
            chunks.append(audio)
            chunks.append(gap)
            total_audio += len(audio) / sr_engine
        # Drop trailing gap
        if chunks:
            chunks = chunks[:-1]
        if not chunks:
            print(f"  SKIP: no usable audio")
            continue
        full = np.concatenate(chunks)

        out_dir = OUT_ROOT / engine_name
        out_dir.mkdir(parents=True, exist_ok=True)
        full_path = out_dir / "long_form_full.wav"
        sf.write(str(full_path), full, sr_engine, subtype="PCM_16")

        # Also save per-sentence wavs for spot-checking
        for i, (audio, _) in enumerate(results):
            if audio.size > 0:
                sf.write(str(out_dir / f"s{i:03d}.wav"), audio, sr_engine, subtype="PCM_16")

        wallclock_min = synth_time / 60
        rtf = synth_time / max(total_audio, 1e-6)
        print(f"  rendered {len(results)} sentences in {synth_time:.1f}s ({wallclock_min:.1f} min)")
        print(f"  total audio: {total_audio:.1f}s ({total_audio/60:.1f} min)")
        print(f"  RTF: {rtf:.2f}")
        print(f"  -> {full_path}")


if __name__ == "__main__":
    main()
