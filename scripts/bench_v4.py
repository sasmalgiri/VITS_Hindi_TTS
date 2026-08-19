"""One-off baseline benchmark for VITS v4 (BEST 4900).

Renders eval/hindi_dubbing_100.jsonl through the exported engine on CPU
(so the running v4 GPU job is not disturbed), saves wavs to
projects/h_tts_1/baseline_v4_step4900/samples/<id>.wav, and appends a
row to eval/results.csv.

Critical: bypasses TTSEngine.load() because the engine manifest currently
writes `apply_prosody=True`, but the v1 trainer was run with
`apply_prosody=False`. Using the manifest setting would feed the model
prosody tokens it never saw during training. We use HindiFrontend with
the *training-time* setting instead.
"""
from __future__ import annotations
import csv
import json
import sys
import time
from pathlib import Path

import soundfile as sf

from hindi_tts_builder.frontend.pipeline import HindiFrontend


PROJECT = "h_tts_1"
BENCH_FILE = Path("eval/hindi_dubbing_100.jsonl")
RESULTS_CSV = Path("eval/results.csv")
ENGINE_DIR = Path(f"projects/{PROJECT}/engine")
OUT_DIR = Path(f"projects/{PROJECT}/baseline_v4_step4900/samples")
SUMMARY_FILE = Path(f"projects/{PROJECT}/baseline_v4_step4900/bench_summary.json")


def load_synthesizer():
    """Direct Coqui Synthesizer load. Two real-world quirks:
      - Skips TTSEngine to bypass the apply_prosody mismatch in the manifest.
      - The Coqui-format config.json (model/audio/datasets all in one Coqpit
        format) is NOT in our engine bundle; it lives next to the checkpoint
        in the run dir. Our engine bundle only ships training_config.yaml
        which is OUR project schema and Coqui's load_config rejects it.
        Sister-find the config.json from the same run dir as the BEST file.
    """
    from TTS.utils.synthesizer import Synthesizer  # type: ignore
    model_path = ENGINE_DIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pt missing at {model_path}")

    # Find the Coqui config.json that corresponds to this BEST.
    # Prefer the run dir whose config.json mtime is closest to the BEST mtime.
    project = Path("projects") / PROJECT
    candidates = sorted(
        project.glob("checkpoints/*/config.json"),
        key=lambda p: abs(p.stat().st_mtime - model_path.stat().st_mtime),
    )
    if not candidates:
        raise FileNotFoundError(
            "No Coqui config.json found under checkpoints/. The export bundle "
            "needs to also include this file — see TASKS.md follow-up."
        )
    coqui_cfg = candidates[0]
    print(f"  using Coqui config: {coqui_cfg}")
    return Synthesizer(
        tts_checkpoint=str(model_path),
        tts_config_path=str(coqui_cfg),
        use_cuda=False,
    )


def main() -> int:
    if not BENCH_FILE.exists():
        print(f"FATAL: benchmark file missing: {BENCH_FILE}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in BENCH_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(rows)} benchmark sentences")

    # Match training-time frontend: prosody tokens disabled (see trainer.py
    # comments — Coqui Graphemes splits them char-by-char).
    fe = HindiFrontend(apply_prosody=False, apply_schwa_deletion=True)

    print("Loading synthesizer (CPU)…")
    t0 = time.time()
    synth = load_synthesizer()
    sample_rate = int(synth.tts_config.audio.sample_rate) if hasattr(synth.tts_config, "audio") else 24000
    print(f"  loaded in {time.time() - t0:.1f}s, sample_rate={sample_rate}")

    by_category: dict[str, list[float]] = {}
    total_synth_sec = 0.0
    total_audio_sec = 0.0
    failures: list[dict] = []

    for i, row in enumerate(rows, 1):
        rid = row["id"]
        cat = row["category"]
        text = row["text"]
        out_wav = OUT_DIR / f"{rid}.wav"

        try:
            processed = fe(text)
            t0 = time.time()
            wav = synth.tts(processed)
            dt = time.time() - t0
            audio = list(wav)
            audio_sec = len(audio) / float(sample_rate)
            sf.write(str(out_wav), audio, sample_rate, subtype="PCM_16")
            total_synth_sec += dt
            total_audio_sec += audio_sec
            by_category.setdefault(cat, []).append(dt / max(audio_sec, 1e-6))
            print(f"  [{i:>2}/{len(rows)}] {cat:14s} {rid}  synth={dt:5.2f}s  audio={audio_sec:5.2f}s  rtf={dt/max(audio_sec,1e-6):4.2f}")
        except Exception as e:
            print(f"  [{i:>2}/{len(rows)}] FAIL {rid}: {e}")
            failures.append({"id": rid, "category": cat, "text": text, "error": str(e)})

    n_ok = len(rows) - len(failures)
    rtf = total_synth_sec / max(total_audio_sec, 1e-6)
    summary = {
        "model": f"vits_v4_step4900",
        "backend": "coqui-vits",
        "data_hours": 16.34,
        "checkpoint": "best_model_4900.pth",
        "n_sentences": len(rows),
        "n_ok": n_ok,
        "n_failed": len(failures),
        "total_synth_seconds": round(total_synth_sec, 2),
        "total_audio_seconds": round(total_audio_sec, 2),
        "rtf_cpu": round(rtf, 3),
        "rtf_per_category_cpu": {
            cat: round(sum(vs) / len(vs), 3) for cat, vs in by_category.items()
        },
        "failures": failures,
        "samples_dir": str(OUT_DIR),
    }
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"Wrote {n_ok}/{len(rows)} samples to {OUT_DIR}")
    print(f"  total synth time: {total_synth_sec:.1f}s  for  {total_audio_sec:.1f}s of audio")
    print(f"  RTF (CPU): {rtf:.2f}")
    if failures:
        print(f"  failed: {[f['id'] for f in failures]}")

    # Append to eval/results.csv
    if not RESULTS_CSV.exists():
        RESULTS_CSV.write_text(
            "model,backend,data_hours,steps,mean_cer,rtf,manual_mos,long_form_stable,date,notes\n",
            encoding="utf-8",
        )
    with RESULTS_CSV.open("a", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "vits_v4_step4900",      # model
            "coqui-vits",            # backend
            16.34,                   # data_hours
            4900,                    # steps (BEST checkpoint)
            "",                      # mean_cer (manual; whisper roundtrip not run here)
            round(rtf, 3),           # rtf (CPU; GPU much faster)
            "",                      # manual_mos (you fill after listening)
            "",                      # long_form_stable
            "2026-04-26",            # date
            f"baseline; {n_ok}/{len(rows)} samples; CPU RTF; samples in {OUT_DIR.relative_to('projects/'+PROJECT)}",
        ])
    print(f"Appended baseline row to {RESULTS_CSV}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
