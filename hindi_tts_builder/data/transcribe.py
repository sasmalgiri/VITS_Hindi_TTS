"""Stage 1.5: transcribe audio to timed words when there is no usable SRT.

Two situations need this:

* **No transcript at all** — a bare YouTube URL. There is nothing to align.
* **A transcript whose timings cannot be trusted** — see
  :mod:`~hindi_tts_builder.data.srt_health`. A fabricated timeline is worse than
  no timeline, because it looks usable.

Output is a word-timing JSON per source at ``aligned/<source_id>.words.json``,
plus a human-readable SRT of the sentence units for spot-checking. Word timings
are what make sentence-accurate cutting possible: with them, a sentence boundary
in the middle of a 7-second span is addressable.

The transcript becomes the training target, so its errors are permanent. Two
guards therefore run here rather than later:

* Characters outside the trainer's vocabulary are folded or reported — otherwise
  ``_preflight_text_compat`` raises hours into a run.
* Latin-script output is measured and reported, never silently dropped. Whisper
  writes English words in Latin for Hinglish audio, and those need
  transliteration, not deletion.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from hindi_tts_builder.data.manifest import Manifest
from hindi_tts_builder.data.sentence_split import (
    split_words_into_sentences,
    words_from_whisper_segments,
)
from hindi_tts_builder.utils import get_logger
from hindi_tts_builder.utils.project import ProjectPaths
from hindi_tts_builder.utils.srt import write_srt
from hindi_tts_builder.utils.text_compat import (
    latin_ratio,
    sanitize_for_training,
    untrainable_chars,
)

#: Tried in order when the GPU cannot hold the previous entry.
_FALLBACK_CHAIN = [
    ("large-v3", "float16"),
    ("medium", "float16"),
    ("medium", "int8_float16"),
    ("small", "int8_float16"),
]


def words_json_path(paths: ProjectPaths, source_id: str) -> Path:
    return paths.aligned / f"{source_id}.words.json"


def _load_model(model: str, compute_type: str, log):
    from faster_whisper import WhisperModel  # imported lazily: heavy + GPU

    log.info(f"[asr] loading {model} ({compute_type})")
    return WhisperModel(model, device="cuda", compute_type=compute_type)


def _load_model_with_fallback(preferred: str, compute_type: str, log):
    """Walk the fallback chain until a model fits in VRAM."""
    chain = [(preferred, compute_type)] + [
        c for c in _FALLBACK_CHAIN if c != (preferred, compute_type)
    ]
    last: Exception | None = None
    for m, ct in chain:
        try:
            return _load_model(m, ct, log), m, ct
        except Exception as e:  # OOM, missing CUDA, unavailable weights
            log.warning(f"[asr] {m}/{ct} unavailable ({type(e).__name__}: {e}); trying next")
            last = e
    raise RuntimeError(f"no usable Whisper model; last error: {last}")


def transcribe_sources(
    paths: ProjectPaths,
    manifest: Manifest,
    *,
    language: str = "hi",
    model: str = "large-v3",
    compute_type: str = "float16",
    beam_size: int = 5,
    min_seconds: float = 2.0,
    max_seconds: float = 15.0,
    max_gap_seconds: float = 0.6,
    skip_existing: bool = True,
    only_untrusted: bool = True,
    logger=None,
) -> dict:
    """Transcribe every source lacking usable timed text.

    ``only_untrusted=True`` (the default) transcribes a source only when it has
    no transcript at all, or when its SRT failed the timeline health check.
    Sources with a trustworthy SRT are left alone.
    """
    log = logger or get_logger("data.transcribe", paths.logs / "transcribe.log")
    summary = {
        "sources_transcribed": 0,
        "sources_skipped": 0,
        "sources_failed": 0,
        "words_total": 0,
        "units_total": 0,
        "terminator_ratio": 0.0,
        "latin_ratio": 0.0,
        "untrainable_chars": [],
        "model_used": None,
    }

    todo = []
    for src in manifest.active():
        if not src.status.downloaded:
            continue
        out = words_json_path(paths, src.id)
        if skip_existing and out.exists():
            summary["sources_skipped"] += 1
            continue
        if only_untrusted and src.transcript_path:
            from hindi_tts_builder.data.srt_health import analyze_timeline
            from hindi_tts_builder.utils.srt import parse_srt

            srt = paths.root / src.transcript_path
            if srt.exists():
                try:
                    if analyze_timeline(parse_srt(srt)).trustworthy:
                        log.info(f"[asr] {src.id}: SRT timeline is trustworthy, not transcribing")
                        summary["sources_skipped"] += 1
                        continue
                except Exception as e:
                    log.warning(f"[asr] {src.id}: could not analyze SRT ({e}); will transcribe")
        todo.append(src)

    if not todo:
        log.info("[asr] nothing to transcribe")
        return summary

    whisper, used_model, used_ct = _load_model_with_fallback(model, compute_type, log)
    summary["model_used"] = f"{used_model}/{used_ct}"

    all_bad: set[str] = set()
    term_num = term_den = 0
    latin_acc = latin_n = 0.0

    for src in todo:
        audio = paths.root / (src.audio_path or "")
        if not audio.exists():
            log.warning(f"[asr] {src.id}: audio missing at {audio}")
            summary["sources_failed"] += 1
            continue
        try:
            log.info(f"[asr] {src.id}: transcribing {audio.name}")
            segments, info = whisper.transcribe(
                str(audio),
                language=language,
                beam_size=beam_size,
                word_timestamps=True,
                vad_filter=True,
            )
            words = words_from_whisper_segments(segments)
            if not words:
                raise RuntimeError("transcription produced no timed words")

            # Fold typography before anything downstream sees the text.
            for w in words:
                w.text, bad = sanitize_for_training(w.text)
                all_bad |= bad
            words = [w for w in words if w.text]

            units, stats = split_words_into_sentences(
                words,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                max_gap_seconds=max_gap_seconds,
                audio_duration=float(getattr(info, "duration", 0.0)) or None,
            )

            payload = {
                "source_id": src.id,
                "language": language,
                "model": summary["model_used"],
                "audio_duration": float(getattr(info, "duration", 0.0)),
                "n_words": len(words),
                "words": [asdict(w) for w in words],
                "unit_stats": stats,
            }
            out = words_json_path(paths, src.id)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            # Human-inspectable SRT of the sentence units.
            write_srt(
                paths.aligned / f"{src.id}.srt",
                [u.to_srt_cue(i + 1) for i, u in enumerate(units)],
            )

            joined = " ".join(w.text for w in words)
            lr = latin_ratio(joined)
            latin_acc += lr
            latin_n += 1
            term_num += sum(1 for u in units if u.terminated)
            term_den += len(units)

            src.status.transcribed = True
            src.status.aligned = True  # timed text now exists for this source
            src.transcript_origin = "asr"
            manifest.save()

            summary["sources_transcribed"] += 1
            summary["words_total"] += len(words)
            summary["units_total"] += len(units)
            log.info(
                f"[asr] {src.id}: {len(words)} words -> {len(units)} units, "
                f"terminated={100 * (sum(1 for u in units if u.terminated) / max(len(units), 1)):.1f}%, "
                f"latin={100 * lr:.1f}%"
            )
        except Exception as e:
            log.error(f"[asr fail] {src.id}: {type(e).__name__}: {e}")
            src.error = f"transcribe: {e}"
            manifest.save()
            summary["sources_failed"] += 1

    summary["terminator_ratio"] = term_num / term_den if term_den else 0.0
    summary["latin_ratio"] = latin_acc / latin_n if latin_n else 0.0
    summary["untrainable_chars"] = sorted(all_bad)

    if all_bad:
        log.warning(
            f"[asr] characters outside the trainer vocabulary appeared: {sorted(all_bad)!r}. "
            f"Punctuation was folded; any Latin letters need transliteration, not deletion."
        )
    if summary["latin_ratio"] > 0.02:
        log.warning(
            f"[asr] {100 * summary['latin_ratio']:.1f}% of letters are Latin script. "
            f"Run the frontend transliterator before building the training set, or "
            f"the trainer's pre-flight check will refuse the corpus."
        )
    log.info(
        f"transcribe complete: {summary['sources_transcribed']} transcribed, "
        f"{summary['sources_skipped']} skipped, {summary['sources_failed']} failed; "
        f"sentence-terminated units {100 * summary['terminator_ratio']:.1f}%"
    )
    return summary


def load_words(paths: ProjectPaths, source_id: str):
    """Read back a source's word timings. Returns [] when absent."""
    from hindi_tts_builder.data.sentence_split import Word

    p = words_json_path(paths, source_id)
    if not p.exists():
        return []
    payload = json.loads(p.read_text(encoding="utf-8"))
    return [Word(**w) for w in payload.get("words", [])]
