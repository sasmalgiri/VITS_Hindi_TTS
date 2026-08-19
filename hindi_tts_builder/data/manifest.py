"""Source manifest: the single source of truth for URL ↔ audio ↔ transcript triples.

The manifest lives at `projects/<n>/sources/manifest.json` and is updated
incrementally. Every downstream stage reads from it and writes its own
per-source status fields. This makes every stage idempotent: if a source was
already downloaded / aligned / segmented, the stage skips it.

Schema (per source):

    {
        "id":              "src_00001",         # deterministic ID
        "url":             "https://youtu.be/...",
        "transcript_path": "sources/transcripts/episode_1.srt",
        "audio_path":      "audio/raw/src_00001.wav",   # set after download
        "duration_sec":    null,                        # set after download
        "status": {
            "downloaded": false,
            "aligned":    false,
            "segmented":  false,
            "qc_passed":  null   # null until filtered
        },
        "error": null                                   # last error if any
    }
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import hashlib
import json
import re


_URL_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{6,})")


def stable_id(url: str, index: int) -> str:
    """Deterministic ID from URL (preferred) or index fallback."""
    m = _URL_ID_RE.search(url)
    if m:
        return f"src_{m.group(1)}"
    # Fallback: positional index, padded
    return f"src_{index:05d}"


@dataclass
class SourceStatus:
    downloaded: bool = False
    aligned: bool = False
    #: ASR produced the transcript (no user-supplied SRT).
    transcribed: bool = False
    segmented: bool = False
    qc_passed: bool | None = None


@dataclass
class Source:
    id: str
    url: str
    #: None for ASR-transcribed sources, which have no user-supplied SRT.
    transcript_path: str | None = None
    audio_path: str | None = None
    duration_sec: float | None = None
    status: SourceStatus = field(default_factory=SourceStatus)
    error: str | None = None
    # --- provenance: how this source's clips came to exist -------------------
    #: "cue" | "sentence" | "aligned_words" | None for pre-provenance corpora.
    segmentation_policy: str | None = None
    #: Short hash over every parameter that affects where audio was cut.
    segmentation_fingerprint: str | None = None
    #: "user_srt" | "whisperx_aligned" | "asr"
    transcript_origin: str | None = None
    #: "full" | "no_whisper" | "skipped" — whether QC actually scored anything.
    qc_mode: str | None = None
    #: Set when a re-run's policy disagrees with the clips already on disk.
    segmentation_state: str | None = None
    #: Excluded from training. Audio and word timings stay on disk so this is
    #: reversible; downstream stages skip the source entirely.
    excluded: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Source":
        # Filter unknown status keys so a manifest written by a newer build stays
        # loadable here — the bare **st splat is the one place that hard-fails.
        st = {k: v for k, v in d.get("status", {}).items()
              if k in SourceStatus.__dataclass_fields__}
        return cls(
            id=d["id"],
            url=d["url"],
            transcript_path=d.get("transcript_path"),
            audio_path=d.get("audio_path"),
            duration_sec=d.get("duration_sec"),
            status=SourceStatus(**st),
            error=d.get("error"),
            segmentation_policy=d.get("segmentation_policy"),
            segmentation_fingerprint=d.get("segmentation_fingerprint"),
            transcript_origin=d.get("transcript_origin"),
            qc_mode=d.get("qc_mode"),
            segmentation_state=d.get("segmentation_state"),
            excluded=bool(d.get("excluded", False)),
        )


class Manifest:
    """Ordered list of sources, persisted as JSON."""

    def __init__(self, path: Path):
        self.path = path
        self.sources: list[Source] = []
        if path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.sources = [Source.from_dict(d) for d in raw.get("sources", [])]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"sources": [s.to_dict() for s in self.sources]}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, url: str, transcript_path: str | None = None, index: int | None = None) -> Source:
        idx = index if index is not None else len(self.sources)
        sid = stable_id(url, idx)
        # Avoid duplicate IDs
        existing = {s.id for s in self.sources}
        if sid in existing:
            # Disambiguate with positional suffix
            sid = f"{sid}_{idx:05d}"
        src = Source(id=sid, url=url, transcript_path=transcript_path)
        self.sources.append(src)
        return src

    def active(self) -> list[Source]:
        """Sources not excluded from training.

        Stages iterate this rather than the raw list so a single `excluded` flag
        removes a source everywhere, without deleting its audio or word timings.
        """
        return [s for s in self.sources if not s.excluded]

    def find(self, source_id: str) -> Source | None:
        for s in self.sources:
            if s.id == source_id:
                return s
        return None

    def __iter__(self):
        return iter(self.sources)

    def __len__(self) -> int:
        return len(self.sources)
