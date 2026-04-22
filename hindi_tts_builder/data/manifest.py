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
    segmented: bool = False
    qc_passed: bool | None = None


@dataclass
class Source:
    id: str
    url: str
    transcript_path: str
    audio_path: str | None = None
    duration_sec: float | None = None
    status: SourceStatus = field(default_factory=SourceStatus)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Source":
        st = d.get("status", {})
        return cls(
            id=d["id"],
            url=d["url"],
            transcript_path=d["transcript_path"],
            audio_path=d.get("audio_path"),
            duration_sec=d.get("duration_sec"),
            status=SourceStatus(**st),
            error=d.get("error"),
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

    def add(self, url: str, transcript_path: str, index: int | None = None) -> Source:
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

    def find(self, source_id: str) -> Source | None:
        for s in self.sources:
            if s.id == source_id:
                return s
        return None

    def __iter__(self):
        return iter(self.sources)

    def __len__(self) -> int:
        return len(self.sources)
