"""Minimal SRT parser and writer."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class SrtCue:
    index: int
    start_sec: float
    end_sec: float
    text: str

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")


def _parse_ts(s: str) -> float:
    # fullmatch, not match: a malformed 4-digit millisecond field like
    # "00:00:59,1000" would otherwise parse as 59.100 instead of being rejected.
    m = _TS_RE.fullmatch(s.strip())
    if not m:
        raise ValueError(f"Bad SRT timestamp: {s}")
    h, mi, se, ms = map(int, m.groups())
    return h * 3600 + mi * 60 + se + ms / 1000.0


def _fmt_ts(sec: float) -> str:
    """Format seconds as SRT time, carrying rounded milliseconds correctly.

    Rounding the fractional part alone overflows: 59.9996 rounded to 1000 ms was
    emitted as "00:00:59,1000" — a four-digit field that `_parse_ts` then read
    back as 59.100, a silent 0.9s shift. Aligner and ASR output is unrounded, so
    this was reachable on ordinary data, and segment.py re-parses these files to
    decide where to cut audio.
    """
    if sec < 0:
        sec = 0.0
    total_ms = int(round(sec * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(path: Path) -> list[SrtCue]:
    raw = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\r?\n", raw.strip())
    cues: list[SrtCue] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            # Some SRTs omit the index line; synthesize one
            idx = len(cues) + 1
            ts_line = lines[0]
            text_lines = lines[1:]
        else:
            ts_line = lines[1]
            text_lines = lines[2:]
        if "-->" not in ts_line:
            continue
        left, right = ts_line.split("-->")
        try:
            start = _parse_ts(left)
            end = _parse_ts(right)
        except ValueError:
            continue
        text = " ".join(tl.strip() for tl in text_lines).strip()
        if text:
            cues.append(SrtCue(idx, start, end, text))
    return cues


def write_srt(path: Path, cues: list[SrtCue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_lines = []
    for i, c in enumerate(cues, 1):
        out_lines.append(str(i))
        out_lines.append(f"{_fmt_ts(c.start_sec)} --> {_fmt_ts(c.end_sec)}")
        out_lines.append(c.text)
        out_lines.append("")
    path.write_text("\n".join(out_lines), encoding="utf-8")
