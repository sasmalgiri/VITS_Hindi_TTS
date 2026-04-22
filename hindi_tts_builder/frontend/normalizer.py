"""Unicode normalization and cleanup for Hindi text.

Handles:
- NFC normalization so visually identical strings are byte-equal
- Nukta consolidation (क़ vs क + nukta combining mark)
- Stray control chars and invisible formatting chars
- Repeated/redundant whitespace
"""
from __future__ import annotations
import unicodedata
import re


# Combining nukta character
NUKTA = "\u093c"

# Precomposed nukta forms → canonical
_NUKTA_MAP = {
    "क" + NUKTA: "क़",
    "ख" + NUKTA: "ख़",
    "ग" + NUKTA: "ग़",
    "ज" + NUKTA: "ज़",
    "ड" + NUKTA: "ड़",
    "ढ" + NUKTA: "ढ़",
    "फ" + NUKTA: "फ़",
    "य" + NUKTA: "य़",
}

# Zero-width and invisible chars that pollute Hindi transcripts
_INVISIBLES = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff]")

# Collapse any whitespace run (including inside Devanagari) to single space
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Return NFC-normalized, cleaned text safe for downstream frontend."""
    if not text:
        return ""
    # NFC normalization: combines decomposed characters to precomposed form.
    t = unicodedata.normalize("NFC", text)

    # Collapse precomposed nukta forms (belt and suspenders on top of NFC)
    for combined, canonical in _NUKTA_MAP.items():
        t = t.replace(combined, canonical)

    # Strip invisible/formatting chars
    t = _INVISIBLES.sub("", t)

    # Collapse whitespace
    t = _WS.sub(" ", t).strip()
    return t
