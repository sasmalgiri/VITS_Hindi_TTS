"""What characters may reach the trainer, and how to fold the rest.

Mirrors the character set in ``train/trainer.py`` (``_CHARS_FOR_COQUI`` +
``_PUNCT_FOR_COQUI``, lines 73-81) but without importing it, so the data layer
stays free of the Coqui dependency. ``tests/test_text_compat.py`` cross-checks
the two so they cannot drift.

Why this exists: ``_preflight_text_compat`` raises RuntimeError at train time on
any character outside the configured vocab. That check is a good thing — it is
what catches the "model trained on the English alphabet, all Hindi discarded as
<unk>" failure. But it means ASR output must be folded *before* it becomes
training text, because Whisper freely emits typographic quotes, ellipses and
en-dashes that are not in the set.
"""
from __future__ import annotations

DEVANAGARI = "".join(chr(i) for i in range(0x0900, 0x0980))
DIGITS = "0123456789"
#: Must stay in sync with trainer._PUNCT_FOR_COQUI.
PUNCT = " !?,.-:;'\"₹"

TRAINABLE: frozenset[str] = frozenset(DEVANAGARI + DIGITS + PUNCT)

#: Typographic characters Whisper emits, mapped onto the trainable set.
FOLD_MAP: dict[str, str] = {
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "„": '"',
    "«": '"',
    "»": '"',
    "‘": "'",   # left single quote
    "’": "'",   # right single quote
    "‚": "'",
    "–": "-",   # en dash
    "—": "-",   # em dash
    "−": "-",   # minus sign
    "…": ".",   # ellipsis -> single period
    " ": " ",   # nbsp
    "​": "",    # zero-width space
    "‌": "",    # ZWNJ  (safe to drop for TTS text)
    "‍": "",    # ZWJ
    "﻿": "",    # BOM
    "。": ".",   # ideographic full stop
    "，": ",",   # fullwidth comma
    "？": "?",   # fullwidth question mark
    "！": "!",   # fullwidth exclamation
    "′": "'",
    "″": '"',
    "،": ",",   # arabic comma
    "؟": "?",   # arabic question mark
}


def fold_punctuation(text: str) -> str:
    """Replace typographic variants with their trainable ASCII equivalents."""
    return "".join(FOLD_MAP.get(ch, ch) for ch in text)


def untrainable_chars(text: str) -> set[str]:
    """Characters in ``text`` the trainer's pre-flight check would reject."""
    return {ch for ch in text if ch not in TRAINABLE}


def text_is_trainable(text: str) -> bool:
    return not untrainable_chars(text)


def latin_ratio(text: str) -> float:
    """Share of letters that are Latin rather than Devanagari.

    Whisper transcribing Hinglish often writes English words in Latin script.
    Latin letters are NOT in the trainable set, so a high ratio means the text
    needs transliteration before it can be used — not that it can be dropped.
    """
    latin = sum(1 for c in text if "a" <= c.lower() <= "z")
    deva = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    total = latin + deva
    return latin / total if total else 0.0


def sanitize_for_training(text: str, *, drop_unknown: bool = True) -> tuple[str, set[str]]:
    """Fold typography, then optionally drop whatever is still untrainable.

    Returns the cleaned text and the set of characters that had to be dropped
    (empty when nothing was). Never drops Latin letters silently — those are
    reported so the caller can transliterate rather than mangle the word.
    """
    folded = fold_punctuation(text)
    bad = untrainable_chars(folded)
    if not bad or not drop_unknown:
        return " ".join(folded.split()), bad
    latin = {c for c in bad if "a" <= c.lower() <= "z"}
    droppable = bad - latin
    cleaned = "".join(ch for ch in folded if ch not in droppable)
    return " ".join(cleaned.split()), bad
