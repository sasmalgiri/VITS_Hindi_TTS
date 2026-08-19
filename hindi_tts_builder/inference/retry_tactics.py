"""Retry tactics for inference self-healing.

A model that produces noise on attempt 1 sometimes produces clean audio
when given the same text after a small mutation. The order of tactics
matters: cheap, conservative mutations first, expensive or destructive
ones last.

Tactics:
    1. seed_only             — re-roll with a different RNG seed (no text change)
    2. extra_normalize       — strip stray non-Devanagari/Latin glyphs, collapse
                                punctuation, expand any leftover digits
    3. apply_dict_aggressive — force every dict entry, even multi-word keys,
                                in case the default dict pass missed casing
    4. shorten_long          — split into sentences and emit only the first
                                clause (last-resort: avoids long-form drift)

After all tactics fail, the caller should mark the result `manual_review=True`
and persist the input + best-attempt audio for human inspection.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import re


@dataclass
class RetryTactic:
    name: str
    transform: Callable[[str], str]
    description: str


def _seed_only(text: str) -> str:
    """No-op transform — caller bumps the seed instead."""
    return text


_PUNCT_COLLAPSE_RE = re.compile(r"([।?!,.:;])\1+")
_NON_DEVANAGARI_LATIN_RE = re.compile(
    r"[^ऀ-ॿ -~\s]"  # keep Devanagari + ASCII printable + whitespace
)


def _extra_normalize(text: str) -> str:
    text = _NON_DEVANAGARI_LATIN_RE.sub("", text)
    text = _PUNCT_COLLAPSE_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _apply_dict_aggressive(text: str, *, dictionary: dict[str, str] | None = None) -> str:
    """Force every dict entry — case-insensitive on the Latin side, exact on
    the Devanagari side. Caller injects the dict via functools.partial.
    """
    if not dictionary:
        return text
    # Sort keys long-first so multi-word entries beat single-word substrings.
    for src in sorted(dictionary.keys(), key=len, reverse=True):
        dst = dictionary[src]
        # Case-insensitive replace for Latin source keys
        if any("a" <= c.lower() <= "z" for c in src):
            text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)
        else:
            text = text.replace(src, dst)
    return text


_SENT_SPLIT_RE = re.compile(r"(?<=[।?!])\s+")


def _shorten_long(text: str) -> str:
    """Last-resort: keep only the first sentence. Long inputs are the most
    common cause of duration drift / word skipping.
    """
    parts = _SENT_SPLIT_RE.split(text)
    if not parts:
        return text
    first = parts[0].strip()
    return first if first else text


def default_tactics(dictionary: dict[str, str] | None = None) -> list[RetryTactic]:
    """Build the canonical retry chain. Cheapest first."""
    from functools import partial

    return [
        RetryTactic("seed_only", _seed_only,
                    "Re-roll with a different RNG seed."),
        RetryTactic("extra_normalize", _extra_normalize,
                    "Strip stray glyphs and collapse repeated punctuation."),
        RetryTactic("apply_dict_aggressive",
                    partial(_apply_dict_aggressive, dictionary=dictionary),
                    "Force every pronunciation-dict entry, case-insensitive."),
        RetryTactic("shorten_long", _shorten_long,
                    "Last-resort: keep only the first sentence."),
    ]
