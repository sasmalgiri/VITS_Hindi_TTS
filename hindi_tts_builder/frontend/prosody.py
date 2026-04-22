"""Punctuation-to-prosody token mapping.

The goal is deterministic, consistent punctuation handling. Each punctuation
type maps to an explicit prosody token the model sees during both training and
inference. Because the same tokens appear in both phases, the model learns
reliable acoustic correlates for each.

Tokens are kept plain ASCII so they survive any text pipeline unchanged:

    <p_short>     short pause (comma)
    <p_medium>    medium pause (period, danda)
    <p_long>      long pause (paragraph break, double danda)
    <falling>     falling pitch (declarative sentence end)
    <rising>      rising pitch (question)
    <emphasis>    emphasis (exclamation)
    <trail>       trailing off (ellipsis)

These tokens are never spoken — the model learns to turn them into silence or
pitch contours. They must be preserved verbatim from training to inference.
"""
from __future__ import annotations
import re


PROSODY_TOKENS = {
    "short_pause":  "<p_short>",
    "medium_pause": "<p_medium>",
    "long_pause":   "<p_long>",
    "falling":      "<falling>",
    "rising":       "<rising>",
    "emphasis":     "<emphasis>",
    "trail":        "<trail>",
}


# Mapping: punctuation → (replacement_string)
# Order matters — longer patterns first so "..." isn't eaten by "."
_RULES: list[tuple[str, str]] = [
    ("...", f" {PROSODY_TOKENS['trail']} {PROSODY_TOKENS['long_pause']} "),
    ("…",   f" {PROSODY_TOKENS['trail']} {PROSODY_TOKENS['long_pause']} "),
    ("॥",   f" {PROSODY_TOKENS['falling']} {PROSODY_TOKENS['long_pause']} "),
    ("।",   f" {PROSODY_TOKENS['falling']} {PROSODY_TOKENS['medium_pause']} "),
    ("?",   f" {PROSODY_TOKENS['rising']} {PROSODY_TOKENS['medium_pause']} "),
    ("!",   f" {PROSODY_TOKENS['emphasis']} {PROSODY_TOKENS['medium_pause']} "),
    (";",   f" {PROSODY_TOKENS['medium_pause']} "),
    (":",   f" {PROSODY_TOKENS['short_pause']} "),
    ("—",   f" {PROSODY_TOKENS['short_pause']} "),
    ("–",   f" {PROSODY_TOKENS['short_pause']} "),
    (",",   f" {PROSODY_TOKENS['short_pause']} "),
    (".",   f" {PROSODY_TOKENS['falling']} {PROSODY_TOKENS['medium_pause']} "),
]


_WS = re.compile(r"\s+")


def inject_prosody(text: str) -> str:
    """Replace punctuation in `text` with prosody tokens."""
    t = text
    for punct, token in _RULES:
        t = t.replace(punct, token)
    # Drop any remaining punctuation-like characters the model shouldn't see.
    # IMPORTANT: angle brackets AND underscore are NOT in this list — they
    # form prosody tokens we just injected (e.g. <p_short>), which must
    # survive this step intact.
    t = re.sub(r"[\"'`~@#$%^&*=+\[\]{}|\\/]", " ", t)
    t = _WS.sub(" ", t).strip()
    return t


def all_prosody_tokens() -> list[str]:
    return list(PROSODY_TOKENS.values())
