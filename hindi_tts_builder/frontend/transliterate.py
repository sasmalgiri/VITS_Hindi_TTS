"""Transliterate stray Latin words to Devanagari.

Most of your SRTs should already be Devanagari. This module only activates for
residual Latin tokens (proper nouns, brand names, terms the translator left in
Roman script). Uses a user-editable pronunciation dictionary first, then falls
back to rule-based phonetic transliteration.
"""
from __future__ import annotations
import re
from pathlib import Path
import json


# Rule-based fallback: very simple consonant/vowel mapping.
# This is intentionally minimal — the primary path is the user dictionary.
# For anything important, put an entry in `pronunciation_dict.json`.
_LATIN_RULES = [
    # digraphs first
    ("sh", "श"), ("ch", "च"), ("th", "थ"), ("ph", "फ"), ("kh", "ख"),
    ("gh", "घ"), ("jh", "झ"), ("dh", "ध"), ("bh", "भ"),
    # vowels
    ("aa", "आ"), ("ee", "ई"), ("oo", "ऊ"),
    ("ai", "ऐ"), ("au", "औ"),
    ("a", "अ"), ("e", "ए"), ("i", "इ"), ("o", "ओ"), ("u", "उ"),
    # consonants
    ("k", "क"), ("g", "ग"), ("j", "ज"), ("t", "त"), ("d", "द"),
    ("n", "न"), ("p", "प"), ("b", "ब"), ("m", "म"), ("y", "य"),
    ("r", "र"), ("l", "ल"), ("v", "व"), ("w", "व"), ("s", "स"),
    ("h", "ह"), ("f", "फ"), ("z", "ज़"), ("x", "क्स"), ("q", "क"),
    ("c", "क"),
]


_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")


def _rule_translit(word: str) -> str:
    w = word.lower()
    out = []
    i = 0
    while i < len(w):
        matched = False
        for src, dst in _LATIN_RULES:
            if w[i : i + len(src)] == src:
                out.append(dst)
                i += len(src)
                matched = True
                break
        if not matched:
            i += 1
    return "".join(out)


class Transliterator:
    """Dictionary-first transliterator. Rule-based fallback."""

    def __init__(self, dict_path: Path | None = None):
        self.dictionary: dict[str, str] = {}
        if dict_path and dict_path.exists():
            self.dictionary = json.loads(dict_path.read_text(encoding="utf-8"))

    def add(self, latin: str, devanagari: str) -> None:
        self.dictionary[latin.lower()] = devanagari

    def save(self, dict_path: Path) -> None:
        dict_path.parent.mkdir(parents=True, exist_ok=True)
        dict_path.write_text(
            json.dumps(self.dictionary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def translit_word(self, word: str) -> str:
        key = word.lower()
        if key in self.dictionary:
            return self.dictionary[key]
        return _rule_translit(word)

    def process(self, text: str) -> str:
        return _LATIN_WORD_RE.sub(lambda m: self.translit_word(m.group(0)), text)
