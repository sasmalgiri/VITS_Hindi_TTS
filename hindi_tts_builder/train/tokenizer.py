"""Tokenizer that maps frontend-processed text to model token IDs.

The frontend output looks like:

    "नमस्ते <p_short> कैसे हो <rising> <p_medium>"

We tokenize this by:
1. Extracting prosody tokens (`<...>`) as single opaque units
2. Splitting the remaining Devanagari into characters
3. Adding whitespace as a single token

This gives a small, closed vocabulary (~80 chars + ~7 prosody tokens + whitespace/pad/bos/eos).
Small vocab is critical: models learn faster and generalize better than subword
tokenizers when the target is a single writing system.

The tokenizer is saved alongside the model so inference always matches training.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import json
import re

from hindi_tts_builder.frontend.prosody import all_prosody_tokens


# Special tokens — reserved IDs
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPACE_TOKEN = "<sp>"   # represents whitespace as a single token

RESERVED_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SPACE_TOKEN]


_PROSODY_RE = re.compile(r"<[a-z_]+>")


# Pre-seed with the full Devanagari Unicode block (U+0900..U+097F) so any
# rare-but-valid char (vocalic R 'ृ', palatal nasal 'ञ', chandrabindu 'ँ',
# visarga 'ः', etc.) is supported even if it doesn't appear in this run's
# training corpus. Better than discovering a missing char only at inference
# and emitting <unk>.
_DEVANAGARI_BLOCK = [chr(i) for i in range(0x0900, 0x0980)]
_ASCII_DIGITS = list("0123456789")
_BASIC_PUNCT = list(" !?,.-:;।'\"")  # space + Hindi punctuation


class HindiTokenizer:
    """Character-level tokenizer with prosody-token awareness."""

    def __init__(self, vocab: list[str] | None = None):
        """Initialize with a given vocab or build an empty one.

        Order of IDs is: RESERVED_TOKENS, prosody tokens, full Devanagari
        block, ASCII digits, basic punctuation. fit() can extend further.
        """
        if vocab is None:
            prosody = all_prosody_tokens()
            seeded = (
                list(RESERVED_TOKENS)
                + sorted(prosody)
                + _DEVANAGARI_BLOCK
                + _ASCII_DIGITS
                + _BASIC_PUNCT
            )
            # Dedupe while preserving first-occurrence order
            seen = set()
            vocab = [t for t in seeded if not (t in seen or seen.add(t))]
        self.id_to_token = list(vocab)
        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}

    # --- public properties for model construction ---
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int: return self.token_to_id[PAD_TOKEN]
    @property
    def bos_id(self) -> int: return self.token_to_id[BOS_TOKEN]
    @property
    def eos_id(self) -> int: return self.token_to_id[EOS_TOKEN]
    @property
    def unk_id(self) -> int: return self.token_to_id[UNK_TOKEN]
    @property
    def space_id(self) -> int: return self.token_to_id[SPACE_TOKEN]

    # --- vocabulary construction ---
    def fit(self, texts: Iterable[str]) -> "HindiTokenizer":
        """Scan training texts and extend vocab with any new characters seen."""
        existing = set(self.id_to_token)
        new_tokens: set[str] = set()
        for text in texts:
            for tok in self._split(text):
                if tok not in existing and tok not in new_tokens:
                    new_tokens.add(tok)
        # Append new tokens sorted for determinism
        for tok in sorted(new_tokens):
            self.token_to_id[tok] = len(self.id_to_token)
            self.id_to_token.append(tok)
        return self

    # --- tokenization ---
    @staticmethod
    def _split(text: str) -> list[str]:
        """Split a frontend-processed string into opaque units."""
        tokens: list[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "<":
                m = _PROSODY_RE.match(text, i)
                if m:
                    tokens.append(m.group(0))
                    i = m.end()
                    continue
                # not a prosody token — treat as regular char
            if ch.isspace():
                tokens.append(SPACE_TOKEN)
                i += 1
                continue
            tokens.append(ch)
            i += 1
        # Collapse consecutive spaces to one
        collapsed: list[str] = []
        for t in tokens:
            if t == SPACE_TOKEN and collapsed and collapsed[-1] == SPACE_TOKEN:
                continue
            collapsed.append(t)
        return collapsed

    def encode(self, text: str, add_bos_eos: bool = True) -> list[int]:
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in self._split(text)]
        if add_bos_eos:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Inverse of encode, for diagnostics. Prosody tokens and <sp> rendered
        as-is so debug output is readable."""
        toks: list[str] = []
        for i in ids:
            if i < 0 or i >= len(self.id_to_token):
                toks.append(UNK_TOKEN)
                continue
            t = self.id_to_token[i]
            if t in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
                continue
            if t == SPACE_TOKEN:
                toks.append(" ")
            else:
                toks.append(t)
        return "".join(toks)

    # --- persistence ---
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "reserved_tokens": RESERVED_TOKENS,
            "vocab": self.id_to_token,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "HindiTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(vocab=payload["vocab"])
