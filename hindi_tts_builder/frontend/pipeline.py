"""Full Hindi text frontend pipeline.

Pipeline order:
    raw text
      → normalize (Unicode NFC, nukta, invisibles, whitespace)
      → expand numbers / dates / times / currency
      → transliterate stray Latin tokens (user dict + fallback)
      → schwa deletion (insert halant where needed)
      → inject prosody tokens for punctuation
    ready for model
"""
from __future__ import annotations
from pathlib import Path

from hindi_tts_builder.frontend.normalizer import normalize
from hindi_tts_builder.frontend.numbers import expand_numbers
from hindi_tts_builder.frontend.transliterate import Transliterator
from hindi_tts_builder.frontend.schwa import delete_schwa
from hindi_tts_builder.frontend.prosody import inject_prosody, all_prosody_tokens


class HindiFrontend:
    """Deterministic text → model-input pipeline.

    IMPORTANT: the same instance (same dictionary, same rules) must be used
    for both training data preparation AND inference. Mismatched frontends
    between training and inference degrade quality significantly.
    """

    def __init__(
        self,
        dictionary_path: Path | None = None,
        apply_schwa_deletion: bool = True,
        apply_prosody: bool = True,
    ):
        self.translit = Transliterator(dictionary_path)
        self.apply_schwa = apply_schwa_deletion
        self.apply_prosody = apply_prosody
        self._dict_path = dictionary_path

    # --- pronunciation dictionary ---
    def add_pronunciation(self, latin_or_devanagari: str, devanagari: str) -> None:
        self.translit.add(latin_or_devanagari, devanagari)

    def save_dictionary(self, path: Path | None = None) -> None:
        path = path or self._dict_path
        if path is None:
            raise ValueError("No dictionary path configured.")
        self.translit.save(path)

    # --- processing ---
    def process(self, text: str) -> str:
        t = normalize(text)
        t = expand_numbers(t)
        t = self.translit.process(t)
        if self.apply_schwa:
            t = delete_schwa(t)
        if self.apply_prosody:
            t = inject_prosody(t)
        return t

    def __call__(self, text: str) -> str:
        return self.process(text)

    @staticmethod
    def prosody_tokens() -> list[str]:
        return all_prosody_tokens()
