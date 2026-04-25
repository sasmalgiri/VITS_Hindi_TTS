"""Tests for hindi_tts_builder.train.tokenizer."""
from pathlib import Path
import pytest

from hindi_tts_builder.train.tokenizer import (
    HindiTokenizer, RESERVED_TOKENS, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SPACE_TOKEN,
)


class TestInitialVocab:
    def test_reserved_tokens_present(self):
        t = HindiTokenizer()
        for tok in RESERVED_TOKENS:
            assert tok in t.token_to_id

    def test_pad_is_zero(self):
        t = HindiTokenizer()
        assert t.pad_id == 0

    def test_prosody_tokens_included(self):
        t = HindiTokenizer()
        assert "<p_short>" in t.token_to_id
        assert "<falling>" in t.token_to_id


class TestSplit:
    def test_prosody_token_is_atomic(self):
        parts = HindiTokenizer._split("क <p_short> ख")
        # Should contain "<p_short>" as a single element, not character-split
        assert "<p_short>" in parts

    def test_space_becomes_sp_token(self):
        parts = HindiTokenizer._split("क ख")
        assert SPACE_TOKEN in parts

    def test_chars_split_individually(self):
        # Tokenizer receives NFC-normalized text from the frontend.
        # Use plain Devanagari chars that aren't composite forms.
        parts = HindiTokenizer._split("कख")
        assert "क" in parts
        assert "ख" in parts

    def test_consecutive_spaces_collapse(self):
        parts = HindiTokenizer._split("क     ख")
        # Only one SPACE_TOKEN between non-space tokens
        space_indices = [i for i, p in enumerate(parts) if p == SPACE_TOKEN]
        # Consecutive spaces collapsed to at most one
        for i, j in zip(space_indices, space_indices[1:]):
            assert j > i + 1


class TestFit:
    def test_devanagari_chars_present_after_fit(self):
        # Tokenizer pre-seeds the full Devanagari Unicode block, so fitting
        # on Hindi text doesn't necessarily grow the vocab — but every char
        # in the training text MUST be representable.
        t = HindiTokenizer()
        t.fit(["नमस्ते दुनिया"])
        for ch in "नमस्तेदुनिया":
            assert ch in t.token_to_id

    def test_adds_truly_new_chars(self):
        # Latin chars are NOT in the pre-seed; fit() must extend vocab.
        t = HindiTokenizer()
        initial_size = t.vocab_size
        t.fit(["hello"])
        assert t.vocab_size > initial_size
        for ch in "helo":
            assert ch in t.token_to_id

    def test_idempotent(self):
        t = HindiTokenizer()
        t.fit(["नमस्ते"])
        size1 = t.vocab_size
        t.fit(["नमस्ते"])
        assert t.vocab_size == size1

    def test_deterministic_order(self):
        # Same input texts in different orders produce same vocab ordering
        t1 = HindiTokenizer()
        t1.fit(["नमस्ते", "दुनिया"])
        t2 = HindiTokenizer()
        t2.fit(["दुनिया", "नमस्ते"])
        assert t1.id_to_token == t2.id_to_token


class TestEncodeDecode:
    def test_round_trip_preserves_content(self):
        t = HindiTokenizer()
        text = "नमस्ते दुनिया <p_short>"
        t.fit([text])
        ids = t.encode(text, add_bos_eos=False)
        decoded = t.decode(ids)
        # Decoded form matches up to whitespace collapsing
        assert "नमस्ते" in decoded
        assert "दुनिया" in decoded
        assert "<p_short>" in decoded

    def test_bos_eos(self):
        t = HindiTokenizer()
        ids = t.encode("क", add_bos_eos=True)
        assert ids[0] == t.bos_id
        assert ids[-1] == t.eos_id

    def test_unknown_char_becomes_unk(self):
        t = HindiTokenizer()
        # English char not in initial vocab
        ids = t.encode("z", add_bos_eos=False)
        # Either z got added via encode's path (it won't — encode doesn't fit)
        # or z becomes UNK
        assert t.unk_id in ids or t.token_to_id.get("z") in ids


class TestPersistence:
    def test_round_trip(self, tmp_path: Path):
        t1 = HindiTokenizer()
        t1.fit(["नमस्ते दुनिया", "कैसे हो"])
        path = tmp_path / "tok.json"
        t1.save(path)

        t2 = HindiTokenizer.load(path)
        assert t2.vocab_size == t1.vocab_size
        assert t2.id_to_token == t1.id_to_token

        # Encoding produces identical IDs
        text = "नमस्ते"
        assert t1.encode(text) == t2.encode(text)
