"""Tests for hindi_tts_builder.inference.roundtrip."""
import pytest

from hindi_tts_builder.inference.roundtrip import (
    RoundTripValidator, ValidationResult, _cer,
)


class TestCER:
    def test_identical(self):
        assert _cer("नमस्ते", "नमस्ते") == 0.0

    def test_empty_ref_empty_hyp(self):
        assert _cer("", "") == 0.0

    def test_empty_ref_non_empty_hyp(self):
        assert _cer("", "something") == 1.0

    def test_whitespace_ignored(self):
        assert _cer("क ख ग", "कखग") == 0.0

    def test_nfc_normalized(self):
        # Nukta decomposed vs composed
        decomposed = "क" + "\u093c"
        composed = "क़"
        assert _cer(decomposed, composed) == 0.0


class TestValidatorBehaviorWithoutWhisper:
    """Test behavior when whisper is not installed / not loadable."""

    def test_unavailable_passes_through(self):
        # If whisper is not loadable, validate() should return passed=True
        # with reason="whisper-not-available" so inference doesn't break.
        v = RoundTripValidator(cer_threshold=0.05)
        # Forcibly mark load as attempted with no model:
        v._load_attempted = True
        v._model = None

        result = v.validate(expected_text="नमस्ते", audio_path=None)
        # Should not raise even with None audio_path when unavailable
        assert result.passed is True
        assert "not-available" in result.reason

    def test_validation_result_structure(self):
        r = ValidationResult(passed=True, cer=0.02, transcription="नमस्ते")
        assert r.passed is True
        assert r.cer == 0.02

    def test_available_property(self):
        v = RoundTripValidator()
        # Force state without actually loading
        v._load_attempted = True
        v._model = None
        assert v.available is False
