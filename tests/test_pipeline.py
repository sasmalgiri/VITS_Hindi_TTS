"""Integration tests for the end-to-end HindiFrontend pipeline."""
import pytest
from pathlib import Path
from hindi_tts_builder.frontend.pipeline import HindiFrontend


class TestEndToEnd:
    def test_simple_sentence(self):
        fe = HindiFrontend()
        result = fe("नमस्ते।")
        # Contains the original word
        assert "नमस्ते" in result
        # Has prosody token for period
        assert "<falling>" in result
        assert "<p_medium>" in result

    def test_number_expands(self):
        fe = HindiFrontend()
        result = fe("यह 2026 है।")
        assert "दो हज़ार" in result
        assert "छब्बीस" in result
        # Digits should not be present
        assert "2026" not in result

    def test_schwa_applied(self):
        fe = HindiFrontend()
        result = fe("राम")
        # Final schwa deletion inserts halant
        assert "\u094d" in result

    def test_prosody_tokens_intact(self):
        fe = HindiFrontend()
        result = fe("नमस्ते, कैसे हो?")
        # Both tokens survive fully — no munging to "<p short>"
        assert "<p_short>" in result
        assert "<rising>" in result

    def test_callable(self):
        fe = HindiFrontend()
        # Can be called as function
        assert fe("नमस्ते") == fe.process("नमस्ते")


class TestFeatureFlags:
    def test_schwa_off(self):
        fe = HindiFrontend(apply_schwa_deletion=False)
        result = fe("राम")
        assert "\u094d" not in result

    def test_prosody_off(self):
        fe = HindiFrontend(apply_prosody=False)
        result = fe("राम।")
        # Prosody tokens not injected; danda remains
        assert "<falling>" not in result


class TestDictionary:
    def test_custom_pronunciation(self, tmp_path: Path):
        dict_path = tmp_path / "pron.json"
        fe = HindiFrontend(dictionary_path=dict_path)
        fe.add_pronunciation("Solo", "सोलो")
        fe.save_dictionary()

        # Reload — dictionary persisted
        fe2 = HindiFrontend(dictionary_path=dict_path)
        result = fe2("Solo Leveling")
        assert "सोलो" in result


class TestProsodyTokensList:
    def test_available(self):
        fe = HindiFrontend()
        tokens = fe.prosody_tokens()
        assert "<p_short>" in tokens
        assert "<p_medium>" in tokens
        assert "<p_long>" in tokens
        assert "<falling>" in tokens
        assert "<rising>" in tokens
        assert "<emphasis>" in tokens
        assert "<trail>" in tokens


class TestRealisticInputs:
    """Real sentences pulled from typical translated SRT content."""

    @pytest.mark.parametrize("text", [
        "वह युद्ध में हार गया।",
        "कहानी यहाँ से शुरू होती है...",
        "क्या तुम तैयार हो?",
        "रुको! यह खतरनाक है।",
        "वर्ष 2020 में, सब कुछ बदल गया।",
        "कीमत ₹1500 है।",
    ])
    def test_processes_without_error(self, text):
        fe = HindiFrontend()
        result = fe(text)
        assert result
        assert isinstance(result, str)

    def test_output_has_no_arabic_digits(self):
        fe = HindiFrontend()
        result = fe("वर्ष 1947, 15 अगस्त को 3 बजे।")
        assert not any(c.isdigit() for c in result)
