"""Tests for hindi_tts_builder.frontend.normalizer."""
import pytest
from hindi_tts_builder.frontend.normalizer import normalize


class TestWhitespace:
    def test_collapse_internal_spaces(self):
        assert normalize("यह   है") == "यह है"

    def test_trim_edges(self):
        assert normalize("  यह है  ") == "यह है"

    def test_newlines_become_spaces(self):
        assert normalize("यह\nहै") == "यह है"

    def test_tabs_become_spaces(self):
        assert normalize("यह\tहै") == "यह है"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_only_whitespace(self):
        assert normalize("   \n\t  ") == ""


class TestInvisibleChars:
    @pytest.mark.parametrize("invisible", [
        "\u200b",  # ZWSP
        "\u200c",  # ZWNJ
        "\u200d",  # ZWJ
        "\ufeff",  # BOM
        "\u2060",  # WJ
    ])
    def test_strips_invisible(self, invisible):
        text = f"यह{invisible}है"
        assert normalize(text) == "यहहै"


class TestNFC:
    def test_decomposed_to_composed(self):
        # क + nukta combining = क़ (precomposed)
        decomposed = "क" + "\u093c"
        result = normalize(decomposed)
        assert result == "क़"

    def test_idempotent(self):
        text = "यह नमस्ते दुनिया है।"
        assert normalize(normalize(text)) == normalize(text)
