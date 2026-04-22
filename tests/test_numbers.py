"""Tests for hindi_tts_builder.frontend.numbers."""
import pytest
from hindi_tts_builder.frontend.numbers import expand_numbers


class TestIntegers:
    def test_simple(self):
        out = expand_numbers("5 लोग")
        assert "पाँच" in out

    def test_large(self):
        out = expand_numbers("यह 2026 है")
        assert "दो हज़ार" in out
        assert "छब्बीस" in out

    def test_multiple_in_sentence(self):
        out = expand_numbers("3 और 4")
        assert "तीन" in out
        assert "चार" in out


class TestDecimal:
    def test_pi(self):
        out = expand_numbers("3.14")
        assert "तीन" in out
        assert "दशमलव" in out
        # decimal fraction is spoken digit-by-digit
        assert "एक" in out
        assert "चार" in out


class TestTime:
    def test_hm(self):
        out = expand_numbers("समय 10:30 है")
        assert "दस" in out
        assert "बजकर" in out
        assert "तीस" in out

    def test_on_hour(self):
        out = expand_numbers("10:00 बजे")
        assert "दस बजे" in out


class TestDate:
    def test_slash_date(self):
        out = expand_numbers("15/03/2026")
        assert "पंद्रह" in out
        assert "मार्च" in out
        assert "दो हज़ार छब्बीस" in out

    def test_dash_date(self):
        out = expand_numbers("01-01-2020")
        assert "एक" in out
        assert "जनवरी" in out


class TestCurrency:
    def test_rupee_symbol(self):
        out = expand_numbers("₹500")
        assert "पाँच सौ" in out
        assert "रुपये" in out

    def test_rs_abbrev(self):
        out = expand_numbers("Rs 100")
        assert "एक सौ" in out
        assert "रुपये" in out


class TestPercent:
    def test_integer_pct(self):
        out = expand_numbers("50%")
        assert "पचास" in out
        assert "प्रतिशत" in out


class TestNoNumbers:
    def test_passthrough(self):
        text = "यह एक वाक्य है।"
        assert expand_numbers(text) == text
