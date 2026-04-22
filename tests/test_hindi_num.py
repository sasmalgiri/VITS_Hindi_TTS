"""Tests for hindi_tts_builder.frontend.hindi_num."""
import pytest
from hindi_tts_builder.frontend.hindi_num import hindi_cardinal


class TestSmallNumbers:
    """0-99 come from a lookup table; verify key entries and irregular forms."""

    @pytest.mark.parametrize("n,expected", [
        (0, "शून्य"),
        (1, "एक"),
        (5, "पाँच"),
        (10, "दस"),
        (15, "पंद्रह"),
        (19, "उन्नीस"),
        (20, "बीस"),
        (25, "पच्चीस"),
        (50, "पचास"),
        (99, "निन्यानवे"),
    ])
    def test_exact(self, n, expected):
        assert hindi_cardinal(n) == expected

    def test_all_0_99_defined(self):
        for n in range(100):
            result = hindi_cardinal(n)
            assert result and isinstance(result, str)


class TestHundreds:
    @pytest.mark.parametrize("n,expected", [
        (100, "एक सौ"),
        (200, "दो सौ"),
        (500, "पाँच सौ"),
        (999, "नौ सौ निन्यानवे"),
        (101, "एक सौ एक"),
        (150, "एक सौ पचास"),
    ])
    def test_hundreds(self, n, expected):
        assert hindi_cardinal(n) == expected


class TestThousands:
    @pytest.mark.parametrize("n,expected", [
        (1000, "एक हज़ार"),
        (2026, "दो हज़ार छब्बीस"),
        (1500, "एक हज़ार पाँच सौ"),
        (99_999, "निन्यानवे हज़ार नौ सौ निन्यानवे"),
    ])
    def test_thousands(self, n, expected):
        assert hindi_cardinal(n) == expected


class TestLakhsAndCrores:
    """Indian numbering system: 1_00_000 = 1 lakh, 1_00_00_000 = 1 crore."""

    @pytest.mark.parametrize("n,expected", [
        (100_000, "एक लाख"),
        (500_000, "पाँच लाख"),
        (10_000_000, "एक करोड़"),
        (12_34_56_789, "बारह करोड़ चौंतीस लाख छप्पन हज़ार सात सौ नवासी"),
    ])
    def test_indian_numbering(self, n, expected):
        assert hindi_cardinal(n) == expected


class TestNegative:
    def test_negative_one(self):
        assert hindi_cardinal(-1) == "ऋण एक"

    def test_negative_large(self):
        assert hindi_cardinal(-100) == "ऋण एक सौ"


class TestArab:
    def test_one_arab(self):
        # 1,00,00,00,000 = 1 arab (1 billion in Indian system)
        assert hindi_cardinal(1_000_000_000) == "एक अरब"
