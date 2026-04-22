"""Expand Arabic-digit numbers and common date/time patterns into Devanagari Hindi.

Uses `num2words` for cardinal/ordinal numbers. Adds small handcrafted rules for
dates, times, percentages, currency so the model sees pure Devanagari.

We keep this module deterministic and test-covered: every number pattern has a
well-defined mapping. The TTS model should never see Arabic digits at runtime.
"""
from __future__ import annotations
import re

from hindi_tts_builder.frontend.hindi_num import hindi_cardinal

# For digit-by-digit fallback (decimal fractional parts, long IDs, etc.)
DEVANAGARI_DIGITS = {
    "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
    "5": "पाँच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ",
}

# Month names
_MONTHS_HI = {
    1: "जनवरी", 2: "फ़रवरी", 3: "मार्च", 4: "अप्रैल",
    5: "मई", 6: "जून", 7: "जुलाई", 8: "अगस्त",
    9: "सितंबर", 10: "अक्टूबर", 11: "नवंबर", 12: "दिसंबर",
}


def _num_hi(n: int) -> str:
    """Spell a non-negative integer in Hindi using Indian numbering."""
    return hindi_cardinal(n)


def _expand_decimal(s: str) -> str:
    """'3.14' → 'तीन दशमलव एक चार'."""
    whole, frac = s.split(".", 1)
    whole_hi = _num_hi(int(whole)) if whole else "शून्य"
    frac_hi = " ".join(DEVANAGARI_DIGITS.get(c, c) for c in frac)
    return f"{whole_hi} दशमलव {frac_hi}"


def _expand_time(h: int, m: int) -> str:
    """'10:30' → 'दस बजकर तीस मिनट'."""
    h_hi = _num_hi(h)
    if m == 0:
        return f"{h_hi} बजे"
    m_hi = _num_hi(m)
    return f"{h_hi} बजकर {m_hi} मिनट"


def _expand_date_dmy(d: int, m: int, y: int) -> str:
    d_hi = _num_hi(d)
    mo_hi = _MONTHS_HI.get(m, _num_hi(m))
    y_hi = _num_hi(y)
    return f"{d_hi} {mo_hi} {y_hi}"


# Regexes — ordered by specificity
_TIME_RE   = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_DATE_RE   = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
_PCT_RE    = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")
_CURR_RE   = re.compile(r"(?:₹|Rs\.?\s*)(\d+(?:\.\d+)?)")
_DECIMAL_RE = re.compile(r"\b\d+\.\d+\b")
_INT_RE    = re.compile(r"\b\d+\b")


def expand_numbers(text: str) -> str:
    """Replace all numeric patterns in `text` with spelled-out Devanagari."""

    def time_sub(m):
        h, mi = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mi <= 59:
            return _expand_time(h, mi)
        return m.group(0)

    text = _TIME_RE.sub(time_sub, text)

    def date_sub(m):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000 if y < 50 else 1900
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return _expand_date_dmy(d, mo, y)
        return m.group(0)

    text = _DATE_RE.sub(date_sub, text)

    def pct_sub(m):
        val = m.group(1)
        if "." in val:
            return _expand_decimal(val) + " प्रतिशत"
        return _num_hi(int(val)) + " प्रतिशत"

    text = _PCT_RE.sub(pct_sub, text)

    def curr_sub(m):
        val = m.group(1)
        if "." in val:
            return _expand_decimal(val) + " रुपये"
        return _num_hi(int(val)) + " रुपये"

    text = _CURR_RE.sub(curr_sub, text)

    text = _DECIMAL_RE.sub(lambda m: _expand_decimal(m.group(0)), text)
    text = _INT_RE.sub(lambda m: _num_hi(int(m.group(0))), text)

    return text
