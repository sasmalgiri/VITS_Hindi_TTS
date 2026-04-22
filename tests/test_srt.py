"""Tests for hindi_tts_builder.utils.srt."""
import pytest
from pathlib import Path
from hindi_tts_builder.utils.srt import SrtCue, parse_srt, write_srt


SAMPLE_SRT = """1
00:00:01,000 --> 00:00:03,500
नमस्ते दुनिया

2
00:00:04,000 --> 00:00:06,250
यह एक परीक्षण है।

3
00:00:07,100 --> 00:00:09,000
कैसे हो?
"""


class TestParse:
    def test_three_cues(self, tmp_path: Path):
        f = tmp_path / "test.srt"
        f.write_text(SAMPLE_SRT, encoding="utf-8")
        cues = parse_srt(f)
        assert len(cues) == 3

    def test_timestamps(self, tmp_path: Path):
        f = tmp_path / "test.srt"
        f.write_text(SAMPLE_SRT, encoding="utf-8")
        cues = parse_srt(f)
        assert cues[0].start_sec == 1.0
        assert cues[0].end_sec == 3.5
        assert cues[2].start_sec == 7.1

    def test_text(self, tmp_path: Path):
        f = tmp_path / "test.srt"
        f.write_text(SAMPLE_SRT, encoding="utf-8")
        cues = parse_srt(f)
        assert cues[0].text == "नमस्ते दुनिया"
        assert cues[1].text == "यह एक परीक्षण है।"

    def test_duration(self, tmp_path: Path):
        f = tmp_path / "test.srt"
        f.write_text(SAMPLE_SRT, encoding="utf-8")
        cues = parse_srt(f)
        assert cues[0].duration == 2.5

    def test_bom_tolerated(self, tmp_path: Path):
        f = tmp_path / "test.srt"
        f.write_text("\ufeff" + SAMPLE_SRT, encoding="utf-8")
        cues = parse_srt(f)
        assert len(cues) == 3

    def test_crlf_tolerated(self, tmp_path: Path):
        f = tmp_path / "test.srt"
        f.write_text(SAMPLE_SRT.replace("\n", "\r\n"), encoding="utf-8")
        cues = parse_srt(f)
        assert len(cues) == 3


class TestWrite:
    def test_round_trip(self, tmp_path: Path):
        cues = [
            SrtCue(1, 0.5, 2.0, "नमस्ते"),
            SrtCue(2, 3.0, 5.5, "यह परीक्षण है।"),
        ]
        out = tmp_path / "out.srt"
        write_srt(out, cues)
        parsed = parse_srt(out)
        assert len(parsed) == 2
        assert parsed[0].text == "नमस्ते"
        assert parsed[1].end_sec == 5.5
