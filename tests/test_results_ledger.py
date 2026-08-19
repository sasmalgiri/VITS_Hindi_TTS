"""Tests for hindi_tts_builder.eval.results."""
from pathlib import Path

from hindi_tts_builder.eval.results import COLUMNS, append_result, read_results


class TestAppend:
    def test_creates_with_header(self, tmp_path: Path):
        p = append_result(model="m1", backend="coqui-vits", path=tmp_path / "r.csv")
        lines = p.read_text(encoding="utf-8").splitlines()
        assert lines[0].split(",") == COLUMNS
        assert len(lines) == 2

    def test_appends_without_rewriting(self, tmp_path: Path):
        p = tmp_path / "r.csv"
        append_result(model="m1", backend="a", path=p)
        append_result(model="m2", backend="b", path=p)
        rows = read_results(p)
        assert [r["model"] for r in rows] == ["m1", "m2"]

    def test_values_round_trip(self, tmp_path: Path):
        p = tmp_path / "r.csv"
        append_result(
            model="vits_v4", backend="coqui-vits", data_hours=16.34, steps=50000,
            mean_cer=0.231, rtf=0.273, notes="baseline", on="2026-04-26", path=p,
        )
        r = read_results(p)[0]
        assert r["model"] == "vits_v4"
        assert r["steps"] == "50000"
        assert r["mean_cer"] == "0.231"
        assert r["date"] == "2026-04-26"
        assert r["notes"] == "baseline"

    def test_blank_metrics_allowed(self, tmp_path: Path):
        p = append_result(model="m", backend="b", path=tmp_path / "r.csv")
        assert read_results(p)[0]["manual_mos"] == ""

    def test_date_defaults_to_today(self, tmp_path: Path):
        from datetime import date

        p = append_result(model="m", backend="b", path=tmp_path / "r.csv")
        assert read_results(p)[0]["date"] == date.today().isoformat()


class TestRead:
    def test_missing_file_is_empty(self, tmp_path: Path):
        assert read_results(tmp_path / "nope.csv") == []
