"""Tests for hindi_tts_builder.data.manifest."""
from pathlib import Path
import json
import pytest

from hindi_tts_builder.data.manifest import Manifest, Source, SourceStatus, stable_id


class TestStableId:
    @pytest.mark.parametrize("url,expected_prefix", [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "src_dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "src_dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/abc123xyz", "src_abc123xyz"),
        ("https://www.youtube.com/embed/XXXyyy123", "src_XXXyyy123"),
    ])
    def test_extracts_youtube_id(self, url, expected_prefix):
        assert stable_id(url, 0) == expected_prefix

    def test_fallback_to_index(self):
        assert stable_id("not-a-url", 7) == "src_00007"

    def test_deterministic(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert stable_id(url, 0) == stable_id(url, 999)


class TestSource:
    def test_default_status(self):
        s = Source(id="src_1", url="u", transcript_path="t.srt")
        assert not s.status.downloaded
        assert not s.status.aligned
        assert s.duration_sec is None
        assert s.error is None

    def test_round_trip(self):
        s = Source(
            id="src_1", url="u", transcript_path="t.srt",
            audio_path="a.wav", duration_sec=10.5,
            status=SourceStatus(downloaded=True, aligned=True),
        )
        d = s.to_dict()
        s2 = Source.from_dict(d)
        assert s2.id == s.id
        assert s2.audio_path == "a.wav"
        assert s2.duration_sec == 10.5
        assert s2.status.downloaded
        assert s2.status.aligned


class TestManifest:
    def test_empty(self, tmp_path: Path):
        m = Manifest(tmp_path / "manifest.json")
        assert len(m) == 0

    def test_add_and_save(self, tmp_path: Path):
        mpath = tmp_path / "manifest.json"
        m = Manifest(mpath)
        m.add(url="https://youtu.be/abc123xyz", transcript_path="t1.srt")
        m.add(url="https://youtu.be/def456uvw", transcript_path="t2.srt")
        m.save()

        assert mpath.exists()
        raw = json.loads(mpath.read_text(encoding="utf-8"))
        assert len(raw["sources"]) == 2

    def test_reload(self, tmp_path: Path):
        mpath = tmp_path / "manifest.json"
        m1 = Manifest(mpath)
        m1.add(url="https://youtu.be/abc123xyz", transcript_path="t1.srt")
        m1.save()

        m2 = Manifest(mpath)
        assert len(m2) == 1
        assert m2.sources[0].url == "https://youtu.be/abc123xyz"

    def test_find(self, tmp_path: Path):
        m = Manifest(tmp_path / "m.json")
        m.add(url="https://youtu.be/abc123xyz", transcript_path="t.srt")
        sid = m.sources[0].id
        assert m.find(sid) is not None
        assert m.find("nonexistent") is None

    def test_no_duplicate_ids(self, tmp_path: Path):
        m = Manifest(tmp_path / "m.json")
        # Force same URL twice to test disambiguation
        m.add(url="https://youtu.be/abc123xyz", transcript_path="t1.srt", index=0)
        m.add(url="https://youtu.be/abc123xyz", transcript_path="t2.srt", index=1)
        ids = {s.id for s in m.sources}
        assert len(ids) == 2
