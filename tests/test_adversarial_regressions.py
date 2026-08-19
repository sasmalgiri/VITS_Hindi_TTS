"""Regressions for defects found by adversarially attacking the scaffolding.

Author-written tests check that code does what the author expected. These check
the things the author did *not* expect, found by six agents trying to make the
scaffolding crash, corrupt data, or lie. Every test here failed before its fix.

The unifying theme is silence. None of these crashed; all of them produced a
confident wrong answer, which is exactly how this project lost two training runs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from hindi_tts_builder.data.gate import check_corpus, read_qc_mode
from hindi_tts_builder.data.sentence_split import Word, split_words_into_sentences
from hindi_tts_builder.utils.project import DEFAULT_CONFIG, ProjectPaths, load_config
from hindi_tts_builder.utils.srt import SrtCue, _fmt_ts, _parse_ts, parse_srt, write_srt


def _corpus(tmp_path: Path, rows, *, qc_rows=None, meta=None, make_audio=True,
            sources=None, config="name: t\n") -> Path:
    root = tmp_path / "c"
    paths = ProjectPaths(root)
    paths.ensure_all()
    (root / "config.yaml").write_text(config, encoding="utf-8")
    with (paths.training_set / "train.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["audio_path", "raw_text", "processed_text", "duration", "source_id"])
        for r in rows:
            w.writerow(r)
    if make_audio:
        for r in rows:
            if len(r) > 0:
                p = root / r[0]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"RIFF")
    if qc_rows is not None:
        with (paths.training_set / "qc_report.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "source_id", "duration", "snr_db", "silence_ratio",
                        "whisper_cer", "passed", "reason"])
            for r in qc_rows:
                w.writerow(r)
    if meta is not None:
        (paths.training_set / "qc_report_meta.json").write_text(
            json.dumps(meta), encoding="utf-8")
    (paths.sources / "manifest.json").write_text(
        json.dumps({"sources": sources if sources is not None else [
            {"id": "s", "url": "u", "status": {"segmented": True},
             "segmentation_policy": "aligned_words"}]}), encoding="utf-8")
    return root


GOOD_QC = [[f"c{i}", "s", "4.000", "31.2", "0.10", "0.0400", 1, "ok"] for i in range(40)]


class TestSrtTimestampCarry:
    """_fmt_ts(59.9996) emitted '00:00:59,1000', which parsed back as 59.100."""

    @pytest.mark.parametrize("sec", [59.9996, 3599.9999, 0.9999, 119.99951])
    def test_round_trip_is_within_a_millisecond(self, sec: float):
        assert _parse_ts(_fmt_ts(sec)) == pytest.approx(sec, abs=0.001)

    @pytest.mark.parametrize("sec", [59.9996, 3599.9999, 0.9999])
    def test_millisecond_field_is_always_three_digits(self, sec: float):
        assert len(_fmt_ts(sec).split(",")[1]) == 3

    def test_carry_propagates_to_minutes_and_hours(self):
        assert _fmt_ts(59.9996) == "00:01:00,000"
        assert _fmt_ts(3599.9999) == "01:00:00,000"

    def test_malformed_timestamp_is_rejected_not_silently_truncated(self):
        with pytest.raises(ValueError):
            _parse_ts("00:00:59,1000")

    def test_write_then_parse_preserves_cue_order_and_duration(self, tmp_path: Path):
        cues = [SrtCue(1, 59.5, 59.9996, "एक।"), SrtCue(2, 60.4, 61.2, "दो।")]
        p = tmp_path / "x.srt"
        write_srt(p, cues)
        back = parse_srt(p)
        assert all(c.duration > 0 for c in back), "a cue must never invert"
        assert back[0].end_sec <= back[1].start_sec


class TestNoSilentDataLoss:
    """Short units that could not attach forward were deleted outright."""

    def test_every_word_survives_a_stream_of_short_sentences(self):
        w, t = [], 0.0
        for _ in range(150):
            for j in range(4):
                w.append(Word(text=("शब्द" if j < 3 else "अंत।"),
                              start=round(t, 3), end=round(t + 0.3, 3)))
                t += 0.35
            t += 0.6                     # wider than max_gap_seconds
        units, stats = split_words_into_sentences(
            w, min_seconds=2.0, max_seconds=15.0, max_gap_seconds=0.4)
        assert sum(len(u.text.split()) for u in units) == len(w)
        assert stats.get("dropped_short", 0) == 0

    def test_a_lone_short_sentence_is_not_deleted(self):
        w = [Word("हाँ।", 0.0, 0.4)]
        units, _ = split_words_into_sentences(w, min_seconds=5.0)
        assert len(units) == 1


class TestGateCannotFailOpen:
    """Every case here previously PASSED a gate it should have blocked."""

    def test_non_finite_duration_cannot_defeat_min_hours(self, tmp_path: Path):
        root = _corpus(tmp_path,
                       [("c0.wav", "ठीक है।", "ठीक है।", "nan", "s")],
                       qc_rows=GOOD_QC, meta={"qc_mode": "full"})
        res = check_corpus(root, min_hours=1000.0)
        assert not res.ok

    def test_infinite_duration_cannot_defeat_min_hours(self, tmp_path: Path):
        root = _corpus(tmp_path,
                       [("c0.wav", "ठीक है।", "ठीक है।", "inf", "s")],
                       qc_rows=GOOD_QC, meta={"qc_mode": "full"})
        assert not check_corpus(root, min_hours=1000.0).ok

    def test_dangling_paths_beyond_the_first_400_are_caught(self, tmp_path: Path):
        rows = [(f"ok{i}.wav", "ठीक है।", "ठीक है।", "10.000", "s") for i in range(400)]
        root = _corpus(tmp_path, rows, qc_rows=GOOD_QC, meta={"qc_mode": "full"})
        # Append 600 rows whose audio was never created.
        with (ProjectPaths(root).training_set / "train.csv").open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="|")
            for i in range(600):
                w.writerow([f"missing{i}.wav", "ठीक है।", "ठीक है।", "10.000", "s"])
        res = check_corpus(root, min_hours=0.5)
        assert not res.ok
        assert any("do not exist on disk" in b for b in res.blockers)

    @pytest.mark.parametrize("bogus", [None, 0, ["skipped"], "SKIPPED", "skiped", ""])
    def test_unrecognised_qc_mode_is_not_trusted(self, tmp_path: Path, bogus):
        root = _corpus(tmp_path,
                       [("c0.wav", "ठीक है।", "ठीक है।", "10.000", "s")] * 500,
                       qc_rows=[[f"c{i}", "s", "4.000", "", "", "", 1, "qc_skipped"] for i in range(40)],
                       meta={"qc_mode": bogus})
        assert not check_corpus(root, min_hours=0.5).ok

    def test_sidecar_cannot_launder_a_passthrough_report(self, tmp_path: Path):
        """meta says 'full'; the report is plainly a passthrough. Evidence wins."""
        root = _corpus(tmp_path,
                       [("c0.wav", "ठीक है।", "ठीक है।", "10.000", "s")] * 500,
                       qc_rows=[[f"c{i}", "s", "4.000", "0.00", "0.000", "", 1, "qc_skipped"]
                                for i in range(200)],
                       meta={"qc_mode": "full"})
        assert read_qc_mode(ProjectPaths(root)) == "skipped"
        assert not check_corpus(root, min_hours=0.5).ok

    def test_one_scored_row_does_not_make_a_report_full(self, tmp_path: Path):
        qc = [[f"c{i}", "s", "4.000", "", "", "", 1, "qc_skipped"] for i in range(199)]
        qc.append(["real", "s", "4.000", "31.2", "0.10", "0.0400", 1, "ok"])
        root = _corpus(tmp_path,
                       [("c0.wav", "ठीक है।", "ठीक है।", "10.000", "s")] * 500,
                       qc_rows=qc, meta=None)
        assert read_qc_mode(ProjectPaths(root)) != "full"
        assert not check_corpus(root, min_hours=0.5).ok

    def test_ragged_rows_cannot_narrow_the_denominator(self, tmp_path: Path):
        """950 mid-phrase fragments hidden as 3-column rows once read as 100%."""
        rows = [("c.wav", "पूरा वाक्य है।", "पूरा वाक्य है।", "10.000", "s")] * 50
        root = _corpus(tmp_path, rows, qc_rows=GOOD_QC, meta={"qc_mode": "full"})
        with (ProjectPaths(root).training_set / "train.csv").open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="|")
            for _ in range(950):
                w.writerow(["c.wav", "अधूरा टुकड़ा", "अधूरा टुकड़ा"])   # only 3 columns
        res = check_corpus(root, min_hours=0.1)
        assert not res.ok
        assert any("malformed row" in b for b in res.blockers)


class TestConfigTypeSafety:
    """A quoted YAML boolean is a truthy string; it inverted a safety flag."""

    @pytest.mark.parametrize("literal", ["'false'", '"false"', "'0'", '"no"'])
    def test_quoted_false_is_not_truthy(self, tmp_path: Path, literal: str):
        root = tmp_path / "p"
        root.mkdir()
        (root / "config.yaml").write_text(
            f"name: t\ngate:\n  allow_no_whisper_qc: {literal}\n", encoding="utf-8")
        assert load_config(root)["gate"]["allow_no_whisper_qc"] is False

    def test_quoted_number_becomes_a_number(self, tmp_path: Path):
        root = tmp_path / "p"
        root.mkdir()
        (root / "config.yaml").write_text(
            "name: t\ngate:\n  min_sentence_terminator_ratio: '0.6'\n", encoding="utf-8")
        assert load_config(root)["gate"]["min_sentence_terminator_ratio"] == pytest.approx(0.6)

    def test_garbage_where_a_number_belongs_is_rejected_loudly(self, tmp_path: Path):
        root = tmp_path / "p"
        root.mkdir()
        (root / "config.yaml").write_text("name: t\nclip_max_seconds: banana\n", encoding="utf-8")
        with pytest.raises(ValueError, match="clip_max_seconds"):
            load_config(root)

    def test_non_mapping_config_is_rejected(self, tmp_path: Path):
        root = tmp_path / "p"
        root.mkdir()
        (root / "config.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(root)

    def test_loading_never_mutates_the_global_default(self, tmp_path: Path):
        """Mutating DEFAULT_CONFIG would corrupt every later project in-process."""
        before = json.dumps(DEFAULT_CONFIG, sort_keys=True, default=str)
        root = tmp_path / "p"
        root.mkdir()
        (root / "config.yaml").write_text(
            "name: t\nqc:\n  min_snr_db: 3.0\nsegmentation:\n  mode: aligned_words\n",
            encoding="utf-8")
        cfg = load_config(root)
        cfg["qc"]["min_snr_db"] = 999.0
        cfg["segmentation"]["mode"] = "mutated"
        assert json.dumps(DEFAULT_CONFIG, sort_keys=True, default=str) == before


class TestDatasetRespectsExclusionsAndFreshness:
    """build_training_set read only the QC report, never the manifest."""

    def _project(self, tmp_path: Path, *, excluded: bool):
        from hindi_tts_builder.data.manifest import Manifest

        root = tmp_path / "d"
        paths = ProjectPaths(root)
        paths.ensure_all()
        (root / "config.yaml").write_text("name: t\n", encoding="utf-8")
        m = Manifest(paths.sources / "manifest.json")
        a = m.add(url="https://youtu.be/aaaaaaaaaaa")
        b = m.add(url="https://youtu.be/bbbbbbbbbbb")
        for s in (a, b):
            s.status.segmented = True
        b.excluded = excluded
        m.save()

        rows = []
        for src in (a.id, b.id):
            d = paths.aligned / src
            d.mkdir(parents=True, exist_ok=True)
            for i in range(6):
                cid = f"{src}_c{i:06d}"
                (d / f"{cid}.wav").write_bytes(b"RIFF")
                (d / f"{cid}.txt").write_text("यह ठीक है।", encoding="utf-8")
                rows.append([cid, src, "4.000", "31.2", "0.10", "0.0400", 1, "ok"])
        with (paths.training_set / "qc_report.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "source_id", "duration", "snr_db", "silence_ratio",
                        "whisper_cer", "passed", "reason"])
            for r in rows:
                w.writerow(r)
        return root, paths, a.id, b.id

    def test_excluded_source_stays_out_of_the_training_set(self, tmp_path: Path):
        from hindi_tts_builder.data.dataset import build_training_set
        from hindi_tts_builder.frontend.pipeline import HindiFrontend

        root, paths, keep, drop = self._project(tmp_path, excluded=True)
        build_training_set(paths, frontend=HindiFrontend(apply_prosody=False))
        text = (paths.training_set / "train.csv").read_text(encoding="utf-8")
        assert keep in text
        assert drop not in text, "an excluded source must not reappear"

    def test_included_sources_are_all_present(self, tmp_path: Path):
        from hindi_tts_builder.data.dataset import build_training_set
        from hindi_tts_builder.frontend.pipeline import HindiFrontend

        root, paths, a, b = self._project(tmp_path, excluded=False)
        build_training_set(paths, frontend=HindiFrontend(apply_prosody=False))
        text = "".join((paths.training_set / f"{s}.csv").read_text(encoding="utf-8")
                       for s in ("train", "val", "test"))
        assert a in text and b in text

    def test_stale_qc_report_is_refused(self, tmp_path: Path):
        """resegment recuts clips under the same ids; the old QC rows then
        describe audio that no longer exists."""
        import os
        import time

        from hindi_tts_builder.data.dataset import build_training_set
        from hindi_tts_builder.frontend.pipeline import HindiFrontend

        root, paths, a, b = self._project(tmp_path, excluded=False)
        # Simulate a recut: touch the clips well after the QC report.
        future = time.time() + 60
        for p in paths.aligned.rglob("*.wav"):
            os.utime(p, (future, future))
        with pytest.raises(RuntimeError, match="older than the clips"):
            build_training_set(paths, frontend=HindiFrontend(apply_prosody=False))
