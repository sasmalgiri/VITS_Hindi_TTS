"""Tests for hindi_tts_builder.data.gate."""
import csv
import json
from pathlib import Path

import pytest

from hindi_tts_builder.data.gate import check_corpus, read_qc_mode
from hindi_tts_builder.utils.project import ProjectPaths

# Sentence-terminated rows, so terminator checks pass unless a test says otherwise.
GOOD_ROWS = [
    ("clips/s/c1.wav", "पहला वाक्य है।", "पहला वाक्य है।", "4.000", "s"),
    ("clips/s/c2.wav", "दूसरा वाक्य है।", "दूसरा वाक्य है।", "4.000", "s"),
]


def make_project(tmp_path: Path, *, rows=None, qc_mode="full", n_repeat=500,
                 manifest_sources=None, make_audio=True) -> Path:
    """Build a minimal on-disk project the gate can inspect."""
    root = tmp_path / "proj"
    paths = ProjectPaths(root)
    paths.ensure_all()
    (root / "config.yaml").write_text("name: proj\n", encoding="utf-8")

    rows = rows if rows is not None else GOOD_ROWS
    with (paths.training_set / "train.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        for _ in range(n_repeat):
            for r in rows:
                w.writerow(r)

    if make_audio:
        for r in rows:
            p = root / r[0]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"RIFF")

    if qc_mode is not None:
        (paths.training_set / "qc_report_meta.json").write_text(
            json.dumps({"qc_mode": qc_mode}), encoding="utf-8"
        )
        # The sidecar is only a claim; read_qc_mode cross-checks it against the
        # report, so the fixture must produce a report that actually matches.
        with (paths.training_set / "qc_report.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "source_id", "duration", "snr_db", "silence_ratio",
                        "whisper_cer", "passed", "reason", "qc_mode"])
            for i in range(20):
                if qc_mode == "skipped":
                    w.writerow([f"c{i}", "s", "4.000", "", "", "", 1, "qc_skipped", "skipped"])
                elif qc_mode == "no_whisper":
                    w.writerow([f"c{i}", "s", "4.000", "31.20", "0.100", "", 1, "ok", "no_whisper"])
                else:
                    w.writerow([f"c{i}", "s", "4.000", "31.20", "0.100", "0.0400", 1, "ok", "full"])

    sources = manifest_sources if manifest_sources is not None else [
        {"id": "s", "url": "u", "status": {"downloaded": True, "segmented": True},
         "segmentation_policy": "aligned_words", "segmentation_fingerprint": "abc123"}
    ]
    (paths.sources / "manifest.json").write_text(
        json.dumps({"sources": sources}), encoding="utf-8"
    )
    return root


class TestHappyPath:
    def test_clean_corpus_passes(self, tmp_path: Path):
        res = check_corpus(make_project(tmp_path), min_hours=0.5)
        assert res.ok, res.describe()
        assert res.stats["qc_mode"] == "full"
        assert res.stats["sentence_terminated"] == "100.0%"

    def test_reports_hours_and_clips(self, tmp_path: Path):
        res = check_corpus(make_project(tmp_path, n_repeat=100), min_hours=0.1)
        assert res.stats["clips"] == 200
        # stats["hours"] is formatted to 2dp for display, so compare at that precision.
        assert float(res.stats["hours"]) == pytest.approx(200 * 4 / 3600, abs=0.005)


class TestQcHonesty:
    def test_skipped_qc_blocks(self, tmp_path: Path):
        res = check_corpus(make_project(tmp_path, qc_mode="skipped"), min_hours=0.5)
        assert not res.ok
        assert any("QC did not run" in b for b in res.blockers)

    def test_no_whisper_blocks_under_require_real_qc(self, tmp_path: Path):
        """SNR and silence cannot see misalignment; only CER can.

        This previously passed as a mere warning, which let `require_real_qc: true`
        admit a corpus where text/audio agreement was never measured on any clip —
        the exact defect that broke the last two models.
        """
        res = check_corpus(make_project(tmp_path, qc_mode="no_whisper"), min_hours=0.5)
        assert not res.ok
        assert any("WITHOUT Whisper CER" in b for b in res.blockers)

    def test_no_whisper_downgrades_to_warning_when_explicitly_allowed(self, tmp_path: Path):
        res = check_corpus(
            make_project(tmp_path, qc_mode="no_whisper"),
            min_hours=0.5,
            allow_no_whisper_qc=True,
        )
        assert res.ok
        assert any("WITHOUT Whisper CER" in w for w in res.warnings)

    def test_require_real_qc_can_be_disabled(self, tmp_path: Path):
        res = check_corpus(
            make_project(tmp_path, qc_mode="skipped"), min_hours=0.5, require_real_qc=False
        )
        assert res.ok

    def test_legacy_passthrough_detected_without_meta(self, tmp_path: Path):
        """The h_tts_1 signature: every row reason=qc_skipped, no meta sidecar."""
        root = make_project(tmp_path, qc_mode=None)
        paths = ProjectPaths(root)
        with (paths.training_set / "qc_report.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "source_id", "duration", "snr_db", "silence_ratio",
                        "whisper_cer", "passed", "reason"])
            for i in range(5):
                w.writerow([f"c{i}", "s", "4.000", "0.00", "0.000", "", 1, "qc_skipped"])
        assert read_qc_mode(paths) == "skipped"
        assert not check_corpus(root, min_hours=0.5).ok

    def test_missing_qc_report_blocks(self, tmp_path: Path):
        root = make_project(tmp_path, qc_mode=None)
        assert read_qc_mode(ProjectPaths(root)) == "missing"
        assert not check_corpus(root, min_hours=0.5).ok


class TestSentenceAlignment:
    def test_very_low_ratio_blocks(self, tmp_path: Path):
        rows = [("clips/s/c1.wav", "धीरे-धीरे वे एक", "धीरे-धीरे वे एक", "4.000", "s")]
        res = check_corpus(make_project(tmp_path, rows=rows), min_hours=0.5)
        assert not res.ok
        assert any("sentence terminator" in b for b in res.blockers)

    def test_middling_ratio_only_warns(self, tmp_path: Path):
        rows = [
            ("clips/s/c1.wav", "पूरा वाक्य है।", "पूरा वाक्य है।", "4.000", "s"),
            ("clips/s/c2.wav", "अधूरा टुकड़ा", "अधूरा टुकड़ा", "4.000", "s"),
        ]
        res = check_corpus(make_project(tmp_path, rows=rows), min_hours=0.5,
                           min_sentence_terminator_ratio=0.8)
        assert res.ok
        assert any("sentence terminator" in w for w in res.warnings)


class TestProvenance:
    def test_mixed_policies_block(self, tmp_path: Path):
        srcs = [
            {"id": "a", "url": "u", "status": {"segmented": True}, "segmentation_policy": "cue"},
            {"id": "b", "url": "u2", "status": {"segmented": True}, "segmentation_policy": "sentence"},
        ]
        res = check_corpus(make_project(tmp_path, manifest_sources=srcs), min_hours=0.5)
        assert not res.ok
        assert any("mixes segmentation policies" in b for b in res.blockers)

    def test_conflict_flag_blocks(self, tmp_path: Path):
        srcs = [{"id": "a", "url": "u", "status": {"segmented": True},
                 "segmentation_policy": "cue", "segmentation_state": "conflict"}]
        res = check_corpus(make_project(tmp_path, manifest_sources=srcs), min_hours=0.5)
        assert not res.ok
        assert any("policy conflicts" in b for b in res.blockers)

    def test_absent_provenance_only_warns(self, tmp_path: Path):
        srcs = [{"id": "a", "url": "u", "status": {"segmented": True}}]
        res = check_corpus(make_project(tmp_path, manifest_sources=srcs), min_hours=0.5)
        assert res.ok
        assert any("no segmentation provenance" in w for w in res.warnings)


class TestIntegrity:
    def test_untrainable_chars_block(self, tmp_path: Path):
        rows = [("clips/s/c1.wav", "ठीक “है”।", "ठीक “है”।", "4.000", "s")]
        res = check_corpus(make_project(tmp_path, rows=rows), min_hours=0.5)
        assert not res.ok
        assert any("outside the trainer" in b for b in res.blockers)

    def test_missing_audio_blocks(self, tmp_path: Path):
        res = check_corpus(make_project(tmp_path, make_audio=False), min_hours=0.5)
        assert not res.ok
        assert any("do not exist on disk" in b for b in res.blockers)

    def test_missing_csv_blocks(self, tmp_path: Path):
        root = tmp_path / "empty"
        ProjectPaths(root).ensure_all()
        res = check_corpus(root)
        assert not res.ok
        assert any("no training CSV" in b for b in res.blockers)

    def test_too_few_hours_blocks(self, tmp_path: Path):
        res = check_corpus(make_project(tmp_path, n_repeat=2), min_hours=10.0)
        assert not res.ok
        assert any("training audio" in b for b in res.blockers)


class TestHeaderHandling:
    """dataset.py writes a header row; scoring it as data invents false blockers."""

    def test_header_row_is_not_scored(self, tmp_path: Path):
        root = make_project(tmp_path, n_repeat=50)
        train = ProjectPaths(root).training_set / "train.csv"
        body = train.read_text(encoding="utf-8")
        train.write_text(
            "audio_path|raw_text|processed_text|duration|source_id\n" + body,
            encoding="utf-8",
        )
        res = check_corpus(root, min_hours=0.1)
        # Without header skipping, "processed_text" contributes Latin letters and
        # "audio_path" counts as a missing clip.
        assert res.ok, res.describe()
        assert "untrainable_chars" not in res.stats
        assert res.stats["clips"] == 100

    def test_headerless_csv_still_read(self, tmp_path: Path):
        res = check_corpus(make_project(tmp_path, n_repeat=50), min_hours=0.1)
        assert res.stats["clips"] == 100


class TestForce:
    def test_force_demotes_every_blocker(self, tmp_path: Path):
        res = check_corpus(make_project(tmp_path, qc_mode="skipped"), min_hours=0.5, force=True)
        assert res.ok
        assert res.forced
        assert any("forced past" in w for w in res.warnings)
