"""Integration tests over the pipeline stage functions.

The rest of the suite tests pure logic — cue merging, sentence splitting,
timeline analysis, gate rules. That logic was never what broke. Every bug that
actually cost time on this project lived in the *plumbing*:

* ``download_audio`` demanding yt-dlp when every file was already on disk
* two concurrent pipelines interleaving ffmpeg writes and corrupting clips
* ``close_gaps`` moving cue ends but not starts, so the aligner pinned words to
  fabricated boundaries

None of those were covered. These tests exercise the stage functions end to end
against a tiny synthetic project, so the orchestration itself is checked rather
than assumed.

Synthetic audio, no network, no GPU, no model downloads. ffmpeg is required for
the segment stage and those tests skip without it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from hindi_tts_builder.data.manifest import Manifest
from hindi_tts_builder.utils.project import ProjectPaths, save_config, DEFAULT_CONFIG

HAS_FFMPEG = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")

SR = 24000
# stable_id() derives this from the URL below; the wav filename must match it
# or download_audio will not recognise the audio as already present.
SRC_ID = "src_testvideo11"
CUE_SECONDS = 4.0
N_CUES = 8
# Two sentences per cue, so cue-level merging cannot reach the interior
# boundaries — the real corpus's shape in miniature.
CUE_TEXT = "यह पहला वाक्य है। और यह दूसरा वाक्य है।"


def _write_speechlike_wav(path: Path, seconds: float) -> None:
    """Band-limited noise bursts separated by silence — enough structure for
    loudnorm, trim_silence and SNR estimation to behave like they would on speech."""
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(0)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    # Voiced-ish carrier plus noise, amplitude-gated into ~0.8s bursts.
    carrier = 0.35 * np.sin(2 * np.pi * 130 * t) + 0.15 * np.sin(2 * np.pi * 260 * t)
    noise = 0.02 * rng.standard_normal(n)
    gate = ((t % 1.0) < 0.8).astype(np.float32)
    x = ((carrier + noise) * gate).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, SR)


def _srt(path: Path, *, n: int, dur: float, gap: float, text: str) -> None:
    def ts(v: float) -> str:
        h, rem = divmod(v, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"

    blocks, t = [], 0.0
    for i in range(1, n + 1):
        blocks.append(f"{i}\n{ts(t)} --> {ts(t + dur)}\n{text}\n")
        t += dur + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(blocks), encoding="utf-8")


@pytest.fixture
def mini_project(tmp_path: Path) -> Path:
    """A complete tiny project: synthetic audio, matching SRT, ready manifest."""
    root = tmp_path / "proj"
    paths = ProjectPaths(root)
    paths.ensure_all()

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["name"] = "proj"
    cfg["target_sample_rate"] = SR
    cfg["clip_min_seconds"] = 1.0
    cfg["clip_max_seconds"] = 15.0
    cfg["segmentation"]["min_seconds"] = 1.0
    cfg["segmentation"]["max_seconds"] = 15.0
    save_config(root, cfg)

    total = N_CUES * (CUE_SECONDS + 1.0) + 2.0
    _write_speechlike_wav(paths.audio_raw / f"{SRC_ID}.wav", total)
    _srt(paths.transcripts / "test.srt", n=N_CUES, dur=CUE_SECONDS, gap=1.0, text=CUE_TEXT)

    m = Manifest(paths.sources / "manifest.json")
    src = m.add(url="https://youtu.be/testvideo11", transcript_path="sources/transcripts/test.srt")
    assert src.id == SRC_ID, f"fixture id drift: {src.id}"
    src.audio_path = f"audio/raw/{SRC_ID}.wav"
    src.status.downloaded = True
    src.status.aligned = True
    m.save()
    return root


def _manifest(root: Path) -> Manifest:
    return Manifest(ProjectPaths(root).sources / "manifest.json")


# ---------------------------------------------------------------- segmentation

@needs_ffmpeg
class TestSegmentStage:
    def test_cue_mode_makes_one_clip_per_cue(self, mini_project: Path):
        from hindi_tts_builder.data.segment import segment_clips

        paths, m = ProjectPaths(mini_project), _manifest(mini_project)
        s = segment_clips(paths, m, sample_rate=SR, min_seconds=1.0, max_seconds=15.0,
                          trust_srt=True, segmentation_mode="cue")
        assert s["clips_created"] == N_CUES
        wavs = sorted((paths.aligned / SRC_ID).glob("*.wav"))
        assert len(wavs) == N_CUES
        # Clip ids follow the SRT's own cue indices, not list positions.
        assert wavs[0].stem == f"{SRC_ID}_c000001"

    def test_every_clip_has_matching_text_file(self, mini_project: Path):
        from hindi_tts_builder.data.segment import segment_clips

        paths, m = ProjectPaths(mini_project), _manifest(mini_project)
        segment_clips(paths, m, sample_rate=SR, min_seconds=1.0, trust_srt=True,
                      segmentation_mode="cue")
        d = paths.aligned / SRC_ID
        for w in d.glob("*.wav"):
            assert (d / f"{w.stem}.txt").exists()
            assert (d / f"{w.stem}.txt").read_text(encoding="utf-8").strip()

    def test_clips_are_readable_audio(self, mini_project: Path):
        """The concurrent-writer incident produced files soundfile could not open."""
        import soundfile as sf

        from hindi_tts_builder.data.segment import segment_clips

        paths, m = ProjectPaths(mini_project), _manifest(mini_project)
        segment_clips(paths, m, sample_rate=SR, min_seconds=1.0, trust_srt=True,
                      segmentation_mode="cue")
        for w in (paths.aligned / SRC_ID).glob("*.wav"):
            info = sf.info(str(w))
            assert info.frames > 0
            assert info.samplerate == SR
            assert info.channels == 1

    def test_provenance_recorded(self, mini_project: Path):
        from hindi_tts_builder.data.segment import segment_clips

        paths, m = ProjectPaths(mini_project), _manifest(mini_project)
        segment_clips(paths, m, sample_rate=SR, min_seconds=1.0, trust_srt=True,
                      segmentation_mode="cue")
        src = _manifest(mini_project).sources[0]
        assert src.segmentation_policy == "cue"
        assert src.segmentation_fingerprint
        assert src.status.segmented

    def test_policy_change_flags_conflict_and_skips(self, mini_project: Path):
        """A corpus must never end up cut two different ways."""
        from hindi_tts_builder.data.segment import segment_clips

        paths, m = ProjectPaths(mini_project), _manifest(mini_project)
        segment_clips(paths, m, sample_rate=SR, min_seconds=1.0, trust_srt=True,
                      segmentation_mode="cue")
        m2 = _manifest(mini_project)
        s = segment_clips(paths, m2, sample_rate=SR, min_seconds=1.0, trust_srt=True,
                          segmentation_mode="sentence", refuse_untrusted_timeline=False)
        assert s["sources_conflicted"] == 1
        assert s["clips_created"] == 0
        assert _manifest(mini_project).sources[0].segmentation_state == "conflict"

    def test_excluded_source_is_skipped(self, mini_project: Path):
        from hindi_tts_builder.data.segment import segment_clips

        paths, m = ProjectPaths(mini_project), _manifest(mini_project)
        m.sources[0].excluded = True
        m.save()
        s = segment_clips(paths, _manifest(mini_project), sample_rate=SR, min_seconds=1.0,
                          trust_srt=True, segmentation_mode="cue")
        assert s["clips_created"] == 0
        assert s["sources_processed"] == 0

    def test_aligned_words_without_timings_fails_that_source(self, mini_project: Path):
        from hindi_tts_builder.data.segment import segment_clips

        paths, m = ProjectPaths(mini_project), _manifest(mini_project)
        s = segment_clips(paths, m, sample_rate=SR, min_seconds=1.0,
                          segmentation_mode="aligned_words")
        assert s["sources_failed"] == 1
        assert s["clips_created"] == 0

    def test_untrusted_timeline_refused_in_sentence_mode(self, mini_project: Path):
        """The fabricated-timeline guard must actually stop a cut."""
        from hindi_tts_builder.data.segment import segment_clips

        paths, m = ProjectPaths(mini_project), _manifest(mini_project)
        s = segment_clips(paths, m, sample_rate=SR, min_seconds=1.0, trust_srt=True,
                          segmentation_mode="sentence", refuse_untrusted_timeline=True)
        assert s["sources_failed"] == 1
        assert s["clips_created"] == 0


# ------------------------------------------------------------------ QC + build

@needs_ffmpeg
class TestQcAndDataset:
    def _segment(self, root: Path):
        from hindi_tts_builder.data.segment import segment_clips

        paths, m = ProjectPaths(root), _manifest(root)
        segment_clips(paths, m, sample_rate=SR, min_seconds=1.0, trust_srt=True,
                      segmentation_mode="cue")
        return paths, _manifest(root)

    def test_passthrough_qc_is_machine_detectable(self, mini_project: Path):
        import csv

        from hindi_tts_builder.data.gate import read_qc_mode
        from hindi_tts_builder.data.pipeline import _passthrough_qc
        from hindi_tts_builder.utils import get_logger

        paths, m = self._segment(mini_project)
        _passthrough_qc(paths, m, get_logger("t", paths.logs / "t.log"))

        assert read_qc_mode(paths) == "skipped"
        rows = list(csv.DictReader((paths.training_set / "qc_report.csv").open(encoding="utf-8")))
        assert rows
        # Metric columns must be EMPTY, never a fake 0.00 that reads as measured.
        assert all(r["snr_db"] == "" for r in rows)
        assert all(r["reason"] == "qc_skipped" for r in rows)
        assert json.loads((paths.training_set / "qc_report_meta.json").read_text())["qc_mode"] == "skipped"

    def test_real_qc_records_measurements(self, mini_project: Path):
        import csv

        from hindi_tts_builder.data.qc import quality_filter

        paths, m = self._segment(mini_project)
        quality_filter(paths, m, min_snr_db=-99.0, max_cer_vs_whisper=1.0,
                       max_silence_ratio=1.0, min_seconds=0.5, max_seconds=30.0,
                       use_whisper=False)
        rows = list(csv.DictReader((paths.training_set / "qc_report.csv").open(encoding="utf-8")))
        assert rows
        assert any(r.get("snr_db") not in ("", None) for r in rows), "SNR must be measured"

    def test_build_training_set_writes_header_and_splits(self, mini_project: Path):
        from hindi_tts_builder.data.dataset import build_training_set
        from hindi_tts_builder.data.pipeline import _passthrough_qc
        from hindi_tts_builder.frontend.pipeline import HindiFrontend
        from hindi_tts_builder.utils import get_logger

        paths, m = self._segment(mini_project)
        _passthrough_qc(paths, m, get_logger("t", paths.logs / "t.log"))
        s = build_training_set(paths, frontend=HindiFrontend(apply_prosody=False))
        assert s["train"] > 0
        first = (paths.training_set / "train.csv").read_text(encoding="utf-8").splitlines()[0]
        assert first.split("|")[0] == "audio_path", "header row is required; auditors skip it"

    def test_gate_blocks_passthrough_qc_corpus(self, mini_project: Path):
        from hindi_tts_builder.data.dataset import build_training_set
        from hindi_tts_builder.data.gate import check_corpus
        from hindi_tts_builder.data.pipeline import _passthrough_qc
        from hindi_tts_builder.frontend.pipeline import HindiFrontend
        from hindi_tts_builder.utils import get_logger

        paths, m = self._segment(mini_project)
        _passthrough_qc(paths, m, get_logger("t", paths.logs / "t.log"))
        build_training_set(paths, frontend=HindiFrontend(apply_prosody=False))
        res = check_corpus(mini_project, min_hours=0.0)
        assert not res.ok
        assert any("QC did not run" in b for b in res.blockers)


# ----------------------------------------------------------------- end-to-end

@needs_ffmpeg
class TestRunPipeline:
    def test_end_to_end_cue_mode(self, mini_project: Path):
        from hindi_tts_builder.data.pipeline import run_pipeline

        summary = run_pipeline(mini_project, skip_qc=True, trust_srt=True,
                               segmentation_mode="cue", transcribe=False)
        assert summary["segment"]["clips_created"] == N_CUES
        assert summary["dataset"]["train"] > 0
        assert "gate" in summary, "prepare must report a gate verdict"

    def test_lock_is_released_after_run(self, mini_project: Path):
        from hindi_tts_builder.data.pipeline import run_pipeline
        from hindi_tts_builder.utils.lockfile import LOCK_NAME

        run_pipeline(mini_project, skip_qc=True, trust_srt=True,
                     segmentation_mode="cue", transcribe=False)
        assert not (mini_project / LOCK_NAME).exists()

    def test_concurrent_run_is_refused(self, mini_project: Path):
        """The failure that corrupted clips: two writers on one project."""
        import os

        from hindi_tts_builder.data.pipeline import run_pipeline
        from hindi_tts_builder.utils.lockfile import LOCK_NAME, ProjectBusy

        live = 1 if os.name != "nt" else os.getpid()
        if live == os.getpid():
            pytest.skip("cannot simulate a foreign live PID on this platform")
        (mini_project / LOCK_NAME).write_text(
            json.dumps({"pid": live, "operation": "prepare", "host": "h", "started_at": "t"}),
            encoding="utf-8",
        )
        with pytest.raises(ProjectBusy):
            run_pipeline(mini_project, skip_qc=True, trust_srt=True, transcribe=False)

    def test_rerun_is_idempotent(self, mini_project: Path):
        from hindi_tts_builder.data.pipeline import run_pipeline

        run_pipeline(mini_project, skip_qc=True, trust_srt=True,
                     segmentation_mode="cue", transcribe=False)
        second = run_pipeline(mini_project, skip_qc=True, trust_srt=True,
                              segmentation_mode="cue", transcribe=False)
        assert second["segment"]["clips_created"] == 0
        assert second["segment"]["clips_skipped_existing"] == N_CUES


class TestDownloadStage:
    def test_no_yt_dlp_needed_when_audio_present(self, mini_project: Path, monkeypatch):
        """A project whose audio is already on disk must not require yt-dlp."""
        from hindi_tts_builder.data import download as dl

        def boom(*a, **k):
            raise AssertionError("yt-dlp must not be resolved when nothing is downloaded")

        monkeypatch.setattr(dl, "_find_yt_dlp", boom)
        monkeypatch.setattr(dl, "_find_ffmpeg", boom)
        s = dl.download_audio(ProjectPaths(mini_project), _manifest(mini_project))
        assert s["skipped"] == 1
        assert s["downloaded"] == 0
