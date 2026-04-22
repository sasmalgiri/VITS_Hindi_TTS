"""Tests for project path management and config."""
from pathlib import Path
import pytest

from hindi_tts_builder.utils.project import (
    ProjectPaths,
    DEFAULT_CONFIG,
    create_project,
    load_config,
    save_config,
)


class TestProjectPaths:
    def test_relative_paths(self, tmp_path: Path):
        p = ProjectPaths(tmp_path / "proj")
        assert p.audio_raw == tmp_path / "proj" / "audio" / "raw"
        assert p.training_set == tmp_path / "proj" / "training_set"
        assert p.engine == tmp_path / "proj" / "engine"

    def test_ensure_all_creates_dirs(self, tmp_path: Path):
        p = ProjectPaths(tmp_path / "proj")
        p.ensure_all()
        assert p.sources.is_dir()
        assert p.transcripts.is_dir()
        assert p.audio_raw.is_dir()
        assert p.checkpoints.is_dir()


class TestCreateProject:
    def test_creates_with_config(self, tmp_path: Path):
        paths = create_project(tmp_path, "test_project")
        assert paths.config_file.exists()
        cfg = load_config(paths.root)
        assert cfg["name"] == "test_project"
        assert cfg["language"] == "hi"

    def test_config_round_trip(self, tmp_path: Path):
        paths = create_project(tmp_path, "rt")
        cfg = load_config(paths.root)
        cfg["training"]["max_steps"] = 100_000
        save_config(paths.root, cfg)
        reloaded = load_config(paths.root)
        assert reloaded["training"]["max_steps"] == 100_000


class TestDefaults:
    def test_has_required_keys(self):
        for key in ("language", "target_sample_rate", "clip_min_seconds", "clip_max_seconds", "qc", "training", "inference"):
            assert key in DEFAULT_CONFIG

    def test_qc_defaults(self):
        qc = DEFAULT_CONFIG["qc"]
        assert qc["min_snr_db"] > 0
        assert 0 < qc["max_cer_vs_whisper"] < 1
        assert 0 < qc["max_silence_ratio"] < 1
