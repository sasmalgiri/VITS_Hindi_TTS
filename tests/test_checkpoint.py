"""Tests for hindi_tts_builder.train.checkpoint — without torch dependency."""
from pathlib import Path
import pytest

from hindi_tts_builder.train.checkpoint import list_checkpoints, latest_checkpoint


class TestListCheckpoints:
    def test_empty_dir(self, tmp_path: Path):
        assert list_checkpoints(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path: Path):
        assert list_checkpoints(tmp_path / "nope") == []

    def test_sorts_by_step(self, tmp_path: Path):
        (tmp_path / "ckpt_step_00001000.pt").touch()
        (tmp_path / "ckpt_step_00000100.pt").touch()
        (tmp_path / "ckpt_step_00010000.pt").touch()
        result = list_checkpoints(tmp_path)
        steps = [s for s, _ in result]
        assert steps == [100, 1000, 10000]

    def test_ignores_non_matching_files(self, tmp_path: Path):
        (tmp_path / "ckpt_step_00000100.pt").touch()
        (tmp_path / "latest.pt").touch()              # pointer — not matched
        (tmp_path / "best_model.pth").touch()         # different format
        (tmp_path / "junk.txt").touch()
        result = list_checkpoints(tmp_path)
        assert len(result) == 1
        assert result[0][0] == 100


class TestLatest:
    def test_empty(self, tmp_path: Path):
        assert latest_checkpoint(tmp_path) is None

    def test_returns_highest_step(self, tmp_path: Path):
        (tmp_path / "ckpt_step_00000100.pt").touch()
        (tmp_path / "ckpt_step_00010000.pt").touch()
        (tmp_path / "ckpt_step_00001000.pt").touch()
        latest = latest_checkpoint(tmp_path)
        assert latest is not None
        assert latest.name == "ckpt_step_00010000.pt"
