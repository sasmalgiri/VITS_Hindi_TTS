"""Tests for hindi_tts_builder.train.config."""
from pathlib import Path
import pytest

from hindi_tts_builder.train.config import TrainingConfig, ModelConfig, OptimConfig


class TestDefaults:
    def test_has_reasonable_defaults(self):
        cfg = TrainingConfig()
        assert cfg.batch_size > 0
        assert cfg.max_steps > 0
        assert cfg.seed > 0
        assert cfg.model.sample_rate == 24000
        assert 0 < cfg.optim.learning_rate_gen < 1

    def test_model_config_defaults(self):
        mc = ModelConfig()
        assert mc.n_mel_channels == 80
        assert mc.hop_length == 256


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path):
        cfg = TrainingConfig()
        cfg.batch_size = 32
        cfg.model.sample_rate = 22050
        cfg.optim.learning_rate_gen = 1e-4

        path = tmp_path / "train.yaml"
        cfg.save(path)
        assert path.exists()

        loaded = TrainingConfig.load(path)
        assert loaded.batch_size == 32
        assert loaded.model.sample_rate == 22050
        assert loaded.optim.learning_rate_gen == 1e-4

    def test_load_missing_returns_defaults(self, tmp_path: Path):
        # Loading a non-existent path returns default config, not error
        cfg = TrainingConfig.load(tmp_path / "nope.yaml")
        assert cfg.batch_size == TrainingConfig().batch_size

    def test_partial_yaml_fills_defaults(self, tmp_path: Path):
        path = tmp_path / "partial.yaml"
        path.write_text("batch_size: 8\n", encoding="utf-8")
        cfg = TrainingConfig.load(path)
        assert cfg.batch_size == 8
        # Other fields keep defaults
        assert cfg.seed == TrainingConfig().seed
