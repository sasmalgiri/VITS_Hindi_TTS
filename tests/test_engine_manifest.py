"""Tests for hindi_tts_builder.inference.manifest."""
from pathlib import Path
import pytest

from hindi_tts_builder.inference.manifest import (
    EngineManifest, EngineFrontendSpec, CURRENT_ENGINE_VERSION,
)


class TestDefaults:
    def test_defaults(self):
        m = EngineManifest()
        assert m.engine_version == CURRENT_ENGINE_VERSION
        assert m.language == "hi"
        assert m.sample_rate == 24000
        assert m.model_type == "vits"

    def test_frontend_spec(self):
        m = EngineManifest()
        assert m.frontend.apply_schwa_deletion is True
        assert m.frontend.apply_prosody is True


class TestPersistence:
    def test_round_trip(self, tmp_path: Path):
        m = EngineManifest(
            engine_version=CURRENT_ENGINE_VERSION,
            package_version="0.4.0",
            project_name="test",
            sample_rate=22050,
        )
        m.frontend = EngineFrontendSpec(
            prosody_tokens=["<p_short>", "<rising>"],
            apply_schwa_deletion=False,
            apply_prosody=True,
        )
        path = tmp_path / "manifest.json"
        m.save(path)

        m2 = EngineManifest.load(path)
        assert m2.project_name == "test"
        assert m2.sample_rate == 22050
        assert m2.frontend.prosody_tokens == ["<p_short>", "<rising>"]
        assert m2.frontend.apply_schwa_deletion is False


class TestCompatibility:
    def test_current_version_ok(self):
        m = EngineManifest(engine_version=CURRENT_ENGINE_VERSION)
        m.check_compatible()  # should not raise

    def test_version_mismatch_raises(self):
        m = EngineManifest(engine_version=CURRENT_ENGINE_VERSION + 1)
        with pytest.raises(ValueError, match="Engine version"):
            m.check_compatible()

    def test_wrong_model_type_raises(self):
        m = EngineManifest(model_type="xtts")
        with pytest.raises(ValueError, match="model type"):
            m.check_compatible()

    def test_wrong_language_raises(self):
        m = EngineManifest(language="en")
        with pytest.raises(ValueError, match="Hindi-only"):
            m.check_compatible()
