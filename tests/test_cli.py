"""Tests for hindi_tts_builder.cli.main.

Uses Click's CliRunner to invoke sub-commands without needing a real GPU or
trained model. We only test the parts of the CLI that don't require heavy
runtime (training, inference).
"""
from pathlib import Path
import pytest
from click.testing import CliRunner

from hindi_tts_builder.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestHelp:
    def test_root_help(self, runner):
        r = runner.invoke(cli, ["--help"])
        assert r.exit_code == 0
        assert "hindi-tts-builder" in r.output

    @pytest.mark.parametrize("cmd", [
        "new", "add-sources", "prepare", "train", "export",
        "speak", "render-srt", "serve", "doctor",
    ])
    def test_subcommand_help(self, runner, cmd):
        r = runner.invoke(cli, [cmd, "--help"])
        assert r.exit_code == 0


class TestNew:
    def test_creates_project(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = runner.invoke(cli, ["new", "myproj"])
        assert r.exit_code == 0, r.output
        assert (tmp_path / "projects" / "myproj" / "config.yaml").exists()

    def test_duplicate_name_fails(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r1 = runner.invoke(cli, ["new", "dup"])
        assert r1.exit_code == 0
        r2 = runner.invoke(cli, ["new", "dup"])
        assert r2.exit_code != 0


class TestAddSources:
    def test_requires_project(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        urls = tmp_path / "urls.txt"
        urls.write_text("https://youtu.be/test12345\n", encoding="utf-8")
        srts = tmp_path / "srts"
        srts.mkdir()
        (srts / "a.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8")

        r = runner.invoke(cli, ["add-sources", "missing", "--urls", str(urls), "--transcripts", str(srts)])
        assert r.exit_code != 0

    def test_add_sources_to_project(self, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(cli, ["new", "p1"])

        urls = tmp_path / "urls.txt"
        urls.write_text(
            "https://youtu.be/aaa12345\n"
            "https://youtu.be/bbb67890\n",
            encoding="utf-8",
        )
        srts = tmp_path / "srts"
        srts.mkdir()
        for name in ("01.srt", "02.srt"):
            (srts / name).write_text("1\n00:00:01,000 --> 00:00:02,000\nतेस्त\n", encoding="utf-8")

        r = runner.invoke(cli, ["add-sources", "p1", "--urls", str(urls), "--transcripts", str(srts)])
        assert r.exit_code == 0, r.output
        manifest_path = tmp_path / "projects" / "p1" / "sources" / "manifest.json"
        assert manifest_path.exists()
