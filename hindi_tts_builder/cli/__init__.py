"""Command-line interface for hindi-tts-builder."""
from __future__ import annotations

__all__ = ["cli"]


def __getattr__(name: str):
    if name == "cli":
        from hindi_tts_builder.cli.main import cli
        return cli
    raise AttributeError(f"module 'hindi_tts_builder.cli' has no attribute {name!r}")
