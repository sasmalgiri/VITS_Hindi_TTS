"""pytest configuration: ensure the package is importable from the repo root."""
import sys
from pathlib import Path

# Add repo root to sys.path so `from hindi_tts_builder...` works even without
# `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent))
