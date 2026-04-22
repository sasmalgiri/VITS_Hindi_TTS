"""Project structure and path management.

Every project lives in a directory with this layout:

    projects/<name>/
        config.yaml           # project settings
        sources/
            urls.txt          # one YouTube URL per line
            transcripts/      # .srt files matching urls.txt order (or by name)
        audio/
            raw/              # downloaded YouTube audio (WAV)
            resampled/        # 24kHz mono
        aligned/              # per-clip audio + transcripts
        training_set/         # train/val/test splits
        checkpoints/          # training checkpoints
        engine/               # exported production engine
        logs/                 # training + pipeline logs
"""
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ProjectPaths:
    root: Path

    @property
    def config_file(self) -> Path: return self.root / "config.yaml"
    @property
    def sources(self) -> Path: return self.root / "sources"
    @property
    def urls_file(self) -> Path: return self.sources / "urls.txt"
    @property
    def transcripts(self) -> Path: return self.sources / "transcripts"
    @property
    def audio_raw(self) -> Path: return self.root / "audio" / "raw"
    @property
    def audio_resampled(self) -> Path: return self.root / "audio" / "resampled"
    @property
    def aligned(self) -> Path: return self.root / "aligned"
    @property
    def training_set(self) -> Path: return self.root / "training_set"
    @property
    def checkpoints(self) -> Path: return self.root / "checkpoints"
    @property
    def engine(self) -> Path: return self.root / "engine"
    @property
    def logs(self) -> Path: return self.root / "logs"

    def ensure_all(self) -> None:
        for p in [
            self.sources, self.transcripts, self.audio_raw, self.audio_resampled,
            self.aligned, self.training_set, self.checkpoints, self.engine, self.logs,
        ]:
            p.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = {
    "name": None,
    "language": "hi",
    "target_sample_rate": 24000,
    "target_loudness_lufs": -23.0,
    "clip_min_seconds": 1.5,
    "clip_max_seconds": 15.0,
    "qc": {
        "min_snr_db": 30.0,
        "max_cer_vs_whisper": 0.05,
        "max_silence_ratio": 0.25,
    },
    "training": {
        "model": "vits",
        "batch_size": 16,
        "grad_accum": 2,
        "max_steps": 500_000,
        "learning_rate": 2e-4,
        "warmup_steps": 4000,
        "checkpoint_every": 10_000,
        "mixed_precision": "bf16",
    },
    "inference": {
        "roundtrip_validation": True,
        "roundtrip_cer_threshold": 0.02,
        "roundtrip_max_retries": 2,
    },
}


def load_config(project_root: Path) -> dict:
    f = project_root / "config.yaml"
    if not f.exists():
        raise FileNotFoundError(f"No config at {f}. Run `hindi-tts-builder new` first.")
    with open(f, encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def save_config(project_root: Path, cfg: dict) -> None:
    f = project_root / "config.yaml"
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp, allow_unicode=True, sort_keys=False)


def create_project(projects_root: Path, name: str) -> ProjectPaths:
    proot = projects_root / name
    paths = ProjectPaths(proot)
    paths.ensure_all()
    cfg = dict(DEFAULT_CONFIG)
    cfg["name"] = name
    save_config(proot, cfg)
    return paths
