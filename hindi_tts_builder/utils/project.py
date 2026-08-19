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
import copy
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
    # How audio is cut into clips. "cue" reproduces the legacy one-clip-per-SRT-cue
    # behaviour exactly. "sentence" merges cues until a sentence terminator.
    # "aligned_words" cuts from word-level timings, the only mode that can reach a
    # sentence boundary sitting *inside* a cue.
    "segmentation": {
        "mode": "cue",
        "min_seconds": 2.0,
        "max_seconds": 15.0,
        "max_gap_seconds": 0.4,
        "max_interior_gap_seconds": 0.6,
        "max_interior_silence_ratio": 0.25,
        "pad_left_ms": 50,
        "pad_right_ms": 100,
        # Reclaim audio sitting in fabricated inter-cue gaps (see data/srt_health.py).
        "close_synthetic_gaps": True,
        # Refuse to cut on a timeline the health check judged fabricated.
        "refuse_untrusted_timeline": True,
    },
    "asr": {
        "model": "large-v3",
        "compute_type": "float16",
        "batch_size": 8,
        "beam_size": 5,
    },
    # Blocking conditions for `hindi-tts-builder gate` / pre-train checks.
    "gate": {
        "min_sentence_terminator_ratio": 0.60,
        "require_real_qc": True,
        # Whisper-CER QC is the only check that can see text/audio misalignment.
        # Set true only for a deliberate, time-boxed exception.
        "allow_no_whisper_qc": False,
        "min_hours": 1.0,
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


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively overlay `override` on a copy of `base`."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(project_root: Path) -> dict:
    """Load a project config, filling gaps from DEFAULT_CONFIG.

    Deep-merged rather than returned raw: a project created before a config key
    existed must still see that key's default, otherwise every new setting is a
    KeyError on every pre-existing project.
    """
    f = project_root / "config.yaml"
    if not f.exists():
        raise FileNotFoundError(f"No config at {f}. Run `hindi-tts-builder new` first.")
    with open(f, encoding="utf-8") as fp:
        user = yaml.safe_load(fp) or {}
    return _deep_merge(DEFAULT_CONFIG, user)


def save_config(project_root: Path, cfg: dict) -> None:
    f = project_root / "config.yaml"
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp, allow_unicode=True, sort_keys=False)


def create_project(projects_root: Path, name: str) -> ProjectPaths:
    proot = projects_root / name
    paths = ProjectPaths(proot)
    paths.ensure_all()
    # deepcopy, not dict(): a shallow copy shares the nested dicts with
    # DEFAULT_CONFIG, so editing one project's qc block would mutate the global.
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["name"] = name
    save_config(proot, cfg)
    return paths
