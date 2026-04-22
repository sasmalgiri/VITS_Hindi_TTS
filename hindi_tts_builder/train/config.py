"""Training configuration.

Separate from the top-level project config because training has many knobs
that shouldn't clutter the main config file. A project's training config
lives at `projects/<n>/training_config.yaml` and is auto-created with
sensible defaults on first run.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import yaml


@dataclass
class OptimConfig:
    learning_rate_gen: float = 2e-4
    learning_rate_disc: float = 2e-4
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    weight_decay: float = 0.01
    grad_clip_norm: float = 5.0
    lr_decay: float = 0.999875
    warmup_steps: int = 4000


@dataclass
class ModelConfig:
    """VITS architecture knobs sized for 12GB VRAM."""
    n_mel_channels: int = 80
    sample_rate: int = 24000
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    mel_fmin: float = 0.0
    mel_fmax: float = 12000.0

    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1

    # Posterior encoder
    posterior_hidden_channels: int = 192
    posterior_kernel_size: int = 5
    posterior_n_layers: int = 16

    # Flow
    flow_hidden_channels: int = 192
    flow_kernel_size: int = 5
    flow_n_layers: int = 4

    # Generator (vocoder)
    resblock_type: str = "1"
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5), (1, 3, 5), (1, 3, 5)
    )
    upsample_rates: tuple[int, ...] = (8, 8, 2, 2)
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4)


@dataclass
class TrainingConfig:
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    # Training loop
    batch_size: int = 16
    grad_accum_steps: int = 2
    max_steps: int = 500_000
    epochs: int = 10_000                 # effectively uncapped; max_steps is the real limit
    seed: int = 1234
    mixed_precision: str = "bf16"        # "bf16", "fp16", or "none"
    num_workers: int = 4                 # WSL2/Linux default; Windows native should use 0

    # Sequence limits
    max_audio_length_sec: float = 10.0   # skip clips longer than this
    min_audio_length_sec: float = 1.0

    # Checkpointing
    checkpoint_every_steps: int = 10_000
    keep_last_n_checkpoints: int = 5
    sample_every_steps: int = 25_000     # generate samples for auditioning
    n_sample_sentences: int = 8

    # Eval / health
    eval_every_steps: int = 5_000
    val_batch_size: int = 8
    nan_check_every: int = 100
    max_grad_norm_warn: float = 50.0

    # Data loading
    shuffle: bool = True
    drop_last: bool = True

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        # Nested dataclasses
        model_data = data.pop("model", None)
        optim_data = data.pop("optim", None)
        cfg = cls(**data) if data else cls()
        if model_data:
            cfg.model = ModelConfig(**{k: v for k, v in model_data.items() if k in ModelConfig.__dataclass_fields__})
        if optim_data:
            cfg.optim = OptimConfig(**{k: v for k, v in optim_data.items() if k in OptimConfig.__dataclass_fields__})
        return cfg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(asdict(self), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
