"""Dynamic batch-size strategy for LoRA training.

Picks `per_device_train_batch_size` and `gradient_accumulation_steps` at
training-time based on:

  1. **Free VRAM** — measured live via `torch.cuda.mem_get_info()`. Any
     other process on the GPU (e.g. an in-progress full-FT training)
     reduces what's available; we adapt instead of OOMing.

  2. **Corpus size (quality dimension)** — the effective batch size we
     *want* depends on data scale:
       <30h:    effective_batch = 16   (small data → more steps, slower
                                        learning, prevents overfitting)
       30-100h: effective_batch = 32   (faster convergence on medium data)
       100h+:   effective_batch = 64   (smoother gradients matter at scale)

  3. **Per-device sweet spot** — even if VRAM allows per_device=32, we cap
     at 8 because voice-cloning literature shows diminishing-or-negative
     quality returns past 8 per-device. Bigger goes into grad-accum.

The output is always sane: never crashes, falls back to (1, 16) on any
detection failure. Logs the chosen strategy so the user knows what ran.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchStrategy:
    per_device_batch: int
    grad_accum: int
    effective_batch: int
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "per_device_train_batch_size": self.per_device_batch,
            "gradient_accumulation_steps": self.grad_accum,
            "_effective_batch_size": self.effective_batch,
            "_dynamic_batch_reasoning": self.reasoning,
        }


def _free_vram_gb() -> float | None:
    """Return free VRAM on the default CUDA device in GB, or None if no
    CUDA / detection fails."""
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return None
        free_bytes, _total = torch.cuda.mem_get_info()
        return free_bytes / (1024 ** 3)
    except Exception:
        return None


def _quality_target_effective_batch(total_hours: float) -> int:
    """Higher data → bigger effective batch helps gradient quality."""
    if total_hours < 30:
        return 16
    if total_hours < 100:
        return 32
    return 64


def _round_to_power_of_two(n: int) -> int:
    """Snap to nearest lower power of 2 in [1, 2, 4, 8]. Empirically these
    are the sweet-spot per-device sizes; uneven values (3, 5, 6) often
    hurt utilisation due to padding."""
    if n >= 8:
        return 8
    if n >= 4:
        return 4
    if n >= 2:
        return 2
    return 1


def compute_strategy(
    *,
    total_hours: float,
    max_audio_seconds: float = 10.0,
    free_vram_gb: float | None = None,
    safety_margin_gb: float = 1.5,
    per_sample_vram_estimate_gb: float = 0.25,
    explicit_per_device: int | None = None,
    explicit_grad_accum: int | None = None,
    is_lora: bool = True,
) -> BatchStrategy:
    """Pick (per_device, grad_accum) given live conditions.

    Explicit overrides (CLI flags) take precedence.

    `per_sample_vram_estimate_gb` defaults to 0.25 GB per sample at 10s
    max audio for Parler 0.9B + LoRA — empirically measured. Increase if
    you see OOM; decrease if you have headroom you want to use.
    """
    # Honor explicit overrides immediately
    if explicit_per_device is not None and explicit_grad_accum is not None:
        eff = explicit_per_device * explicit_grad_accum
        return BatchStrategy(
            per_device_batch=explicit_per_device,
            grad_accum=explicit_grad_accum,
            effective_batch=eff,
            reasoning=f"explicit override (per_device={explicit_per_device}, grad_accum={explicit_grad_accum}, eff={eff})",
        )

    # Quality-driven target effective batch
    target_eff = _quality_target_effective_batch(total_hours)

    # VRAM detection
    if free_vram_gb is None:
        free_vram_gb = _free_vram_gb()
    if free_vram_gb is None:
        # No CUDA or detection failed — return safe default
        return BatchStrategy(
            per_device_batch=1,
            grad_accum=target_eff,
            effective_batch=target_eff,
            reasoning="no CUDA / VRAM detection failed; falling back to per_device=1",
        )

    # For full-FT, optimizer states swallow ~70% of VRAM — much less room
    # for batched activations. For LoRA, optimizer states are tiny so
    # we have far more headroom.
    if not is_lora:
        # Full-FT cap stays conservative: 1 per device on 12 GB
        per_device = 1 if explicit_per_device is None else explicit_per_device
        grad_accum = target_eff // max(1, per_device)
        return BatchStrategy(
            per_device_batch=per_device,
            grad_accum=grad_accum,
            effective_batch=per_device * grad_accum,
            reasoning=f"full-FT path: per_device={per_device} fixed for VRAM safety",
        )

    # Adjust per-sample estimate by max audio length (linear-ish scaling)
    audio_factor = max(1.0, max_audio_seconds / 10.0)
    per_sample_gb = per_sample_vram_estimate_gb * audio_factor

    # VRAM-driven per-device cap
    usable_vram = max(0.0, free_vram_gb - safety_margin_gb)
    vram_max = max(1, int(usable_vram / per_sample_gb))

    # Apply explicit per-device override OR pick per_device with quality cap (8)
    if explicit_per_device is not None:
        per_device = max(1, explicit_per_device)
        per_device_source = f"explicit={explicit_per_device}"
    else:
        # Take the smaller of (vram-allows, target-effective, quality-sweet-spot=8)
        per_device_raw = min(vram_max, target_eff, 8)
        per_device = _round_to_power_of_two(per_device_raw)
        per_device_source = (
            f"vram_allows={vram_max}, target={target_eff}, sweet_spot_cap=8 "
            f"-> rounded to {per_device}"
        )

    # grad_accum to hit target_eff
    if explicit_grad_accum is not None:
        grad_accum = max(1, explicit_grad_accum)
    else:
        grad_accum = max(1, (target_eff + per_device - 1) // per_device)

    eff = per_device * grad_accum
    reasoning = (
        f"hours={total_hours:.1f}h -> target_effective={target_eff}; "
        f"free_vram={free_vram_gb:.1f}GB - safety={safety_margin_gb:.1f}GB / "
        f"~{per_sample_gb:.2f}GB/sample = vram_max={vram_max}/device; "
        f"per_device: {per_device_source}; "
        f"grad_accum={grad_accum}; effective={eff}"
    )
    return BatchStrategy(
        per_device_batch=per_device,
        grad_accum=grad_accum,
        effective_batch=eff,
        reasoning=reasoning,
    )


def total_hours_from_csv(csv_path) -> float:
    """Sum the `duration` column of a pipe-CSV. Returns 0.0 on any error."""
    import csv as _csv
    from pathlib import Path
    p = Path(csv_path)
    if not p.exists():
        return 0.0
    total = 0.0
    try:
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            for row in _csv.DictReader(f, delimiter="|"):
                try:
                    total += float(row.get("duration", 0))
                except (ValueError, TypeError):
                    pass
    except Exception:
        return 0.0
    return total / 3600.0
