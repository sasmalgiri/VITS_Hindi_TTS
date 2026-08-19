"""Checkpoint management for resumable training.

A checkpoint is a single `.pt` file containing:
    - generator state_dict
    - discriminator state_dict
    - generator optimizer state
    - discriminator optimizer state
    - step counter
    - training config dict
    - tokenizer vocab
    - torch/CUDA RNG states for exact resumption

Files are named `ckpt_step_XXXXXXXX.pt` and live in `checkpoints/`.
`latest.pt` is a symlink (or copy on Windows) pointing at the most recent.

Keep-policy: retain `keep_last_n` checkpoints, plus never delete the very
first checkpoint (step 0 / initial) so retraining from scratch is always
possible.
"""
from __future__ import annotations
from pathlib import Path
import os
import re
import shutil


_CKPT_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")
_COQUI_BEST_RE = re.compile(r"^best_model_(\d+)\.pth$")
_COQUI_CKPT_RE = re.compile(r"^checkpoint_(\d+)\.pth$")


def list_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """Return [(step, path)] sorted ascending by step."""
    out: list[tuple[int, Path]] = []
    if not checkpoint_dir.exists():
        return out
    for p in checkpoint_dir.iterdir():
        if p.is_file() and p.suffix == ".pt":
            m = _CKPT_RE.match(p.name)
            if m:
                out.append((int(m.group(1)), p))
    return sorted(out, key=lambda t: t[0])


def list_coqui_checkpoints(checkpoint_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Find Coqui's checkpoints anywhere under `checkpoint_dir` (it nests one
    timestamped run-dir deep). Returns {"best": [(step, path)...], "ckpt": [...]}
    each sorted ascending by step. Skips `_archive*` subtrees so manually-quarantined
    NaN-poisoned checkpoints aren't picked up.

    Note: this function still sorts by step number. Callers that need to
    pick across-run resume points should use mtime instead — see
    `latest_checkpoint()` for why.
    """
    out: dict[str, list[tuple[int, Path]]] = {"best": [], "ckpt": []}
    if not checkpoint_dir.exists():
        return out
    for p in checkpoint_dir.rglob("*.pth"):
        if any(part.startswith("_archive") for part in p.parts):
            continue
        mb = _COQUI_BEST_RE.match(p.name)
        mc = _COQUI_CKPT_RE.match(p.name)
        if mb:
            out["best"].append((int(mb.group(1)), p))
        elif mc:
            out["ckpt"].append((int(mc.group(1)), p))
    out["best"].sort(key=lambda t: t[0])
    out["ckpt"].sort(key=lambda t: t[0])
    return out


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return a path to resume/export from, or None.

    Resolution order:
      1. Our own ckpt_step_*.pt (legacy from-scratch trainer).
      2. Coqui's best_model_*.pth — picked by **most recent mtime**, NOT by
         highest step number. Without this, a fresh run that resumed from
         best_model_7350.pth would still see 7350 as "latest" forever (since
         no in-run BEST step exceeds 7350 until ~step 8000+) — and exports
         would ship the v3-era pre-resume weights. Coqui writes BESTs
         atomically (temp + rename), so mtime = creation time. Skips
         `_archive*` subtrees.
      3. Coqui's checkpoint_*.pth (most recent mtime) only if no BEST exists.
    """
    ours = list_checkpoints(checkpoint_dir)
    if ours:
        return ours[-1][1]
    coqui = list_coqui_checkpoints(checkpoint_dir)
    if coqui["best"]:
        return max((p for _, p in coqui["best"]), key=lambda p: p.stat().st_mtime)
    if coqui["ckpt"]:
        return max((p for _, p in coqui["ckpt"]), key=lambda p: p.stat().st_mtime)
    return None


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    *,
    generator,
    discriminator,
    opt_g,
    opt_d,
    training_config: dict,
    vocab: list[str],
    keep_last_n: int = 5,
):
    """Save a checkpoint atomically (write tmp then rename)."""
    import torch  # type: ignore

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final = checkpoint_dir / f"ckpt_step_{step:08d}.pt"
    tmp = final.with_suffix(".pt.tmp")

    state = {
        "step": step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict() if discriminator is not None else None,
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict() if opt_d is not None else None,
        "training_config": training_config,
        "vocab": list(vocab),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, tmp)
    tmp.replace(final)

    # Update latest pointer (copy on Windows where symlinks need admin rights)
    latest = checkpoint_dir / "latest.pt"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        os.symlink(final.name, latest)
    except (OSError, NotImplementedError):
        shutil.copy2(final, latest)

    # Prune
    ckpts = list_checkpoints(checkpoint_dir)
    # Always keep step 0 if present, and the last keep_last_n
    preserve = set()
    if ckpts and ckpts[0][0] == 0:
        preserve.add(ckpts[0][1])
    for _, p in ckpts[-keep_last_n:]:
        preserve.add(p)
    for _, p in ckpts:
        if p not in preserve:
            try:
                p.unlink()
            except OSError:
                pass


def load_checkpoint(path: Path, map_location=None) -> dict:
    """Load a checkpoint dict. Caller applies state to model/optimizer."""
    import torch  # type: ignore
    return torch.load(str(path), map_location=map_location, weights_only=False)
