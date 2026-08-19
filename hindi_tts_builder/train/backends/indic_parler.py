"""AI4Bharat Indic-Parler-TTS fine-tune backend.

Production wrapper around the official Parler-TTS training script. This
module owns:
  - dataset export (delegates to exporters/indic_parler.py)
  - Parler repo bootstrap (one-time `git clone` into .external/parler-tts)
  - per-profile training-config JSON generation tuned for RTX 3060 12 GB
  - `accelerate launch` of the upstream training script
  - checkpoint discovery
  - engine-bundle export (model + tokenizer + pronunciation_dict)

It does NOT reimplement Parler's training loop — it drives it. That keeps
us aligned with upstream fixes and avoids forking a moving target.

Usage from CLI: `hindi-tts-builder train <project> --backend indic-parler`
(plumbed through the existing CoquiVitsBackend-style adapter). Optional
profile + smoke flags accepted via env vars when called through the
generic --backend route:
    HINDI_TTS_PROFILE=10h_clean
    HINDI_TTS_SMOKE=1
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hindi_tts_builder.train.backends.base import TrainerBackend


@dataclass
class IndicParlerBackendConfig:
    project_root: Path
    profile: str = "10h_clean"

    # The final Indic-Parler-TTS checkpoint as the first adaptation base.
    # Switch to ai4bharat/indic-parler-tts-pretrained only if you specifically
    # want the pre-finetuned base.
    model_name_or_path: str = "ai4bharat/indic-parler-tts"
    feature_extractor_name: str = "parler-tts/dac_44khZ_8kbps"
    description_tokenizer_name: str = "google/flan-t5-large"

    output_subdir: str = "runs/indic_parler"

    # RTX 3060 12 GB-safe starting point.
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    audio_encoder_per_device_batch_size: int = 1
    dataloader_num_workers: int = 2
    preprocessing_num_workers: int = 2

    learning_rate: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = 10_000

    save_steps: int = 1_000
    eval_steps: int = 1_000
    logging_steps: int = 25

    min_duration_sec: float = 1.0
    max_duration_sec: float = 10.0

    # float16 on RTX 3060 unless bfloat16 is verified. If loss becomes
    # unstable, lower LR before abandoning the model.
    dtype: str = "float16"
    attn_implementation: str = "sdpa"
    report_to: str = "none"

    overwrite_output_dir: bool = False
    speaker_name: str = "Divya"
    voice_description: str = (
        "Divya speaks Hindi in a clear, close-sounding narrator voice, "
        "with a natural storytelling pace, moderate expressiveness, balanced pitch, "
        "and very clear audio with no background noise."
    )

    parler_repo_dir: Optional[Path] = None
    smoke: bool = False

    # Path to a Python interpreter that has parler_tts installed.
    # parler-tts hard-pins transformers==4.46.1, but our coqui-tts side
    # needs transformers>=4.57 — they cannot coexist in one venv. So the
    # backend lives in the coqui venv (for the exporter, which only needs
    # soundfile/torchaudio) but spawns the actual `accelerate launch` in
    # a dedicated parler venv. Default points at the venv set up by
    # `requirements-indic-parler.txt`.
    python_executable: Optional[str] = None


class IndicParlerBackend(TrainerBackend):
    """The `--backend indic-parler` entry point.

    Compatible with `hindi_tts_builder.train.backends.base.TrainerBackend`
    contract: prepare(), train(), export_engine(). Heavy upstream imports
    (parler-tts, transformers) are deferred to inference / training time.
    """

    name = "indic-parler"
    supports_resume = True

    def __init__(self, project_root: Path, config: Optional[IndicParlerBackendConfig] = None):
        super().__init__(project_root)
        if config is None:
            config = IndicParlerBackendConfig(
                project_root=self.project_root,
                profile=os.environ.get("HINDI_TTS_PROFILE", "10h_clean"),
                smoke=bool(int(os.environ.get("HINDI_TTS_SMOKE", "0"))),
            )
        self.config = config
        self.run_root = self.project_root / config.output_subdir / config.profile
        self.export_dir = self.project_root / "exports" / "indic_parler" / config.profile
        self.parler_repo_dir = (
            config.parler_repo_dir
            or self.project_root / ".external" / "parler-tts"
        ).resolve()

    # ------------------------------------------------------------------
    # TrainerBackend contract
    # ------------------------------------------------------------------
    def prepare(self) -> dict:
        """Run dataset export so train() has a HF-loadable folder ready."""
        out = self.export_dataset(overwrite=self.config.overwrite_output_dir)
        card = json.loads((out / "dataset_card.json").read_text(encoding="utf-8"))
        return {
            "profile": self.config.profile,
            "export_dir": str(out),
            "splits": card.get("splits", {}),
            "rejections": len(card.get("rejections", [])),
        }

    def train(self) -> None:
        self._train(smoke=self.config.smoke, overwrite_export=False)

    def export_engine(self) -> Path:
        return self._export_engine()

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def export_dataset(self, overwrite: bool = False) -> Path:
        # Lazy import so plain `--help` doesn't pull soundfile/torchaudio.
        from hindi_tts_builder.exporters.indic_parler import (
            IndicParlerExportConfig,
            IndicParlerExporter,
        )
        exporter = IndicParlerExporter(
            IndicParlerExportConfig(
                project_root=self.project_root,
                profile=self.config.profile,
                out_dir=self.export_dir,
                speaker_name=self.config.speaker_name,
                voice_description=self.config.voice_description,
                min_duration_sec=self.config.min_duration_sec,
                max_duration_sec=self.config.max_duration_sec,
                overwrite=overwrite,
            )
        )
        return exporter.export()

    def ensure_parler_repo(self) -> Path:
        if self.parler_repo_dir.exists():
            return self.parler_repo_dir
        self.parler_repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call([
            "git", "clone",
            "https://github.com/huggingface/parler-tts.git",
            str(self.parler_repo_dir),
        ])
        return self.parler_repo_dir

    def write_training_config(
        self,
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
    ) -> Path:
        self.run_root.mkdir(parents=True, exist_ok=True)
        dataset_script = self.export_dir / "indic_parler_dataset.py"
        if not dataset_script.exists():
            raise FileNotFoundError(
                f"Dataset export missing: {dataset_script}. Call prepare()/export_dataset() first."
            )

        output_dir = self.run_root / "checkpoints"
        temp_codec_dir = self.run_root / "codec_cache"
        vectorized_dir = self.run_root / "vectorized_dataset"

        cfg = {
            # Model
            "model_name_or_path": self.config.model_name_or_path,
            "feature_extractor_name": self.config.feature_extractor_name,
            "description_tokenizer_name": self.config.description_tokenizer_name,
            "prompt_tokenizer_name": self.config.model_name_or_path,
            "attn_implementation": self.config.attn_implementation,
            "freeze_text_encoder": True,

            # Dataset
            "train_dataset_name": str(dataset_script),
            # Parler's training script expects `_config_name` strings (split by
            # "+" internally for multi-dataset). For our single local dataset
            # script, "default" is the only config — without this, data.py
            # crashes with AttributeError on None.split("+").
            "train_dataset_config_name": "default",
            "train_split_name": "train",
            "train_metadata_dataset_name": str(dataset_script),
            "eval_dataset_name": str(dataset_script),
            "eval_dataset_config_name": "default",
            "eval_split_name": "validation",
            "eval_metadata_dataset_name": str(dataset_script),
            "target_audio_column_name": "audio",
            "prompt_column_name": "text",
            "description_column_name": "description",
            "id_column_name": "id",
            "max_duration_in_seconds": self.config.max_duration_sec,
            "min_duration_in_seconds": self.config.min_duration_sec,
            "preprocessing_num_workers": self.config.preprocessing_num_workers,
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "eval_dataloader_num_workers": 0,
            "temporary_save_to_disk": str(temp_codec_dir),
            "save_to_disk": str(vectorized_dir),
            "save_codec_steps": 250,

            # Training
            "output_dir": str(output_dir),
            "overwrite_output_dir": self.config.overwrite_output_dir,
            "do_train": True,
            "do_eval": True,
            "max_steps": self.config.max_steps,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "per_device_eval_batch_size": 1,
            "audio_encoder_per_device_batch_size": self.config.audio_encoder_per_device_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "warmup_steps": self.config.warmup_steps,
            "lr_scheduler_type": "cosine",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "dtype": self.config.dtype,

            # Logging / checkpoints
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
            "eval_steps": self.config.eval_steps,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "save_total_limit": 3,
            "report_to": self.config.report_to,
            "wandb_project": "hindi-tts-builder",
            "wandb_run_name": f"indic-parler-{self.config.profile}-{int(time.time())}",

            # Eval metrics are expensive and can break small local runs.
            # Disabled here; we run our own ASR-roundtrip eval after export.
            "compute_clap_similarity_metric": False,
            "compute_noise_level_metric": False,
            "predict_with_generate": False,

            # Safety / reproducibility
            "seed": 42,
            "remove_unused_columns": False,
        }

        if max_train_samples is not None:
            cfg["max_train_samples"] = max_train_samples
        if max_eval_samples is not None:
            cfg["max_eval_samples"] = max_eval_samples

        config_path = self.run_root / "parler_training_config.json"
        config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return config_path

    def _train(
        self,
        smoke: bool = False,
        export_dataset_if_missing: bool = True,
        overwrite_export: bool = False,
    ) -> None:
        if export_dataset_if_missing and not (self.export_dir / "indic_parler_dataset.py").exists():
            self.export_dataset(overwrite=overwrite_export)

        repo = self.ensure_parler_repo()

        if smoke:
            config_path = self.write_training_config(
                max_train_samples=64,
                max_eval_samples=16,
            )
        else:
            config_path = self.write_training_config()

        script = repo / "training" / "run_parler_tts_training.py"
        if not script.exists():
            raise FileNotFoundError(f"Parler training script not found: {script}")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH', '')}"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Our exporter writes a custom dataset script (indic_parler_dataset.py)
        # which datasets>=2.16 treats as "remote code" and refuses to run
        # without explicit consent. The Parler training script doesn't pass
        # trust_remote_code=True to load_dataset, so we whitelist via env.
        env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

        # Resolve the python interpreter for the parler venv.
        # Order: explicit config → HINDI_TTS_PARLER_PYTHON env → default
        # /root/parler-venv/bin/python3 → fall back to current sys.executable
        # (which only works if the user installed parler in the same venv).
        python_exe = (
            self.config.python_executable
            or os.environ.get("HINDI_TTS_PARLER_PYTHON")
            or "/root/parler-venv/bin/python3"
        )
        if not Path(python_exe).exists():
            print(
                f"[parler] WARNING: {python_exe} not found, falling back to "
                f"{sys.executable}. If parler_tts is not installed there this "
                "will fail with ModuleNotFoundError. See ROADMAP.md dual-venv setup."
            )
            python_exe = sys.executable

        cmd = [
            python_exe,
            "-m", "accelerate.commands.launch",
            "--num_processes=1",
            str(script),
            str(config_path),
        ]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(repo), env=env)

    def latest_checkpoint(self) -> Path:
        ckpt_root = self.run_root / "checkpoints"
        if not ckpt_root.exists():
            raise FileNotFoundError(f"No checkpoint dir: {ckpt_root}")
        candidates = sorted(
            [p for p in ckpt_root.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        if candidates:
            return candidates[-1]
        if (ckpt_root / "config.json").exists():
            return ckpt_root
        raise FileNotFoundError(f"No checkpoints found in {ckpt_root}")

    def _export_engine(self, engine_name: str = "indic_parler") -> Path:
        ckpt = self.latest_checkpoint()
        engine_dir = self.project_root / "engines" / engine_name / self.config.profile
        if engine_dir.exists():
            shutil.rmtree(engine_dir)
        engine_dir.mkdir(parents=True, exist_ok=True)

        model_dir = engine_dir / "model"
        shutil.copytree(ckpt, model_dir)

        manifest = {
            "engine_type": "indic-parler",
            "profile": self.config.profile,
            "model_dir": "model",
            "base_model": self.config.model_name_or_path,
            "feature_extractor_name": self.config.feature_extractor_name,
            "description_tokenizer_name": self.config.description_tokenizer_name,
            "speaker_name": self.config.speaker_name,
            "default_description": self.config.voice_description,
            "sample_rate": 44100,
            "pronunciation_dict_required": True,
        }
        (engine_dir / "engine.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Stub pronunciation_dict — required by manifest. Replace per project.
        pron_dict = {
            "Jiang Yuechu": "जियांग यूएछू",
            "Pei Changqing": "पेई चांगछिंग",
            "system": "सिस्टम",
            "activate": "एक्टिवेट",
        }
        (engine_dir / "pronunciation_dict.json").write_text(
            json.dumps(pron_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return engine_dir
