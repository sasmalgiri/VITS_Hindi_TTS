"""LoRA fine-tune backend for Indic-Parler-TTS.

Wraps the same upstream `run_parler_tts_training.py` as the full-FT
backend, but injects `use_lora`, `lora_rank`, `lora_alpha`, etc. into
the config JSON. The (previously-applied) `patch_parler_lora.py` hook
in the upstream script reads those fields and wraps the model with
peft.LoraConfig.

Output is a small adapter directory (~50 MB) instead of a full ~3.5 GB
checkpoint. The adapter goes into:

    projects/<name>/voice_library/adapters/<adapter_name>/
        adapter_config.json
        adapter_model.safetensors
        voice_card.json

Use voice-add to register it in the voice library, then speak-voice to
render with it.
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
from hindi_tts_builder.train.backends.indic_parler import IndicParlerBackendConfig


@dataclass
class IndicParlerLoraConfig(IndicParlerBackendConfig):
    """Same as IndicParlerBackendConfig + LoRA hyperparameters."""
    adapter_name: str = "narrator"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "out_proj"]
    )
    # LoRA tolerates higher LR than full-FT — typical: 1e-4 vs 5e-6
    learning_rate: float = 1e-4
    # LoRA needs more steps for equivalent adaptation; default higher than full-FT
    max_steps: int = 30_000

    # Dynamic batch strategy. None = auto-pick from free VRAM + corpus size.
    # Set to an explicit int to override (used by --per-device-batch / --grad-accum).
    auto_batch: bool = True
    explicit_per_device_batch: int | None = None
    explicit_grad_accum: int | None = None


class IndicParlerLoraBackend(TrainerBackend):
    """LoRA fine-tune backend. Identical wiring to IndicParlerBackend
    except the config carries LoRA fields and the saved checkpoint is
    a small adapter, not a full model.
    """

    name = "indic-parler-lora"
    supports_resume = True

    def __init__(self, project_root: Path, config: Optional[IndicParlerLoraConfig] = None):
        super().__init__(project_root)
        if config is None:
            config = IndicParlerLoraConfig(
                project_root=self.project_root,
                profile=os.environ.get("HINDI_TTS_PROFILE", "10h_clean"),
                adapter_name=os.environ.get("HINDI_TTS_ADAPTER_NAME", "narrator"),
                smoke=bool(int(os.environ.get("HINDI_TTS_SMOKE", "0"))),
            )
        self.config = config
        self.run_root = self.project_root / config.output_subdir / f"lora_{config.adapter_name}_{config.profile}"
        self.export_dir = self.project_root / "exports" / "indic_parler" / config.profile
        self.parler_repo_dir = (
            config.parler_repo_dir
            or self.project_root / ".external" / "parler-tts"
        ).resolve()

    def prepare(self) -> dict:
        """Reuse the full-FT exporter — dataset shape is identical."""
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
                overwrite=self.config.overwrite_output_dir,
            )
        )
        out = exporter.export()
        card = json.loads((out / "dataset_card.json").read_text(encoding="utf-8"))
        return {
            "profile": self.config.profile,
            "adapter_name": self.config.adapter_name,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_target_modules": self.config.lora_target_modules,
            "export_dir": str(out),
            "splits": card.get("splits", {}),
            "rejections": len(card.get("rejections", [])),
        }

    def train(self) -> None:
        self._train(smoke=self.config.smoke, overwrite_export=False)

    def export_engine(self) -> Path:
        return self._export_adapter()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def write_training_config(
        self,
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
    ) -> Path:
        """Mirror IndicParlerBackend.write_training_config but add LoRA fields.

        If `auto_batch=True` (default), picks per_device_batch and grad_accum
        live from free VRAM + corpus size via dynamic_batch.compute_strategy().
        """
        self.run_root.mkdir(parents=True, exist_ok=True)
        dataset_script = self.export_dir / "indic_parler_dataset.py"
        if not dataset_script.exists():
            raise FileNotFoundError(
                f"Dataset export missing: {dataset_script}. Call prepare() first."
            )

        output_dir = self.run_root / "checkpoints"
        temp_codec_dir = self.run_root / "codec_cache"
        vectorized_dir = self.run_root / "vectorized_dataset"

        # ---- Dynamic batch strategy ----
        from hindi_tts_builder.train.dynamic_batch import (
            compute_strategy, total_hours_from_csv,
        )
        train_csv = (self.project_root / "training_set" /
                     f"train_{self.config.profile}.csv")
        total_hours = total_hours_from_csv(train_csv)
        strategy = compute_strategy(
            total_hours=total_hours,
            max_audio_seconds=self.config.max_duration_sec,
            explicit_per_device=(
                self.config.explicit_per_device_batch
                if not self.config.auto_batch
                else self.config.explicit_per_device_batch
            ),
            explicit_grad_accum=(
                self.config.explicit_grad_accum
                if not self.config.auto_batch
                else self.config.explicit_grad_accum
            ),
            is_lora=True,
        )
        per_device_batch = strategy.per_device_batch
        grad_accum_steps = strategy.grad_accum
        print(f"[lora batch strategy] {strategy.reasoning}")

        cfg = {
            # Model
            "model_name_or_path": self.config.model_name_or_path,
            "feature_extractor_name": self.config.feature_extractor_name,
            "description_tokenizer_name": self.config.description_tokenizer_name,
            "prompt_tokenizer_name": self.config.model_name_or_path,
            "attn_implementation": self.config.attn_implementation,
            "freeze_text_encoder": True,

            # === LoRA (read by patched upstream training script) ===
            "use_lora": True,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "lora_target_modules": self.config.lora_target_modules,

            # Dataset
            "train_dataset_name": str(dataset_script),
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
            # Dynamic — see strategy.reasoning printed above
            "per_device_train_batch_size": per_device_batch,
            "per_device_eval_batch_size": 1,
            "audio_encoder_per_device_batch_size": self.config.audio_encoder_per_device_batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
            "_dynamic_batch_strategy": strategy.reasoning,
            "_effective_batch_size": strategy.effective_batch,
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
            "save_total_limit": 5,  # adapters are small, keep more
            "report_to": self.config.report_to,
            "wandb_project": "hindi-tts-builder",
            "wandb_run_name": f"indic-parler-lora-{self.config.adapter_name}-{self.config.profile}-{int(time.time())}",

            # Eval metrics off (we run our own ASR-roundtrip)
            "compute_clap_similarity_metric": False,
            "compute_noise_level_metric": False,
            "predict_with_generate": False,

            # Safety
            "seed": 42,
            "remove_unused_columns": False,
        }

        if max_train_samples is not None:
            cfg["max_train_samples"] = max_train_samples
        if max_eval_samples is not None:
            cfg["max_eval_samples"] = max_eval_samples

        config_path = self.run_root / "parler_lora_config.json"
        config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return config_path

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

    def _train(
        self,
        smoke: bool = False,
        export_dataset_if_missing: bool = True,
        overwrite_export: bool = False,
    ) -> None:
        if export_dataset_if_missing and not (self.export_dir / "indic_parler_dataset.py").exists():
            self.prepare()

        repo = self.ensure_parler_repo()

        # Verify LoRA patch is present in the script
        script = repo / "training" / "run_parler_tts_training.py"
        if "PATCH: optional LoRA wrapping via peft" not in script.read_text(encoding="utf-8"):
            raise RuntimeError(
                f"Upstream script missing LoRA patch. Run: python /root/patch_parler_lora.py"
            )

        if smoke:
            config_path = self.write_training_config(max_train_samples=64, max_eval_samples=16)
        else:
            config_path = self.write_training_config()

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH', '')}"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

        python_exe = (
            self.config.python_executable
            or os.environ.get("HINDI_TTS_PARLER_PYTHON")
            or "/root/parler-venv/bin/python3"
        )
        if not Path(python_exe).exists():
            print(f"[parler-lora] WARNING: {python_exe} not found, falling back to {sys.executable}")
            python_exe = sys.executable

        cmd = [
            python_exe,
            "-m", "accelerate.commands.launch",
            "--num_processes=1",
            str(script),
            str(config_path),
        ]
        print("Running (LoRA):", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(repo), env=env)

    def latest_adapter_checkpoint(self) -> Path:
        """Find the latest checkpoint dir that contains adapter_model.safetensors."""
        ckpt_root = self.run_root / "checkpoints"
        if not ckpt_root.exists():
            raise FileNotFoundError(f"no checkpoint dir: {ckpt_root}")
        candidates = []
        for p in ckpt_root.glob("checkpoint-*"):
            if not p.is_dir():
                continue
            # PeftModel checkpoints have adapter_model.safetensors
            if (p / "adapter_model.safetensors").exists() or (p / "adapter_config.json").exists():
                step = int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
                candidates.append((step, p))
        if candidates:
            candidates.sort()
            return candidates[-1][1]
        # Fall back to checkpoint root if it has adapter files directly
        if (ckpt_root / "adapter_model.safetensors").exists():
            return ckpt_root
        raise FileNotFoundError(f"no LoRA adapter checkpoint under {ckpt_root}")

    def _export_adapter(self) -> Path:
        """Bundle the trained adapter into projects/<name>/voice_library/adapters/<adapter_name>/."""
        ckpt = self.latest_adapter_checkpoint()
        lib_dir = self.project_root / "voice_library" / "adapters" / self.config.adapter_name
        if lib_dir.exists():
            shutil.rmtree(lib_dir)
        lib_dir.mkdir(parents=True, exist_ok=True)

        # Copy ALL adapter-related files from the checkpoint
        for f in ckpt.iterdir():
            # Skip optimizer/scheduler/random states — adapter only
            if f.name in {"optimizer.bin", "scheduler.bin", "scaler.pt",
                          "random_states_0.pkl", "rng_state.pth"}:
                continue
            # Skip the full pytorch_model.bin if it accidentally exists (shouldn't for peft)
            if f.name == "pytorch_model.bin":
                continue
            if f.is_file():
                shutil.copy2(f, lib_dir / f.name)
            elif f.is_dir():
                shutil.copytree(f, lib_dir / f.name)

        # Voice card metadata
        voice_card = {
            "adapter_name": self.config.adapter_name,
            "speaker_name": self.config.speaker_name,
            "default_description": self.config.voice_description,
            "base_model": self.config.model_name_or_path,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_target_modules": self.config.lora_target_modules,
            "training_profile": self.config.profile,
            "training_steps": self.config.max_steps,
            "learning_rate": self.config.learning_rate,
            "sample_rate": 44100,
            "language": "hi",
            "trained_from_checkpoint": str(ckpt),
        }
        (lib_dir / "voice_card.json").write_text(
            json.dumps(voice_card, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return lib_dir
