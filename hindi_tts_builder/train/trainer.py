"""The main Trainer class.

Uses Coqui TTS (the `coqui-tts` package) under the hood because re-implementing
VITS from scratch is a multi-month research project and Coqui's VITS is
battle-tested. We drive it with our project's data + config + tokenizer.

If `coqui-tts` is not installed, `Trainer.train()` raises a helpful
ImportError with install instructions. Everything else (config, tokenizer,
dataset, checkpoint manager) works without Coqui so you can still introspect
and prepare the project.

High-level flow:

    trainer = Trainer(project_root)
    trainer.prepare()       # fit tokenizer, validate data, write configs
    trainer.train()         # long-running; calls Coqui's trainer
    trainer.export_engine() # bundle for inference
"""
from __future__ import annotations
from pathlib import Path
import json
import shutil

from hindi_tts_builder.frontend.pipeline import HindiFrontend
from hindi_tts_builder.train.config import TrainingConfig
from hindi_tts_builder.train.dataset import read_split_csv
from hindi_tts_builder.train.tokenizer import HindiTokenizer
from hindi_tts_builder.utils import get_logger
from hindi_tts_builder.utils.project import ProjectPaths, load_config


_COQUI_INSTALL_MSG = (
    "The `coqui-tts` package is required for training. Install with:\n\n"
    "    pip install coqui-tts\n\n"
    "Then re-run. All other trainer features (config, tokenizer, dataset "
    "preparation) work without it."
)


def _try_import_coqui():
    """Return the Coqui modules we need, or None if unavailable."""
    try:
        from TTS.tts.configs.vits_config import VitsConfig  # type: ignore
        from TTS.tts.models.vits import Vits, VitsAudioConfig  # type: ignore
        from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore
        from TTS.utils.audio import AudioProcessor  # type: ignore
        from trainer import Trainer as CoquiTrainer, TrainerArgs  # type: ignore
        return {
            "VitsConfig": VitsConfig,
            "VitsAudioConfig": VitsAudioConfig,
            "BaseDatasetConfig": BaseDatasetConfig,
            "Vits": Vits,
            "AudioProcessor": AudioProcessor,
            "CoquiTrainer": CoquiTrainer,
            "TrainerArgs": TrainerArgs,
        }
    except ImportError:
        return None


def _hindi_csv_formatter(root_path, meta_file, **kwargs):
    """Coqui-compatible formatter for our train/val CSVs.

    Our build_training_set writes pipe-delimited CSVs at
    `<project>/training_set/{train,val,test}.csv` with header
    `audio_path | raw_text | processed_text | duration | source_id`.

    `audio_path` is recorded as a relative path from the project root
    (one directory up from training_set/). Coqui's loader passes
    root_path = the directory holding the CSV (training_set), so we
    resolve audio paths against `root_path.parent`.
    """
    import csv as _csv
    csv_file = Path(root_path) / meta_file
    project_root = Path(root_path).parent
    items = []
    with open(csv_file, encoding="utf-8") as f:
        reader = _csv.DictReader(f, delimiter="|")
        for row in reader:
            ap = (row.get("audio_path") or "").strip()
            if not ap:
                continue
            audio_file = ap if Path(ap).is_absolute() else str(project_root / ap)
            text = (row.get("processed_text") or row.get("raw_text") or "").strip()
            if not text:
                continue
            items.append({
                "text": text,
                "audio_file": audio_file,
                "speaker_name": "hindi",
                "root_path": str(project_root),
            })
    return items


def _register_formatter():
    """Inject our formatter into Coqui's formatters namespace so
    BaseDatasetConfig(formatter='hindi_csv') resolves to it.
    """
    try:
        from TTS.tts.datasets import formatters as _fmt  # type: ignore
        if not hasattr(_fmt, "hindi_csv"):
            setattr(_fmt, "hindi_csv", _hindi_csv_formatter)
    except ImportError:
        pass


class Trainer:
    """Project-scoped trainer.

    Parameters
    ----------
    project_root : Path
        Path to a project directory that has already been through the data
        pipeline. Must contain `training_set/train.csv` and `training_set/val.csv`.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.paths = ProjectPaths(self.project_root)
        self.project_config = load_config(self.project_root)
        self.log = get_logger("train", self.paths.logs / "train.log")

        # Training config lives separately so you can tune it without touching
        # the main project config.
        self.training_config_path = self.project_root / "training_config.yaml"
        self.training_config = TrainingConfig.load(self.training_config_path)

        # Produced by prepare()
        self.tokenizer: HindiTokenizer | None = None

    # ------------------------------------------------------------------
    # Preparation (runs before training; does not require GPU or Coqui)
    # ------------------------------------------------------------------
    def prepare(self) -> dict:
        """Validate data, fit tokenizer, save everything.

        Returns a summary dict you can print.
        """
        self.log.info("=== Training prepare ===")
        train_csv = self.paths.training_set / "train.csv"
        val_csv = self.paths.training_set / "val.csv"
        if not train_csv.exists():
            raise FileNotFoundError(
                f"Training set not found at {train_csv}. "
                "Run the data pipeline first."
            )

        train = read_split_csv(train_csv)
        val = read_split_csv(val_csv) if val_csv.exists() else []
        self.log.info(f"Loaded {len(train)} train records, {len(val)} val records.")

        if not train:
            raise RuntimeError("Training CSV has no records.")

        # Fit tokenizer on combined train+val texts (closed vocab)
        tok = HindiTokenizer()
        tok.fit(r.processed_text for r in train + val)
        self.tokenizer = tok
        tok_path = self.paths.checkpoints / "tokenizer.json"
        tok.save(tok_path)
        self.log.info(f"Tokenizer vocab size: {tok.vocab_size}. Saved to {tok_path.name}")

        # Save a snapshot of training config so re-runs use the same settings
        self.training_config.save(self.training_config_path)

        # Compute total duration stats
        train_hours = sum(r.duration for r in train) / 3600.0
        val_hours = sum(r.duration for r in val) / 3600.0

        # Persist a manifest of ready-to-train state
        ready = {
            "train_records": len(train),
            "val_records": len(val),
            "train_hours": round(train_hours, 3),
            "val_hours": round(val_hours, 3),
            "vocab_size": tok.vocab_size,
        }
        (self.paths.training_set / "ready.json").write_text(
            json.dumps(ready, indent=2), encoding="utf-8"
        )
        self.log.info(f"Prepare complete: {ready}")
        return ready

    # ------------------------------------------------------------------
    # Training (requires Coqui TTS + GPU)
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run VITS training end-to-end. Requires `coqui-tts` installed and
        a CUDA-capable GPU. Resumable if interrupted — call `train()` again
        and it picks up from the latest checkpoint.
        """
        coqui = _try_import_coqui()
        if coqui is None:
            raise ImportError(_COQUI_INSTALL_MSG)

        if self.tokenizer is None:
            tok_path = self.paths.checkpoints / "tokenizer.json"
            if not tok_path.exists():
                raise RuntimeError(
                    "Tokenizer not found. Call `trainer.prepare()` first."
                )
            self.tokenizer = HindiTokenizer.load(tok_path)

        self.log.info("=== Training run starting ===")

        tc = self.training_config
        mc = tc.model

        # Build Coqui's VitsConfig from our typed config
        VitsConfig = coqui["VitsConfig"]
        VitsAudioConfig = coqui["VitsAudioConfig"]
        BaseDatasetConfig = coqui["BaseDatasetConfig"]

        # Register our custom formatter so the dataset loader can find it
        # by the string name we pass below.
        _register_formatter()

        # Coqui >=0.27 wants a BaseDatasetConfig instance (Coqpit), not a dict.
        dataset_cfg = BaseDatasetConfig(
            formatter="hindi_csv",
            dataset_name="hindi_single_speaker",
            path=str(self.paths.training_set),
            meta_file_train="train.csv",
            meta_file_val="val.csv",
            language="hi",
        )

        # Coqui >=0.27 requires a VitsAudioConfig instance (attribute access);
        # passing a plain dict crashes init_from_config with
        # AttributeError: 'dict' object has no attribute 'hop_length'.
        audio_cfg = VitsAudioConfig(
            sample_rate=mc.sample_rate,
            hop_length=mc.hop_length,
            win_length=mc.win_length,
            num_mels=mc.n_mel_channels,
            fft_size=mc.n_fft,
            mel_fmin=mc.mel_fmin,
            mel_fmax=mc.mel_fmax,
        )

        config = VitsConfig(
            run_name=self.project_config.get("name", "hindi_tts"),
            output_path=str(self.paths.checkpoints),
            audio=audio_cfg,
            # Training
            batch_size=tc.batch_size,
            eval_batch_size=tc.val_batch_size,
            num_loader_workers=tc.num_workers,
            num_eval_loader_workers=max(1, tc.num_workers // 2),
            epochs=tc.epochs,
            save_step=tc.checkpoint_every_steps,
            save_n_checkpoints=tc.keep_last_n_checkpoints,
            print_step=50,
            log_model_step=tc.sample_every_steps,
            mixed_precision=(tc.mixed_precision != "none"),
            lr_gen=tc.optim.learning_rate_gen,
            lr_disc=tc.optim.learning_rate_disc,
            grad_clip=[tc.optim.grad_clip_norm, tc.optim.grad_clip_norm],
            max_audio_len=int(tc.max_audio_length_sec * mc.sample_rate),
            datasets=[dataset_cfg],
            phonemizer=None,              # we pre-tokenize; Coqui uses our chars directly
            use_phonemes=False,
            text_cleaner=None,            # our frontend handled cleaning already
            enable_eos_bos_chars=True,
        )

        # Find latest checkpoint for resumption
        from hindi_tts_builder.train.checkpoint import latest_checkpoint
        restore_path = latest_checkpoint(self.paths.checkpoints)

        TrainerArgs = coqui["TrainerArgs"]
        CoquiTrainer = coqui["CoquiTrainer"]

        args = TrainerArgs(
            continue_path=str(self.paths.checkpoints) if restore_path else "",
            restore_path=str(restore_path) if restore_path else "",
        )

        Vits = coqui["Vits"]
        model = Vits.init_from_config(config)

        trainer = CoquiTrainer(
            args,
            config,
            str(self.paths.checkpoints),
            model=model,
        )
        trainer.fit()
        self.log.info("=== Training run complete ===")

    # ------------------------------------------------------------------
    # Engine export (packages model + tokenizer + frontend state for inference)
    # ------------------------------------------------------------------
    def export_engine(self) -> Path:
        """Copy the final model + tokenizer + frontend dictionary + manifest
        into `projects/<name>/engine/`. That folder is the TTSEngine's load path.
        """
        from hindi_tts_builder.train.checkpoint import latest_checkpoint

        engine_dir = self.paths.engine
        engine_dir.mkdir(parents=True, exist_ok=True)

        # 1. Model checkpoint
        latest = latest_checkpoint(self.paths.checkpoints)
        if latest is None:
            # Coqui writes checkpoints differently (`best_model.pth`); look for those too.
            candidates = list(self.paths.checkpoints.rglob("best_model*.pth")) + \
                         list(self.paths.checkpoints.rglob("*.pth"))
            if not candidates:
                raise RuntimeError(
                    "No checkpoint found to export. Train first."
                )
            # Pick most recent by mtime
            latest = max(candidates, key=lambda p: p.stat().st_mtime)

        shutil.copy2(latest, engine_dir / "model.pt")

        # 2. Tokenizer
        tok_path = self.paths.checkpoints / "tokenizer.json"
        if tok_path.exists():
            shutil.copy2(tok_path, engine_dir / "tokenizer.json")

        # 3. Frontend pronunciation dictionary (may not exist — that's ok)
        dict_candidates = list(self.project_root.rglob("pronunciation_dict.json"))
        if dict_candidates:
            shutil.copy2(dict_candidates[0], engine_dir / "pronunciation_dict.json")

        # 4. Training config snapshot
        if self.training_config_path.exists():
            shutil.copy2(self.training_config_path, engine_dir / "training_config.yaml")

        # 5. Manifest describing engine
        import hindi_tts_builder
        manifest = {
            "engine_version": 1,
            "package_version": hindi_tts_builder.__version__,
            "project_name": self.project_config.get("name"),
            "language": "hi",
            "sample_rate": self.training_config.model.sample_rate,
            "model_type": "vits",
            "frontend": {
                "prosody_tokens": HindiFrontend.prosody_tokens(),
                "apply_schwa_deletion": True,
                "apply_prosody": True,
            },
        }
        (engine_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.log.info(f"Engine exported to {engine_dir}")
        return engine_dir
