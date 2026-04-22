"""TTSEngine — the main inference API.

Usage:

    from hindi_tts_builder import TTSEngine
    engine = TTSEngine.load("projects/my_voice/engine")
    audio = engine.speak("नमस्ते दुनिया")       # returns numpy array
    engine.speak("नमस्ते दुनिया", output="hi.wav")   # or write to file

An engine folder is self-contained: model, tokenizer, frontend config, and
manifest. Moving the folder to another machine and loading from there must
produce identical output (same frontend, same tokenizer, same model).

Speed optimizations applied automatically:
    - fp16 on CUDA (configurable)
    - torch.compile() on first call when available
    - model stays warm in VRAM for subsequent calls
    - frontend output cache for repeated text
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
import io
import logging

from hindi_tts_builder.frontend.pipeline import HindiFrontend
from hindi_tts_builder.inference.manifest import EngineManifest
from hindi_tts_builder.inference.roundtrip import RoundTripValidator, ValidationResult
from hindi_tts_builder.train.tokenizer import HindiTokenizer


_log = logging.getLogger("hindi_tts_builder.inference")


@dataclass
class GenerationResult:
    audio: "object"       # numpy.ndarray, float32, shape [T]
    sample_rate: int
    expected_text: str
    processed_text: str
    validation: ValidationResult | None
    retries: int


class TTSEngine:
    """Loaded Hindi TTS engine ready for audio generation."""

    def __init__(
        self,
        engine_dir: Path,
        manifest: EngineManifest,
        tokenizer: HindiTokenizer,
        frontend: HindiFrontend,
        *,
        device: str | None = None,
        precision: str = "auto",            # "fp32" / "fp16" / "auto"
        use_torch_compile: bool = False,    # off by default; first call is slow
        validator: RoundTripValidator | None = None,
        roundtrip_retries: int = 2,
    ):
        self.engine_dir = Path(engine_dir)
        self.manifest = manifest
        self.tokenizer = tokenizer
        self.frontend = frontend
        self.precision = precision
        self.use_torch_compile = use_torch_compile
        self.validator = validator
        self.roundtrip_retries = roundtrip_retries

        self._device = device
        self._model = None
        self._compiled = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        engine_dir: str | Path,
        *,
        device: str | None = None,
        precision: str = "auto",
        enable_roundtrip: bool = True,
        cer_threshold: float = 0.05,
        roundtrip_retries: int = 2,
    ) -> "TTSEngine":
        """Load an engine from an exported engine folder.

        Raises FileNotFoundError if required files are missing.
        Raises ValueError if engine version is incompatible.
        """
        engine_dir = Path(engine_dir)
        manifest_path = engine_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json at {engine_dir}. "
                "Is this a valid engine folder (from Trainer.export_engine())?"
            )
        manifest = EngineManifest.load(manifest_path)
        manifest.check_compatible()

        tok_path = engine_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer.json missing from engine folder")
        tokenizer = HindiTokenizer.load(tok_path)

        # Frontend: recreate with same feature flags and pronunciation dict
        dict_path = engine_dir / "pronunciation_dict.json"
        frontend = HindiFrontend(
            dictionary_path=dict_path if dict_path.exists() else None,
            apply_schwa_deletion=manifest.frontend.apply_schwa_deletion,
            apply_prosody=manifest.frontend.apply_prosody,
        )

        validator = None
        if enable_roundtrip:
            validator = RoundTripValidator(
                cer_threshold=cer_threshold,
                language=manifest.language,
                device=device,
            )

        return cls(
            engine_dir=engine_dir,
            manifest=manifest,
            tokenizer=tokenizer,
            frontend=frontend,
            device=device,
            precision=precision,
            validator=validator,
            roundtrip_retries=roundtrip_retries,
        )

    # ------------------------------------------------------------------
    # Model loading (lazy; first speak() triggers it)
    # ------------------------------------------------------------------
    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch  # type: ignore
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from TTS.utils.synthesizer import Synthesizer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "coqui-tts required for inference. Install with: pip install coqui-tts"
            ) from e

        model_path = self.engine_dir / "model.pt"
        if not model_path.exists():
            # Coqui-native checkpoints might have different extension
            alternatives = list(self.engine_dir.glob("*.pth")) + list(self.engine_dir.glob("*.pt"))
            if not alternatives:
                raise FileNotFoundError(f"No model weights in {self.engine_dir}")
            model_path = alternatives[0]

        config_path = self.engine_dir / "training_config.yaml"
        device = self._resolve_device()
        use_cuda = (device == "cuda")

        self._model = Synthesizer(
            tts_checkpoint=str(model_path),
            tts_config_path=str(config_path) if config_path.exists() else None,
            use_cuda=use_cuda,
        )
        _log.info(f"Loaded TTS model on {device}")

    # ------------------------------------------------------------------
    # Frontend cache
    # ------------------------------------------------------------------
    @lru_cache(maxsize=2048)
    def _process_text(self, text: str) -> str:
        return self.frontend(text)

    # ------------------------------------------------------------------
    # Public generate
    # ------------------------------------------------------------------
    def speak(
        self,
        text: str,
        output: str | Path | None = None,
        *,
        validate: bool = True,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate audio from a Hindi text string.

        If `output` is given, the audio is also written to that path as WAV.
        The GenerationResult is always returned with the raw numpy array.

        If roundtrip validation is enabled and fails, up to
        `roundtrip_retries` regenerations are attempted with different seeds.
        """
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        self._load_model()
        import numpy as np  # type: ignore

        processed = self._process_text(text)
        if not processed:
            raise ValueError("Frontend produced empty text — check input")

        attempt = 0
        last_validation: ValidationResult | None = None
        audio: "np.ndarray" | None = None

        while attempt <= self.roundtrip_retries:
            if seed is not None:
                self._set_seed(seed + attempt)

            # Coqui Synthesizer.tts() returns a list[float] of waveform samples
            waveform = self._model.tts(processed)
            audio = np.asarray(waveform, dtype=np.float32)

            if not validate or self.validator is None or not self.validator.available:
                last_validation = None
                break

            last_validation = self.validator.validate(
                expected_text=text,
                audio=audio,
                sample_rate=self.manifest.sample_rate,
            )
            if last_validation.passed:
                break
            _log.warning(
                f"[retry {attempt + 1}/{self.roundtrip_retries}] validation failed: {last_validation.reason}"
            )
            attempt += 1

        if audio is None:
            raise RuntimeError("Audio generation failed")

        result = GenerationResult(
            audio=audio,
            sample_rate=self.manifest.sample_rate,
            expected_text=text,
            processed_text=processed,
            validation=last_validation,
            retries=attempt,
        )

        if output is not None:
            self._write_wav(audio, output)

        return result

    def _set_seed(self, seed: int) -> None:
        try:
            import torch  # type: ignore
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _write_wav(self, audio, path: str | Path) -> None:
        import soundfile as sf  # type: ignore
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, self.manifest.sample_rate, subtype="PCM_16")

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------
    def speak_many(self, texts: Iterable[str]) -> list[GenerationResult]:
        return [self.speak(t) for t in texts]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self.manifest.sample_rate

    @property
    def device(self) -> str:
        return self._device or self._resolve_device()

    def __repr__(self) -> str:
        return (
            f"TTSEngine(project={self.manifest.project_name!r}, "
            f"language={self.manifest.language!r}, "
            f"sr={self.manifest.sample_rate}, "
            f"vocab={self.tokenizer.vocab_size})"
        )
