"""Multi-voice library inference engine.

Loads base Indic-Parler-TTS once + hot-swaps LoRA adapters at inference
time to render different voices. Storage cost per voice: ~50 MB (adapter)
vs ~3.5 GB (full fine-tune per voice).

Library layout (inside a project):
    projects/<name>/voice_library/
        adapters/
            male_narrator_16h/
                adapter_config.json
                adapter_model.safetensors
                voice_card.json     # description prompt, sample rate, etc.
            female_dramatic_150h/
                adapter_config.json
                adapter_model.safetensors
                voice_card.json
            ...

Or globally at:
    /root/hindi-tts/voice_library/adapters/<voice_name>/

Usage:
    lib = VoiceLibrary(library_dir="/root/hindi-tts/voice_library")
    lib.list_voices()                                  # ["male_narrator_16h", ...]
    audio, sr = lib.synthesize("नमस्ते दुनिया।",
                                voice="male_narrator_16h")
    audio, sr = lib.synthesize("...", voice="female_dramatic_150h")
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch


def _basic_tts_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("।।", "।")
    return text


def _safe_squeeze_audio(generation: torch.Tensor) -> np.ndarray:
    arr = generation.detach().to(torch.float32).cpu().numpy()
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 0:
        arr = np.asarray([float(arr)], dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


class VoiceCard:
    """Metadata for one trained voice in the library."""

    def __init__(self, adapter_dir: Path):
        self.adapter_dir = Path(adapter_dir)
        card_path = self.adapter_dir / "voice_card.json"
        if card_path.exists():
            self.data = json.loads(card_path.read_text(encoding="utf-8"))
        else:
            # Bare adapter without our metadata — best-effort defaults
            self.data = {
                "adapter_name": self.adapter_dir.name,
                "default_description": (
                    "A speaker speaks Hindi clearly in a close-sounding "
                    "recording with no background noise."
                ),
                "sample_rate": 44100,
                "language": "hi",
            }

    @property
    def name(self) -> str:
        return self.data.get("adapter_name", self.adapter_dir.name)

    @property
    def description(self) -> str:
        return self.data.get("default_description", "")

    @property
    def sample_rate(self) -> int:
        return int(self.data.get("sample_rate", 44100))

    def __repr__(self) -> str:
        return f"VoiceCard(name={self.name!r}, lang={self.data.get('language', 'hi')})"


class VoiceLibrary:
    """Base Parler + N hot-swappable LoRA adapters. Load once, switch voices fast."""

    def __init__(
        self,
        library_dir: str | Path,
        base_model: str = "ai4bharat/indic-parler-tts",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        self.library_dir = Path(library_dir).resolve()
        self.adapters_dir = self.library_dir / "adapters"
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model

        # Heavy imports lazy
        from parler_tts import ParlerTTSForConditionalGeneration  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        torch_dtype = self._resolve_dtype(dtype)
        self.torch_dtype = torch_dtype

        self.base_model = ParlerTTSForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=torch_dtype,
        ).to(self.device)
        self.base_model.eval()

        self.prompt_tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model.config.text_encoder._name_or_path
        )

        # Adapter state
        self._loaded_adapter: Optional[str] = None
        self._peft_model = None

        # Cache for description encoding
        self._desc_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _resolve_dtype(self, dtype: Optional[str]) -> torch.dtype:
        if dtype == "float32":
            return torch.float32
        if dtype == "bfloat16":
            return torch.bfloat16
        if dtype == "float16":
            return torch.float16
        # auto
        if self.device.startswith("cuda") and torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    # ------------------------------------------------------------------
    # Library management
    # ------------------------------------------------------------------
    def list_voices(self) -> list[str]:
        """Return sorted list of voice names available in the library."""
        return sorted(
            p.name for p in self.adapters_dir.iterdir()
            if p.is_dir() and (
                (p / "adapter_config.json").exists()
                or (p / "voice_card.json").exists()
            )
        )

    def voice_card(self, voice_name: str) -> VoiceCard:
        adapter_dir = self.adapters_dir / voice_name
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"voice '{voice_name}' not in library {self.adapters_dir}. "
                f"Available: {self.list_voices()}"
            )
        return VoiceCard(adapter_dir)

    def add_voice(self, src_adapter_dir: Path, voice_name: Optional[str] = None) -> Path:
        """Copy an externally-trained adapter into the library."""
        import shutil
        src_adapter_dir = Path(src_adapter_dir)
        if not src_adapter_dir.exists():
            raise FileNotFoundError(src_adapter_dir)
        name = voice_name or src_adapter_dir.name
        dst = self.adapters_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_adapter_dir, dst)
        return dst

    def remove_voice(self, voice_name: str) -> None:
        import shutil
        adapter_dir = self.adapters_dir / voice_name
        if not adapter_dir.exists():
            raise FileNotFoundError(voice_name)
        shutil.rmtree(adapter_dir)
        if self._loaded_adapter == voice_name:
            self._loaded_adapter = None
            # Reload the base model fresh to drop any peft state
            self._unload_adapter()

    # ------------------------------------------------------------------
    # Adapter swap
    # ------------------------------------------------------------------
    def _ensure_adapter(self, voice_name: str) -> None:
        """Load the requested adapter onto the base model. Idempotent —
        does nothing if it's already loaded.
        """
        if self._loaded_adapter == voice_name:
            return
        from peft import PeftModel  # type: ignore
        adapter_dir = self.adapters_dir / voice_name
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"voice '{voice_name}' not found. Available: {self.list_voices()}"
            )
        if self._peft_model is None:
            # First adapter load — wrap base in PeftModel
            self._peft_model = PeftModel.from_pretrained(
                self.base_model, str(adapter_dir),
                adapter_name=voice_name,
            )
            self._peft_model.eval()
        else:
            # Subsequent — load and switch
            try:
                self._peft_model.load_adapter(str(adapter_dir), adapter_name=voice_name)
            except ValueError:
                # Already loaded; just switch to it
                pass
            self._peft_model.set_adapter(voice_name)
        self._loaded_adapter = voice_name

    def _unload_adapter(self) -> None:
        """Drop the peft wrapper — return to base-only model."""
        self._peft_model = None
        self._loaded_adapter = None

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def _encode_description(self, description: str) -> tuple[torch.Tensor, torch.Tensor]:
        if description in self._desc_cache:
            return self._desc_cache[description]
        d = self.description_tokenizer(description, return_tensors="pt").to(self.device)
        result = (d.input_ids, d.attention_mask)
        if len(self._desc_cache) >= 16:
            self._desc_cache.pop(next(iter(self._desc_cache)))
        self._desc_cache[description] = result
        return result

    def _active_model(self):
        """Return whichever model the user wants to call generate() on —
        peft-wrapped if an adapter is loaded, else the bare base."""
        return self._peft_model if self._peft_model is not None else self.base_model

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        description: Optional[str] = None,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> tuple[np.ndarray, int]:
        """Render `text` with the given `voice` (if None, uses the base
        model with no speaker adaptation).

        `description` overrides the voice card's default description.
        """
        text = _basic_tts_clean(text)
        sample_rate = int(self.base_model.config.sampling_rate)

        if voice is not None:
            self._ensure_adapter(voice)
            card = self.voice_card(voice)
            description = description or card.description
            sample_rate = card.sample_rate
        else:
            self._unload_adapter()
            description = description or (
                "A Hindi speaker speaks clearly in a close-sounding "
                "recording with very clear audio and no background noise."
            )

        desc_ids, desc_mask = self._encode_description(description)
        prompt_inputs = self.prompt_tokenizer(text, return_tensors="pt").to(self.device)

        model = self._active_model()
        with torch.inference_mode():
            generation = model.generate(
                input_ids=desc_ids,
                attention_mask=desc_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                do_sample=do_sample,
                temperature=temperature,
            )
        audio = _safe_squeeze_audio(generation)
        return audio, sample_rate

    def speak_to_file(
        self,
        text: str,
        out_path: str | Path,
        *,
        voice: Optional[str] = None,
        description: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Path:
        audio, sr = self.synthesize(
            text, voice=voice, description=description, temperature=temperature,
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, sr, subtype="PCM_16")
        return out_path
