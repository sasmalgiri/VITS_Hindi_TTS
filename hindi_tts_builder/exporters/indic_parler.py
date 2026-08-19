"""Indic-Parler-TTS dataset exporter.

Converts a project's pipe-CSV training set (built by `data.dataset`) into
a local Hugging Face dataset script that the official Parler-TTS training
script can `load_dataset()` directly.

Output layout under `<project>/exports/indic_parler/<profile>/`:

    train.jsonl
    validation.jsonl
    test.jsonl
    clips_44k/{train,validation,test}/<clip_id>.wav   (resampled to 44.1 kHz)
    indic_parler_dataset.py                           (HF dataset script)
    dataset_card.json                                 (export summary)
    rejections.jsonl                                  (audit trail of dropped rows)

Critical contract: Indic-Parler eats Devanagari raw text + a natural-language
description per clip. We always prefer `raw_text` over our Coqui-frontend's
`processed_text` because the latter may strip diacritics or romanize.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torchaudio.functional as AF


@dataclass(frozen=True)
class IndicParlerExportConfig:
    project_root: Path
    profile: str = "10h_clean"
    out_dir: Optional[Path] = None

    # Parler/Descript DAC path is 44.1 kHz oriented.
    target_sample_rate: int = 44100

    # Speaker identity controls (Indic-Parler conditions on description).
    speaker_name: str = "Divya"
    voice_description: str = (
        "Divya speaks Hindi in a clear, close-sounding narrator voice, "
        "with a natural storytelling pace, moderate expressiveness, balanced pitch, "
        "and very clear audio with no background noise."
    )

    min_duration_sec: float = 1.0
    max_duration_sec: float = 10.0

    # IMPORTANT: Indic-Parler wants Devanagari, not Coqui-romanized text.
    prefer_raw_text: bool = True

    overwrite: bool = False


_PUNCT_DOUBLE_DANDA = "।।"


def _clean_devanagari_text(text: str) -> str:
    text = text.replace("﻿", "")
    text = re.sub(r"\s+", " ", text).strip()
    # Subtitle artifacts that survive the SRT parser
    text = text.replace("♪", "").replace("♫", "")
    text = re.sub(r"<[^>]+>", "", text).strip()
    text = text.replace(_PUNCT_DOUBLE_DANDA, "।")
    return text


def _is_probably_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def _read_pipe_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        return [dict(r) for r in reader]


def _find_profile_csvs(project_root: Path, profile: str) -> dict[str, Path]:
    """Probe several layouts so this works against both legacy and current
    project structures.

    Recognised layouts (first match wins):
      training_set/{train,val,test}_<profile>.csv   (current — make-profile output)
      profiles/<profile>/{train,val,test}.csv
      dataset/<profile>/{train,val,test}.csv
      <profile>/{train,val,test}.csv
      training_set/{train,val,test}.csv             (no-profile fallback)
      {train,val,test}.csv                           (root fallback)
    """
    # Layout: per-profile CSVs as written by data.profiles.make_profile()
    ts = project_root / "training_set"
    suffixed_train = ts / f"train_{profile}.csv"
    if suffixed_train.exists():
        out = {"train": suffixed_train}
        for split, name in (("validation", f"val_{profile}.csv"), ("test", f"test_{profile}.csv")):
            p = ts / name
            if p.exists():
                out[split] = p
        if "validation" in out:
            return out

    # Layouts: profile-as-dir. We deliberately do NOT fall back to bare
    # `train.csv` at the project root or in training_set/, because that
    # would silently return the FULL unprofiled corpus when a user asked
    # for e.g. `60h_clean` — making the staged-comparison ladder useless.
    # Only paths that contain the profile name in some form are accepted.
    for base in (
        project_root / "profiles" / profile,
        project_root / "dataset" / profile,
        project_root / profile,
    ):
        train = base / "train.csv"
        val = base / "val.csv"
        if not val.exists():
            val = base / "validation.csv"
        test = base / "test.csv"
        if train.exists() and val.exists():
            out = {"train": train, "validation": val}
            if test.exists():
                out["test"] = test
            return out

    # Surface what profiles ARE available so the user doesn't have to grep.
    ts = project_root / "training_set"
    available = sorted(
        p.stem.replace("train_", "")
        for p in ts.glob("train_*.csv")
    ) if ts.exists() else []
    avail_msg = (
        f"Available profiles in this project: {available}\n  "
        if available else
        "No make-profile output detected.\n  "
    )
    raise FileNotFoundError(
        f"Could not find train/val CSVs for profile '{profile}' under {project_root}.\n  "
        f"{avail_msg}"
        f"To create one: hindi-tts-builder make-profile <project> {profile}\n  "
        "Or, if your CSVs live elsewhere, supported layouts are: "
        "training_set/{train,val,test}_<profile>.csv, profiles/<profile>/, "
        "dataset/<profile>/, <profile>/, or bare CSVs at project root."
    )


def _resolve_audio_path(csv_path: Path, project_root: Path, audio_path: str) -> Path:
    p = Path(audio_path)
    if p.is_absolute() and p.exists():
        return p

    candidates = [
        csv_path.parent / p,
        project_root / p,
        project_root / "processed" / p,
        project_root / "clips" / p,
        # Our pipeline writes clips under aligned/<source_id>/...
        project_root / "aligned" / p,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(f"Audio not found: {audio_path}")


def _read_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def _copy_or_resample_mono(src: Path, dst: Path, target_sr: int) -> float:
    dst.parent.mkdir(parents=True, exist_ok=True)
    data, sr = sf.read(str(src), always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        wav = torch.tensor(mono, dtype=torch.float32).unsqueeze(0)
        wav = AF.resample(wav, orig_freq=sr, new_freq=target_sr).squeeze(0).cpu().numpy()
    else:
        wav = mono
    sf.write(str(dst), wav, target_sr, subtype="PCM_16")
    return _read_duration_sec(dst)


def _safe_id(split: str, idx: int) -> str:
    return f"{split}_{idx:07d}"


def _style_description(base_description: str, text: str, duration: float) -> str:
    """Keep description mostly constant for speaker consistency, but lightly
    control pace for very short/long lines. Per the model card and the user's
    'most important rule' guidance: do not vary descriptions too much early
    on or speaker identity drifts.
    """
    word_count = max(1, len(text.split()))
    wps = word_count / max(duration, 0.1)
    if wps >= 3.2:
        pace = "slightly fast-paced"
    elif wps <= 1.4:
        pace = "slightly slow-paced"
    else:
        pace = "natural-paced"
    return (
        base_description
        + f" The delivery is {pace}, clean, close-mic, and suitable for Hindi story narration."
    )


class IndicParlerExporter:
    """Produces a HF-loadable dataset folder from one project profile."""

    def __init__(self, config: IndicParlerExportConfig):
        self.config = config
        self.project_root = config.project_root.resolve()
        self.out_dir = (
            config.out_dir
            or self.project_root / "exports" / "indic_parler" / config.profile
        ).resolve()

    def export(self) -> Path:
        if self.out_dir.exists():
            if self.config.overwrite:
                shutil.rmtree(self.out_dir)
            else:
                raise FileExistsError(
                    f"{self.out_dir} exists. Use overwrite=True or delete it."
                )

        self.out_dir.mkdir(parents=True, exist_ok=True)
        csvs = _find_profile_csvs(self.project_root, self.config.profile)

        summary = {
            "profile": self.config.profile,
            "project_root": str(self.project_root),
            "out_dir": str(self.out_dir),
            "target_sample_rate": self.config.target_sample_rate,
            "splits": {},
            "rejections": [],
        }

        for split, csv_path in csvs.items():
            rows = _read_pipe_csv(csv_path)
            accepted: list[dict] = []
            rejected: list[dict] = []

            for idx, row in enumerate(rows):
                try:
                    raw_text = (row.get("raw_text") or row.get("text") or "").strip()
                    processed_text = (row.get("processed_text") or "").strip()
                    text = raw_text if self.config.prefer_raw_text else (processed_text or raw_text)
                    text = _clean_devanagari_text(text)
                    if not text:
                        raise ValueError("empty_text")
                    if not _is_probably_devanagari(text):
                        raise ValueError(
                            "non_devanagari_text_for_indic_parler; use raw_text not Coqui romanized text"
                        )

                    src_audio = _resolve_audio_path(
                        csv_path=csv_path,
                        project_root=self.project_root,
                        audio_path=row["audio_path"],
                    )

                    duration = float(row.get("duration") or _read_duration_sec(src_audio))
                    if duration < self.config.min_duration_sec:
                        raise ValueError(f"too_short:{duration:.2f}")
                    if duration > self.config.max_duration_sec:
                        raise ValueError(f"too_long:{duration:.2f}")

                    clip_id = _safe_id(split, idx)
                    dst_audio = self.out_dir / "clips_44k" / split / f"{clip_id}.wav"
                    final_duration = _copy_or_resample_mono(
                        src_audio, dst_audio, self.config.target_sample_rate
                    )

                    accepted.append({
                        "id": clip_id,
                        "audio_filepath": str(dst_audio.resolve()),
                        "text": text,
                        "description": _style_description(
                            self.config.voice_description,
                            text=text,
                            duration=final_duration,
                        ),
                        "speaker": self.config.speaker_name,
                        "duration": round(final_duration, 4),
                        "source_id": row.get("source_id", ""),
                    })

                except Exception as e:
                    rejected.append({
                        "split": split,
                        "row_index": idx,
                        "reason": str(e),
                        "row": row,
                    })

            jsonl_path = self.out_dir / f"{split}.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as f:
                for rec in accepted:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            summary["splits"][split] = {
                "source_csv": str(csv_path),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "jsonl": str(jsonl_path),
            }
            summary["rejections"].extend(rejected)

        self._write_dataset_script()
        self._write_summary(summary)
        return self.out_dir

    def _write_summary(self, summary: dict) -> None:
        (self.out_dir / "dataset_card.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (self.out_dir / "rejections.jsonl").open("w", encoding="utf-8") as f:
            for r in summary["rejections"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _write_dataset_script(self) -> None:
        # IMPORTANT: bake the absolute path to the JSONL/wav files into the
        # generated script. HF `datasets` *copies* loaded scripts into a
        # cache directory before executing them, so `Path(__file__).parent`
        # resolves to the cache dir — not where our train.jsonl actually lives.
        # That gave "Unknown split 'train'. Should be one of []." Hard-coding
        # the absolute path of `self.out_dir` makes this resilient.
        export_root = str(self.out_dir.resolve())
        script = '''import json
from pathlib import Path

import datasets


# Absolute path baked at export time. Regenerate the script if you move
# the export folder.
_EXPORT_ROOT = Path(r"{export_root}")


class IndicParlerLocalDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="Local Hindi single-speaker dataset exported for Indic-Parler-TTS fine-tuning.",
            features=datasets.Features(
                {{
                    "id": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=44100),
                    "text": datasets.Value("string"),
                    "description": datasets.Value("string"),
                    "speaker": datasets.Value("string"),
                    "duration": datasets.Value("float32"),
                    "source_id": datasets.Value("string"),
                }}
            ),
        )

    def _split_generators(self, dl_manager):
        splits = []
        for name, jsonl, split_const in (
            ("train", "train.jsonl", datasets.Split.TRAIN),
            ("validation", "validation.jsonl", datasets.Split.VALIDATION),
            ("test", "test.jsonl", datasets.Split.TEST),
        ):
            p = _EXPORT_ROOT / jsonl
            if p.exists():
                splits.append(
                    datasets.SplitGenerator(
                        name=split_const,
                        gen_kwargs={{"path": str(p)}},
                    )
                )
        if not splits:
            raise FileNotFoundError(
                f"No JSONL files found under {{_EXPORT_ROOT}}. "
                "Did the export get moved? Re-run the exporter."
            )
        return splits

    def _generate_examples(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)
                key = row.get("id") or str(idx)
                yield key, {{
                    "id": key,
                    "audio": row["audio_filepath"],
                    "text": row["text"],
                    "description": row["description"],
                    "speaker": row.get("speaker", "narrator"),
                    "duration": float(row.get("duration", 0.0)),
                    "source_id": row.get("source_id", ""),
                }}
'''.format(export_root=export_root)
        (self.out_dir / "indic_parler_dataset.py").write_text(script, encoding="utf-8")
