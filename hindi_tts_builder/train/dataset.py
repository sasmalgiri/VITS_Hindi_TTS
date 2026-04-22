"""PyTorch Dataset + collate function for VITS training.

Reads train.csv / val.csv produced by `data.dataset`. Each row:
    audio_path | raw_text | processed_text | duration | source_id

At training time, each item yields:
    text_ids:     LongTensor [T_text]
    mel:          FloatTensor [n_mels, T_mel]
    audio:        FloatTensor [T_audio]     (needed by VITS for posterior encoder)
    text_length:  int
    mel_length:   int
    audio_length: int

Heavy imports (torch, torchaudio) are inside functions so `import
hindi_tts_builder` stays lightweight.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv


@dataclass
class DatasetRecord:
    audio_path: str
    raw_text: str
    processed_text: str
    duration: float


def read_split_csv(csv_path: Path) -> list[DatasetRecord]:
    """Read a split CSV (pipe-separated) produced by `data.dataset`."""
    records: list[DatasetRecord] = []
    with csv_path.open(encoding="utf-8") as f:
        r = csv.reader(f, delimiter="|")
        header = next(r, None)
        if header is None:
            return records
        col = {name: i for i, name in enumerate(header)}
        for row in r:
            if not row or len(row) < 4:
                continue
            records.append(DatasetRecord(
                audio_path=row[col["audio_path"]],
                raw_text=row[col["raw_text"]],
                processed_text=row[col["processed_text"]],
                duration=float(row[col["duration"]]),
            ))
    return records


class TTSDataset:
    """PyTorch Dataset. Constructed lazily so torch is only imported when used."""

    def __init__(
        self,
        project_root: Path,
        split: str,             # "train" / "val" / "test"
        tokenizer,              # HindiTokenizer
        *,
        sample_rate: int = 24000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_fmin: float = 0.0,
        mel_fmax: float = 12000.0,
        min_duration_sec: float = 1.0,
        max_duration_sec: float = 10.0,
    ):
        self.project_root = Path(project_root)
        self.split = split
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

        csv_path = self.project_root / "training_set" / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        all_records = read_split_csv(csv_path)
        self.records = [
            r for r in all_records
            if min_duration_sec <= r.duration <= max_duration_sec
        ]
        if not self.records:
            raise RuntimeError(
                f"No usable records in {csv_path} "
                f"(after filtering {min_duration_sec}..{max_duration_sec}s)"
            )

        # Cache the mel transform lazily
        self._mel_transform = None

    def __len__(self) -> int:
        return len(self.records)

    def _ensure_mel_transform(self):
        if self._mel_transform is not None:
            return
        import torchaudio  # type: ignore
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            n_mels=self.n_mels,
            power=1.0,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )

    def __getitem__(self, idx: int):
        import torch  # type: ignore
        import torchaudio  # type: ignore
        self._ensure_mel_transform()

        rec = self.records[idx]
        audio_path = self.project_root / rec.audio_path
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.squeeze(0)

        # Compute mel on log-scale (standard for vocoder training)
        mel = self._mel_transform(waveform.unsqueeze(0)).squeeze(0)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        text_ids = torch.tensor(
            self.tokenizer.encode(rec.processed_text), dtype=torch.long
        )

        return {
            "text_ids": text_ids,
            "mel": mel,                      # [n_mels, T_mel]
            "audio": waveform,               # [T_audio]
            "text_length": int(text_ids.shape[0]),
            "mel_length": int(mel.shape[1]),
            "audio_length": int(waveform.shape[0]),
            "text": rec.processed_text,
            "audio_path": rec.audio_path,
        }


def collate_batch(batch: list[dict], pad_id: int = 0):
    """Pad variable-length items to a uniform batch tensor."""
    import torch  # type: ignore

    B = len(batch)
    text_lengths = torch.tensor([b["text_length"] for b in batch], dtype=torch.long)
    mel_lengths = torch.tensor([b["mel_length"] for b in batch], dtype=torch.long)
    audio_lengths = torch.tensor([b["audio_length"] for b in batch], dtype=torch.long)

    T_text = int(text_lengths.max().item())
    T_mel = int(mel_lengths.max().item())
    T_audio = int(audio_lengths.max().item())
    n_mels = batch[0]["mel"].shape[0]

    text_padded = torch.full((B, T_text), pad_id, dtype=torch.long)
    mel_padded = torch.zeros((B, n_mels, T_mel), dtype=torch.float32)
    audio_padded = torch.zeros((B, T_audio), dtype=torch.float32)

    for i, b in enumerate(batch):
        tl = b["text_length"]
        ml = b["mel_length"]
        al = b["audio_length"]
        text_padded[i, :tl] = b["text_ids"]
        mel_padded[i, :, :ml] = b["mel"]
        audio_padded[i, :al] = b["audio"]

    return {
        "text": text_padded,
        "text_lengths": text_lengths,
        "mel": mel_padded,
        "mel_lengths": mel_lengths,
        "audio": audio_padded.unsqueeze(1),   # [B, 1, T] for VITS
        "audio_lengths": audio_lengths,
    }
