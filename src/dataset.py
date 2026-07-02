"""
Dataset loading for the Speech Commands keyword-spotting task.

Design note: audio loading is injected as a callable rather than hardcoded to
torchaudio.load, so tests can run with fake in-memory waveforms instead of
real files on disk (same pattern I used in mini-pki / cert-transparency-monitor).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch.utils.data import Dataset

# Signature: (file_path: Path) -> (waveform: torch.Tensor, sample_rate: int)
AudioLoader = Callable[[Path], tuple[torch.Tensor, int]]

# Speech Commands clips are ~1s at 16kHz. Not all clips are exactly this
# length, so every waveform gets padded/truncated to this before batching.
TARGET_LENGTH_SAMPLES = 16000


def pad_or_truncate(waveform: torch.Tensor, target_length: int = TARGET_LENGTH_SAMPLES) -> torch.Tensor:
    """Makes waveform exactly target_length samples long, on the last dim.

    - Shorter clips: right-padded with zeros (silence).
    - Longer clips: truncated from the end.

    Chose simple start-aligned pad/truncate over center-padding: it's the
    more common approach in keyword-spotting literature and keeps the logic
    trivial to reason about. Revisit if accuracy suffers on short words.
    """
    current_length = waveform.shape[-1]

    if current_length == target_length:
        return waveform

    if current_length < target_length:
        padding = target_length - current_length
        return torch.nn.functional.pad(waveform, (0, padding))

    return waveform[..., :target_length]


def default_audio_loader(path: Path) -> tuple[torch.Tensor, int]:
    """Real loader used in production. Kept thin on purpose so tests
    never have to touch torchaudio's backend."""
    import torchaudio  # local import: only needed for the real path

    waveform, sample_rate = torchaudio.load(str(path))
    return waveform, sample_rate


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


def load_manifest(manifest_path: Path) -> list[Sample]:
    """Reads a CSV with columns: filepath,label

    NOTE: this expects a manifest file we haven't generated yet — next session
    is writing the script that walks the Speech Commands folder structure
    (one subfolder per word) and produces this CSV. For now this just defines
    the contract so the dataset class has something to depend on.
    """
    samples: list[Sample] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(Sample(path=Path(row["filepath"]), label=row["label"]))
    return samples


class KeywordSpottingDataset(Dataset):
    """PyTorch Dataset over (waveform, label_index) pairs.

    audio_loader is injected so unit tests don't need real .wav files.
    """

    def __init__(
        self,
        samples: Iterable[Sample],
        label_to_index: dict[str, int],
        audio_loader: AudioLoader = default_audio_loader,
    ) -> None:
        self.samples = list(samples)
        self.label_to_index = label_to_index
        self.audio_loader = audio_loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        waveform, sample_rate = self.audio_loader(sample.path)
        waveform = pad_or_truncate(waveform)

        label_index = self.label_to_index[sample.label]
        return waveform, label_index
