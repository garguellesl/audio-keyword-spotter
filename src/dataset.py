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

        # TODO (next session): clips in Speech Commands are ~1s but not all
        # exactly the same number of samples, and I haven't decided yet
        # whether to pad or truncate. Leaving this unhandled for now — the
        # test below documents the gap instead of silently ignoring it.

        label_index = self.label_to_index[sample.label]
        return waveform, label_index
