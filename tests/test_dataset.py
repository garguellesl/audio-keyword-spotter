from pathlib import Path

import torch

from src.dataset import KeywordSpottingDataset, Sample, load_manifest


def fake_loader_factory(waveform: torch.Tensor, sample_rate: int = 16000):
    """Returns an audio_loader that ignores the path and always returns
    the same fake waveform. Lets us test dataset logic without real audio."""

    def _loader(path: Path) -> tuple[torch.Tensor, int]:
        return waveform, sample_rate

    return _loader


def test_dataset_length_matches_sample_count():
    samples = [
        Sample(path=Path("fake1.wav"), label="yes"),
        Sample(path=Path("fake2.wav"), label="no"),
    ]
    loader = fake_loader_factory(torch.zeros(1, 16000))
    ds = KeywordSpottingDataset(
        samples, label_to_index={"yes": 0, "no": 1}, audio_loader=loader
    )
    assert len(ds) == 2


def test_dataset_returns_correct_label_index():
    samples = [Sample(path=Path("fake1.wav"), label="yes")]
    loader = fake_loader_factory(torch.zeros(1, 16000))
    ds = KeywordSpottingDataset(
        samples, label_to_index={"yes": 0, "no": 1}, audio_loader=loader
    )
    _, label_index = ds[0]
    assert label_index == 0


def test_dataset_pads_short_clips_to_fixed_length():
    """Speech Commands clips are ~1s @ 16kHz but not always exactly 16000
    samples. Every item returned by the dataset should have the same
    waveform length so batching works without a custom collate_fn."""
    short_clip = torch.zeros(1, 12000)  # shorter than the 16000 target
    samples = [Sample(path=Path("fake_short.wav"), label="yes")]
    loader = fake_loader_factory(short_clip)
    ds = KeywordSpottingDataset(
        samples, label_to_index={"yes": 0}, audio_loader=loader
    )
    waveform, _ = ds[0]
    assert waveform.shape[-1] == 16000


def test_dataset_truncates_long_clips_to_fixed_length():
    """Mirror case: a clip longer than 16000 samples (e.g. someone held the
    button a bit too long during recording) should be truncated, not error
    out or silently change the batch shape."""
    long_clip = torch.zeros(1, 20000)
    samples = [Sample(path=Path("fake_long.wav"), label="no")]
    loader = fake_loader_factory(long_clip)
    ds = KeywordSpottingDataset(
        samples, label_to_index={"no": 0}, audio_loader=loader
    )
    waveform, _ = ds[0]
    assert waveform.shape[-1] == 16000


def test_load_manifest_reads_csv(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filepath,label\nclips/yes/1.wav,yes\nclips/no/1.wav,no\n")

    samples = load_manifest(manifest)

    assert len(samples) == 2
    assert samples[0] == Sample(path=Path("clips/yes/1.wav"), label="yes")
    assert samples[1].label == "no"
