from pathlib import Path

from src.build_manifest import find_labeled_clips, write_manifest


def _make_fake_clip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")  # content doesn't matter, we're only testing paths


def test_find_labeled_clips_groups_by_folder_name(tmp_path):
    _make_fake_clip(tmp_path / "yes" / "clip1.wav")
    _make_fake_clip(tmp_path / "yes" / "clip2.wav")
    _make_fake_clip(tmp_path / "no" / "clip1.wav")

    clips = find_labeled_clips(tmp_path)

    labels = sorted(label for _, label in clips)
    assert labels == ["no", "yes", "yes"]
    assert len(clips) == 3


def test_find_labeled_clips_ignores_background_noise_folder(tmp_path):
    _make_fake_clip(tmp_path / "yes" / "clip1.wav")
    _make_fake_clip(tmp_path / "_background_noise_" / "pink_noise.wav")

    clips = find_labeled_clips(tmp_path)

    assert len(clips) == 1
    assert clips[0][1] == "yes"


def test_find_labeled_clips_ignores_non_wav_files(tmp_path):
    _make_fake_clip(tmp_path / "yes" / "clip1.wav")
    _make_fake_clip(tmp_path / "yes" / "README.txt")

    clips = find_labeled_clips(tmp_path)

    assert len(clips) == 1


def test_write_manifest_produces_readable_csv(tmp_path):
    clips = [
        (tmp_path / "yes" / "clip1.wav", "yes"),
        (tmp_path / "no" / "clip1.wav", "no"),
    ]
    output = tmp_path / "manifest.csv"

    write_manifest(clips, output)

    content = output.read_text()
    assert "filepath,label" in content
    assert "yes" in content
    assert "no" in content
    # round-trip through our own loader to make sure the format actually matches
    from src.dataset import load_manifest
    samples = load_manifest(output)
    assert len(samples) == 2
