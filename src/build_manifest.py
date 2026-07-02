"""
Builds the manifest.csv that KeywordSpottingDataset consumes, by walking the
Speech Commands directory layout: one subfolder per label, containing .wav
files. e.g.:

    data/speech_commands/
        yes/0a2b3c.wav
        no/1f2e3d.wav
        _background_noise_/pink_noise.wav   <- special folder, not a label

Kept as a standalone script (not a method on the dataset class) because it's
a one-time offline step, not something that runs during training.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# This folder ships with the dataset for data augmentation (mixing noise
# into clips) but isn't itself a keyword label — skip it when scanning.
IGNORED_FOLDERS = {"_background_noise_"}


def find_labeled_clips(data_dir: Path) -> list[tuple[Path, str]]:
    """Walks data_dir and returns (file_path, label) for every .wav file,
    where label is the name of the immediate parent folder."""
    clips: list[tuple[Path, str]] = []

    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        if label_dir.name in IGNORED_FOLDERS:
            continue

        for wav_path in sorted(label_dir.glob("*.wav")):
            clips.append((wav_path, label_dir.name))

    return clips


def write_manifest(clips: list[tuple[Path, str]], output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        for path, label in clips:
            writer.writerow([str(path), label])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Path to the extracted Speech Commands folder",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifest.csv"),
        help="Where to write the manifest CSV (default: data/manifest.csv)",
    )
    args = parser.parse_args()

    clips = find_labeled_clips(args.data_dir)
    write_manifest(clips, args.output)
    print(f"Wrote {len(clips)} entries to {args.output}")

    # NOTE: this doesn't respect the official train/val/test split yet
    # (Speech Commands ships testing_list.txt / validation_list.txt for
    # that). Right now everything goes into one manifest — splitting is
    # next session's problem, not solving it here to keep this session scoped.


if __name__ == "__main__":
    main()
