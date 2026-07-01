# audio-keyword-spotter

Keyword spotting (wake-word style) classifier for short audio clips, built with PyTorch.

**Status: early development — data pipeline in progress, no trained model yet.**

## Goal
Classify short (~1s) audio clips into a small set of keywords vs. background noise,
using a lightweight CNN on mel-spectrograms. Target dataset: Google Speech Commands.

## Why this project
Built as part of my portfolio for audio/ML engineering roles (e.g. speech processing
teams working on wake-word detection, voice enhancement). Focus is on a clean,
testable pipeline rather than chasing state-of-the-art accuracy.

## Setup
```bash
pip install -r requirements.txt
```

## Project structure
```
src/
  dataset.py   # audio loading + preprocessing (WIP)
  model.py     # CNN architecture (not started yet)
  train.py     # training loop (not started yet)
tests/
  test_dataset.py
data/          # raw/processed audio (gitignored)
```

## Progress log
- [x] Repo scaffold + dependency injection design for dataset loading
- [ ] Dataset download + label parsing
- [ ] Mel-spectrogram feature extraction
- [ ] CNN model
- [ ] Training loop
- [ ] Evaluation + confusion matrix
