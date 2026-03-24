# gigaam-mlx

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1%2FM2%2FM3%2FM4-black?logo=apple)](https://github.com/ml-explore/mlx)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/aystream/GigaAM-v3-e2e-ctc-mlx)
[![arXiv](https://img.shields.io/badge/arXiv-2506.01192-b31b1b.svg)](https://arxiv.org/abs/2506.01192)

> Fast Russian speech recognition on Apple Silicon — **180x realtime**

MLX port of [GigaAM-v3](https://github.com/salute-developers/GigaAM) (220M params, Conformer + CTC) by Salute Developers. Produces **punctuated, normalized text** directly. No PyTorch required.

<p align="center">
  <img src="assets/benchmark.svg" alt="Benchmark comparison" width="600">
</p>

## Quick Start

```bash
pip install gigaam-mlx
```

```python
from gigaam_mlx import load_model, transcribe

model, tokenizer = load_model()  # auto-downloads from HuggingFace
text = transcribe(model, tokenizer, "meeting.wav")
print(text)
```

## CLI

```bash
# Transcribe any audio/video file
gigaam-mlx recording.mkv

# Output subtitles
gigaam-mlx call.wav --output-dir ./transcripts --format srt
```

Outputs `.srt` (subtitles) and `.txt` (plain text). Model weights download automatically on first run.

## Performance

MacBook Pro M2 Max, 20-second audio chunk (avg of 3 runs, warmed up):

| Backend | Model | Time | Realtime factor |
|---|---|---|---|
| **MLX (this)** | **v3_e2e_ctc** | **0.11s** | **~180x** |
| PyTorch MPS | v3_e2e_rnnt | 0.76s | ~26x |
| PyTorch CPU | v3_e2e_rnnt | 1.13s | ~18x |
| ONNX CPU | v3_e2e_ctc | 1.66s | ~12x |

Full 18-minute video: **21.5 seconds** end-to-end (~50x realtime).

## Features

- **180x realtime** on Apple Silicon (M1/M2/M3/M4)
- **Russian + English** — recognizes English words/terms in Russian speech
- **Punctuation** built-in — end-to-end model, no post-processing
- **No PyTorch** — pure MLX + librosa + numpy
- **Any format** — video and audio via ffmpeg (mkv, mp4, wav, mp3, ...)
- **Auto-download** — model weights from HuggingFace Hub

## Requirements

- macOS with Apple Silicon (M1+)
- Python >= 3.10
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`)

## How it works

<p align="center">
  <img src="https://raw.githubusercontent.com/salute-developers/GigaAM/main/assets/gigaam_scheme.svg" alt="GigaAM architecture" width="700">
  <br>
  <em>GigaAM model family (<a href="https://github.com/salute-developers/GigaAM">source</a>)</em>
</p>

```
Audio/Video → ffmpeg (16kHz mono) → Mel spectrogram (librosa)
    → Conformer encoder (16 layers, 768d, 16 heads, RoPE)
    → CTC head → greedy decode → punctuated text
```

The model is a 220M parameter Conformer pretrained on 700,000 hours of Russian speech. The `v3_e2e_ctc` variant produces punctuated, normalized text directly — no language model or post-processing needed.

## Converting weights yourself

```bash
pip install gigaam-mlx[convert]
python -m gigaam_mlx.convert --model v3_e2e_ctc --output-dir ./weights
```

## Acknowledgments

- [GigaAM](https://github.com/salute-developers/GigaAM) by Salute Developers / SberDevices — original model ([paper](https://arxiv.org/abs/2506.01192), InterSpeech 2025)
- [MLX](https://github.com/ml-explore/mlx) by Apple — ML framework for Apple Silicon
- [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3) — HuggingFace transformers integration

## License

MIT — same as the original GigaAM model.
