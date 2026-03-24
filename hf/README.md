---
library_name: mlx
license: mit
language:
  - ru
  - en
tags:
  - automatic-speech-recognition
  - mlx
  - apple-silicon
  - russian
  - gigaam
  - conformer
  - ctc
base_model: ai-sage/GigaAM-v3
pipeline_tag: automatic-speech-recognition
model-index:
  - name: GigaAM-v3-e2e-ctc-mlx
    results:
      - task:
          type: automatic-speech-recognition
        metrics:
          - name: RTF (M2 Max)
            type: rtf
            value: 0.006
---

# GigaAM v3 e2e CTC — MLX

MLX port of [GigaAM-v3](https://github.com/salute-developers/GigaAM) for fast Russian speech recognition on Apple Silicon. **180x realtime** on M2 Max.

## Usage

```bash
pip install gigaam-mlx
```

```python
from gigaam_mlx import load_model, transcribe

model, tokenizer = load_model()  # downloads weights automatically
text = transcribe(model, tokenizer, "recording.wav")
print(text)
```

Or via CLI:

```bash
gigaam-mlx recording.wav
```

## Performance

MacBook Pro M2 Max, 20-second chunk:

| Backend | Time | Realtime |
|---|---|---|
| **MLX CTC (this)** | **0.11s** | **180x** |
| PyTorch MPS RNNT | 0.76s | 26x |
| ONNX CPU CTC | 1.66s | 12x |

## Model

- **Architecture:** Conformer (16 layers, 768d, 16 heads, RoPE) + CTC
- **Parameters:** 220M
- **Vocabulary:** 257 tokens (SentencePiece)
- **Features:** Punctuation, text normalization, Russian + English code-switching

## Links

- **Code:** [github.com/aystream/gigaam-mlx](https://github.com/aystream/gigaam-mlx)
- **Original:** [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM) ([paper](https://arxiv.org/abs/2506.01192))
- **License:** MIT
