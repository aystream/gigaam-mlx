"""GigaAM-MLX: Fast Russian speech recognition on Apple Silicon."""

from .model import GigaAMMLX
from .audio import load_audio, compute_mel
from .transcribe import transcribe_file

__version__ = "0.1.0"

REPOS = {
    "ctc": "aystream/GigaAM-v3-e2e-ctc-mlx",
    "rnnt": "aystream/GigaAM-v3-e2e-rnnt-mlx",
}


def load_model(model_type: str = "ctc", repo_id: str | None = None):
    """
    Load GigaAM MLX model and tokenizer.

    Args:
        model_type: "ctc" (fast) or "rnnt" (higher quality)
        repo_id: HuggingFace repo ID or local path (auto-selected if None)

    Returns:
        tuple: (model, tokenizer)
    """
    import os
    import mlx.core as mx
    from sentencepiece import SentencePieceProcessor

    if model_type not in ("ctc", "rnnt"):
        raise ValueError(f"model_type must be 'ctc' or 'rnnt', got '{model_type}'")

    if repo_id is None:
        repo_id = REPOS[model_type]

    # Local path or HuggingFace download
    if os.path.isdir(repo_id):
        model_dir = repo_id
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id)

    # Try suffixed name first (local dev with both models), then standard (HF)
    weights_path = os.path.join(model_dir, f"weights_{model_type}.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, "weights.safetensors")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found in {model_dir}")

    tokenizer_path = os.path.join(model_dir, f"tokenizer_{model_type}.model")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(model_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found in {model_dir}")

    model = GigaAMMLX(model_type=model_type)
    weights = mx.load(weights_path)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    return model, tokenizer


def transcribe(model, tokenizer, audio_path: str) -> str:
    """
    Transcribe an audio or video file.

    Args:
        model: GigaAMMLX model instance
        tokenizer: SentencePiece tokenizer
        audio_path: Path to audio/video file (any format ffmpeg supports)

    Returns:
        Transcribed text
    """
    import mlx.core as mx
    import numpy as np

    audio = load_audio(audio_path)
    mel = compute_mel(audio)
    mel_mx = mx.array(mel[np.newaxis])

    encoded, seq_len = model.encode(mel_mx)
    mx.eval(encoded)
    token_ids = model.decode(encoded, seq_len)
    return tokenizer.decode(token_ids)
