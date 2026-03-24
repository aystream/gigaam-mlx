"""GigaAM-MLX: Fast Russian speech recognition on Apple Silicon."""

from .model import GigaAMMLX
from .audio import load_audio, compute_mel
from .transcribe import transcribe_file

__version__ = "0.1.0"

DEFAULT_REPO = "aystream/GigaAM-v3-e2e-ctc-mlx"


def load_model(repo_id: str = DEFAULT_REPO, local_dir: str | None = None):
    """
    Load GigaAM MLX model and tokenizer.

    Args:
        repo_id: HuggingFace repo ID or local path
        local_dir: Optional local directory to cache model files

    Returns:
        tuple: (model, tokenizer)
    """
    import os
    import mlx.core as mx
    from sentencepiece import SentencePieceProcessor

    # Check if repo_id is a local path
    if os.path.isdir(repo_id):
        model_dir = repo_id
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id, local_dir=local_dir)

    weights_path = os.path.join(model_dir, "weights.safetensors")
    tokenizer_path = os.path.join(model_dir, "tokenizer.model")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    model = GigaAMMLX()
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
    token_ids = model.ctc_decode(encoded, seq_len)
    return tokenizer.decode(token_ids)
