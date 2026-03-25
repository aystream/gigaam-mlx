"""CLI and API for transcribing audio/video files with GigaAM MLX."""

import argparse
import os
import time
from typing import Optional

import mlx.core as mx
import numpy as np

from .audio import compute_mel, load_audio, split_audio
from .model import GigaAMMLX


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(
                f"{format_srt_time(seg['start'])} --> "
                f"{format_srt_time(seg['end'])}\n"
            )
            f.write(f"{seg['text'].strip()}\n\n")


def transcribe_file(
    audio_path: str,
    model: Optional[GigaAMMLX] = None,
    tokenizer=None,
    model_type: str = "ctc",
    repo_id: Optional[str] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Transcribe an audio or video file.

    Args:
        audio_path: Path to audio/video file
        model: Pre-loaded model (loads from HF if None)
        tokenizer: Pre-loaded tokenizer
        model_type: "ctc" (fast) or "rnnt" (higher quality)
        repo_id: HuggingFace repo ID (auto-selected if None)
        verbose: Print progress

    Returns:
        List of segments with 'start', 'end', 'text' keys
    """
    def log(msg):
        if verbose:
            print(msg, flush=True)

    if model is None or tokenizer is None:
        from . import load_model
        model, tokenizer = load_model(model_type=model_type, repo_id=repo_id)

    log(f"Loading audio: {os.path.basename(audio_path)}")
    audio = load_audio(audio_path)
    log(f"Audio: {len(audio) / 16000:.1f}s")

    chunks = split_audio(audio)
    log(f"Split into {len(chunks)} chunks")

    t0 = time.time()
    segments = []
    for i, chunk in enumerate(chunks):
        chunk_audio = audio[chunk["start_sample"]:chunk["end_sample"]]
        mel = compute_mel(chunk_audio)
        mel_mx = mx.array(mel[np.newaxis])

        encoded, seq_len = model.encode(mel_mx)
        mx.eval(encoded)
        token_ids = model.decode(encoded, seq_len)
        text = tokenizer.decode(token_ids)

        if text.strip():
            seg = {
                "start": chunk["start_sec"],
                "end": chunk["end_sec"],
                "text": text,
            }
            segments.append(seg)
            log(
                f"  [{format_srt_time(seg['start'])} -> "
                f"{format_srt_time(seg['end'])}] {text}"
            )

        if verbose and (i + 1) % 10 == 0:
            log(f"  ... {i + 1}/{len(chunks)} chunks")

    elapsed = time.time() - t0
    log(f"Transcribed in {elapsed:.1f}s ({len(segments)} segments)")
    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video with GigaAM MLX"
    )
    parser.add_argument("input", help="Path to audio or video file")
    parser.add_argument(
        "--model-type", default="ctc", choices=["ctc", "rnnt"],
        help="Model variant: ctc (fast) or rnnt (higher quality)",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--model", default=None, help="HF repo ID or local model path")
    parser.add_argument("--format", choices=["srt", "txt", "both"], default="both")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return

    output_dir = args.output_dir or os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    segments = transcribe_file(
        input_path,
        model_type=args.model_type,
        repo_id=args.model,
        verbose=not args.quiet,
    )

    if not segments:
        print("No speech detected.")
        return

    if args.format in ("srt", "both"):
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        write_srt(segments, srt_path)
        print(f"Saved: {srt_path}")

    if args.format in ("txt", "both"):
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(s["text"].strip() for s in segments))
        print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
