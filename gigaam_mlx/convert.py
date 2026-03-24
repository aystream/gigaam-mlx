"""Convert GigaAM PyTorch weights to MLX safetensors format.

Requires: pip install gigaam-mlx[convert]
"""

import argparse
import os
import shutil

import numpy as np


def _conv1d_weights(w) -> np.ndarray:
    """PyTorch Conv1d (C_out, C_in, K) → MLX Conv1d (C_out, K, C_in)."""
    return w.permute(0, 2, 1).detach().cpu().numpy()


def convert_encoder(pt_state: dict) -> dict:
    """Convert Conformer encoder weights."""
    weights = {}

    # Pre-encode (2x Conv1d subsampling)
    for i, conv_idx in enumerate([0, 2]):
        pt_pfx = f"encoder.pre_encode.conv.{conv_idx}"
        mlx_pfx = f"encoder.pre_encode.conv{i + 1}"
        weights[f"{mlx_pfx}.weight"] = _conv1d_weights(pt_state[f"{pt_pfx}.weight"])
        weights[f"{mlx_pfx}.bias"] = pt_state[f"{pt_pfx}.bias"].detach().cpu().numpy()

    # 16 Conformer layers
    for layer_idx in range(16):
        pt = f"encoder.layers.{layer_idx}"
        mlx = f"encoder.layers.{layer_idx}"

        # LayerNorms
        for name in [
            "norm_feed_forward1", "norm_conv", "norm_self_att",
            "norm_feed_forward2", "norm_out",
        ]:
            for p in ["weight", "bias"]:
                weights[f"{mlx}.{name}.{p}"] = (
                    pt_state[f"{pt}.{name}.{p}"].detach().cpu().numpy()
                )

        # Feed-forwards
        for ff in ["feed_forward1", "feed_forward2"]:
            for lin in ["linear1", "linear2"]:
                for p in ["weight", "bias"]:
                    weights[f"{mlx}.{ff}.{lin}.{p}"] = (
                        pt_state[f"{pt}.{ff}.{lin}.{p}"].detach().cpu().numpy()
                    )

        # Self-attention
        for lin in ["linear_q", "linear_k", "linear_v", "linear_out"]:
            for p in ["weight", "bias"]:
                weights[f"{mlx}.self_attn.{lin}.{p}"] = (
                    pt_state[f"{pt}.self_attn.{lin}.{p}"].detach().cpu().numpy()
                )

        # Convolution module
        for conv_name in ["pointwise_conv1", "pointwise_conv2"]:
            weights[f"{mlx}.conv.{conv_name}.weight"] = _conv1d_weights(
                pt_state[f"{pt}.conv.{conv_name}.weight"]
            )
            weights[f"{mlx}.conv.{conv_name}.bias"] = (
                pt_state[f"{pt}.conv.{conv_name}.bias"].detach().cpu().numpy()
            )

        # Depthwise conv (groups=d_model)
        weights[f"{mlx}.conv.depthwise_conv.weight"] = _conv1d_weights(
            pt_state[f"{pt}.conv.depthwise_conv.weight"]
        )
        weights[f"{mlx}.conv.depthwise_conv.bias"] = (
            pt_state[f"{pt}.conv.depthwise_conv.bias"].detach().cpu().numpy()
        )

        # BatchNorm (LayerNorm in v3)
        weights[f"{mlx}.conv.batch_norm.weight"] = (
            pt_state[f"{pt}.conv.batch_norm.weight"].detach().cpu().numpy()
        )
        weights[f"{mlx}.conv.batch_norm.bias"] = (
            pt_state[f"{pt}.conv.batch_norm.bias"].detach().cpu().numpy()
        )

    return weights


def convert_ctc_head(pt_state: dict) -> dict:
    """Convert CTC head weights."""
    return {
        "head.decoder_layers.weight": _conv1d_weights(
            pt_state["head.decoder_layers.0.weight"]
        ),
        "head.decoder_layers.bias": (
            pt_state["head.decoder_layers.0.bias"].detach().cpu().numpy()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert GigaAM PyTorch → MLX")
    parser.add_argument(
        "--model", default="v3_e2e_ctc",
        help="GigaAM model name (default: v3_e2e_ctc)",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Output directory for weights and tokenizer",
    )
    args = parser.parse_args()

    # Lazy imports — only needed for conversion
    import ssl
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )

    import mlx.core as mx
    import gigaam

    print(f"Loading PyTorch GigaAM {args.model}...")
    pt_model = gigaam.load_model(args.model)
    pt_state = {k: v for k, v in pt_model.named_parameters()}
    for k, v in pt_model.named_buffers():
        pt_state[k] = v

    print("Converting encoder weights...")
    weights = convert_encoder(pt_state)

    print("Converting CTC head weights...")
    weights.update(convert_ctc_head(pt_state))

    mlx_weights = {k: mx.array(v) for k, v in weights.items()}

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "weights.safetensors")
    print(f"Saving to {out_path}...")
    mx.save_safetensors(out_path, mlx_weights)

    tokenizer_src = os.path.expanduser(
        f"~/.cache/gigaam/{args.model}_tokenizer.model"
    )
    tokenizer_dst = os.path.join(args.output_dir, "tokenizer.model")
    if os.path.exists(tokenizer_src):
        shutil.copy2(tokenizer_src, tokenizer_dst)
        print(f"Copied tokenizer: {tokenizer_dst}")

    total = sum(v.size for v in mlx_weights.values())
    print(f"Done! {len(mlx_weights)} tensors, {total:,} parameters")


if __name__ == "__main__":
    main()
