"""GigaAM v3 e2e — Conformer encoder + CTC/RNNT head on Apple MLX."""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ── Rotary Positional Encoding ──────────────────────────────────

def create_rotary_pe(
    length: int, dim: int, base: int = 5000
) -> Tuple[mx.array, mx.array]:
    """Create rotary positional embeddings (cos, sin)."""
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(length, dtype=mx.float32)
    freqs = mx.outer(t, inv_freq)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def _rotate_half(x: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def _apply_rotary(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    """Apply RoPE to q, k. Input shape: (T, B, H, D)."""
    T = q.shape[0]
    cos = cos[:T, None, None, :]
    sin = sin[:T, None, None, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# ── Conformer Building Blocks ───────────────────────────────────

class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.silu(self.linear1(x)))


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=padding, groups=d_model,
        )
        self.batch_norm = nn.LayerNorm(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pointwise_conv1(x)
        a, b = mx.split(x, 2, axis=-1)
        x = a * mx.sigmoid(b)  # GLU
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = nn.silu(x)
        return self.pointwise_conv2(x)


class RotaryMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int):
        super().__init__()
        self.h = n_head
        self.d_k = n_feat // n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

    def __call__(
        self, query: mx.array, key: mx.array, value: mx.array,
        cos: mx.array, sin: mx.array,
    ) -> mx.array:
        B, T, D = query.shape

        # Apply RoPE to raw input before linear projections
        q_raw = mx.transpose(query.reshape(B, T, self.h, self.d_k), (1, 0, 2, 3))
        k_raw = mx.transpose(key.reshape(B, T, self.h, self.d_k), (1, 0, 2, 3))
        v_raw = mx.transpose(value.reshape(B, T, self.h, self.d_k), (1, 0, 2, 3))
        q_raw, k_raw = _apply_rotary(q_raw, k_raw, cos, sin)
        query = mx.transpose(q_raw, (1, 0, 2, 3)).reshape(B, T, D)
        key = mx.transpose(k_raw, (1, 0, 2, 3)).reshape(B, T, D)
        value = mx.transpose(v_raw, (1, 0, 2, 3)).reshape(B, T, D)

        # Project and compute attention
        q = mx.transpose(self.linear_q(query).reshape(B, T, self.h, self.d_k), (0, 2, 1, 3))
        k = mx.transpose(self.linear_k(key).reshape(B, T, self.h, self.d_k), (0, 2, 1, 3))
        v = mx.transpose(self.linear_v(value).reshape(B, T, self.h, self.d_k), (0, 2, 1, 3))

        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.d_k)
        out = mx.softmax(scores, axis=-1) @ v

        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, T, self.h * self.d_k)
        return self.linear_out(out)


# ── Conformer Layer & Encoder ───────────────────────────────────

class ConformerLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, conv_kernel_size: int):
        super().__init__()
        self.fc_factor = 0.5
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model, d_ff)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model, conv_kernel_size)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RotaryMultiHeadAttention(n_heads, d_model)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model, d_ff)
        self.norm_out = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        residual = x
        x = self.feed_forward1(self.norm_feed_forward1(x))
        residual = residual + x * self.fc_factor

        x = self.self_attn(
            self.norm_self_att(residual), self.norm_self_att(residual),
            self.norm_self_att(residual), cos, sin,
        )
        residual = residual + x

        x = self.conv(self.norm_conv(residual))
        residual = residual + x

        x = self.feed_forward2(self.norm_feed_forward2(residual))
        residual = residual + x * self.fc_factor

        return self.norm_out(residual)


class Conv1dSubsampling(nn.Module):
    """2x Conv1d with stride 2 each → 4x subsampling."""

    def __init__(self, feat_in: int, feat_out: int, kernel_size: int = 5):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(feat_in, feat_out, kernel_size=kernel_size, stride=2, padding=padding)
        self.conv2 = nn.Conv1d(feat_out, feat_out, kernel_size=kernel_size, stride=2, padding=padding)

    def __call__(self, x: mx.array) -> Tuple[mx.array, int]:
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        return x, x.shape[1]


class ConformerEncoder(nn.Module):
    def __init__(
        self, feat_in: int = 64, n_layers: int = 16, d_model: int = 768,
        n_heads: int = 16, ff_expansion_factor: int = 4,
        conv_kernel_size: int = 5, subs_kernel_size: int = 5,
    ):
        super().__init__()
        self.pre_encode = Conv1dSubsampling(feat_in, d_model, subs_kernel_size)
        self.layers = [
            ConformerLayer(d_model, d_model * ff_expansion_factor, n_heads, conv_kernel_size)
            for _ in range(n_layers)
        ]
        self.rope_dim = d_model // n_heads

    def __call__(self, features: mx.array) -> Tuple[mx.array, int]:
        x, seq_len = self.pre_encode(features)
        cos, sin = create_rotary_pe(seq_len, self.rope_dim)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return mx.transpose(x, (0, 2, 1)), seq_len


# ── CTC Head ────────────────────────────────────────────────────

class CTCHead(nn.Module):
    def __init__(self, feat_in: int = 768, num_classes: int = 257):
        super().__init__()
        self.decoder_layers = nn.Conv1d(feat_in, num_classes, kernel_size=1)

    def __call__(self, encoder_output: mx.array) -> mx.array:
        x = mx.transpose(encoder_output, (0, 2, 1))
        logits = self.decoder_layers(x)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


# ── RNNT Decoder & Joint ────────────────────────────────────────

class RNNTDecoder(nn.Module):
    def __init__(self, pred_hidden: int = 320, num_classes: int = 1025):
        super().__init__()
        self.pred_hidden = pred_hidden
        self.blank_id = num_classes - 1
        self.embed = nn.Embedding(num_classes, pred_hidden)
        self.lstm = nn.LSTM(pred_hidden, pred_hidden)

    def predict(
        self, x: Optional[mx.array], state: Optional[Tuple[mx.array, mx.array]]
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        if x is not None:
            emb = self.embed(x)
        else:
            emb = mx.zeros((1, 1, self.pred_hidden))
        if state is not None:
            h, c = state
            all_hidden, all_cell = self.lstm(emb, h, c)
        else:
            all_hidden, all_cell = self.lstm(emb)
        return all_hidden, (all_hidden[:, -1, :], all_cell[:, -1, :])


class RNNTJoint(nn.Module):
    def __init__(
        self, enc_hidden: int = 768, pred_hidden: int = 320,
        joint_hidden: int = 320, num_classes: int = 1025,
    ):
        super().__init__()
        self.enc_proj = nn.Linear(enc_hidden, joint_hidden)
        self.pred_proj = nn.Linear(pred_hidden, joint_hidden)
        self.out = nn.Linear(joint_hidden, num_classes)

    def __call__(self, enc: mx.array, pred: mx.array) -> mx.array:
        e = mx.expand_dims(self.enc_proj(enc), axis=2)
        p = mx.expand_dims(self.pred_proj(pred), axis=1)
        joint = nn.relu(e + p)
        logits = self.out(joint)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


# ── Full Model ──────────────────────────────────────────────────

CTC_CLASSES = 257
RNNT_CLASSES = 1025


class GigaAMMLX(nn.Module):
    """
    GigaAM v3 e2e on Apple MLX.

    220M parameter Conformer encoder for Russian ASR.
    Supports CTC (fast) and RNNT (higher quality) decoding.
    """

    def __init__(self, model_type: str = "ctc"):
        super().__init__()
        self.model_type = model_type
        self.encoder = ConformerEncoder()

        if model_type == "ctc":
            self.num_classes = CTC_CLASSES
            self.head = CTCHead(num_classes=CTC_CLASSES)
        elif model_type == "rnnt":
            self.num_classes = RNNT_CLASSES
            self.decoder = RNNTDecoder(pred_hidden=320, num_classes=RNNT_CLASSES)
            self.joint = RNNTJoint(
                enc_hidden=768, pred_hidden=320,
                joint_hidden=320, num_classes=RNNT_CLASSES,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'ctc' or 'rnnt'.")

    def encode(self, features: mx.array) -> Tuple[mx.array, int]:
        """Run conformer encoder. Input: (B, T, 64) mel spectrogram."""
        return self.encoder(features)

    def decode(self, encoded: mx.array, seq_len: int) -> List[int]:
        """Decode using the model's head (CTC or RNNT)."""
        if self.model_type == "ctc":
            return self._ctc_decode(encoded, seq_len)
        return self._rnnt_decode(encoded, seq_len)

    def _ctc_decode(self, encoded: mx.array, seq_len: int) -> List[int]:
        """CTC greedy decoding — fully vectorized."""
        log_probs = self.head(encoded)
        labels = mx.argmax(log_probs[0, :seq_len, :], axis=-1)
        mx.eval(labels)

        blank_id = self.num_classes - 1
        token_ids = []
        prev = blank_id
        for tok in labels.tolist():
            if tok != blank_id and tok != prev:
                token_ids.append(tok)
            prev = tok
        return token_ids

    def _rnnt_decode(
        self, encoded: mx.array, seq_len: int, max_symbols: int = 10
    ) -> List[int]:
        """RNNT greedy decoding (sequential)."""
        enc = encoded[0]  # (C, T)
        blank_id = self.decoder.blank_id
        hyp: List[int] = []
        state: Optional[Tuple[mx.array, mx.array]] = None
        last_label: Optional[mx.array] = None

        for t in range(seq_len):
            f = enc[:, t:t + 1].T
            f = mx.expand_dims(f, axis=0) if f.ndim == 2 else f
            not_blank = True
            symbols = 0
            while not_blank and symbols < max_symbols:
                g, new_state = self.decoder.predict(last_label, state)
                logits = self.joint(f, g)
                k = mx.argmax(logits[0, 0, 0, :]).item()
                if k == blank_id:
                    not_blank = False
                else:
                    hyp.append(int(k))
                    state = new_state
                    last_label = mx.array([[hyp[-1]]])
                    symbols += 1
        return hyp

    # Keep backward compat
    def ctc_decode(self, encoded: mx.array, seq_len: int) -> List[int]:
        return self._ctc_decode(encoded, seq_len)
