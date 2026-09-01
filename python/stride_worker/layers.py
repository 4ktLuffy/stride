"""Transformer layers, written against the paged cache.

These are reference implementations in plain PyTorch. They are correct and
readable, and they are the baseline every Triton kernel in ``kernels/`` is
gated against — a kernel that disagrees with the code here is rejected no
matter how fast it is.

Only Llama-family geometry is covered: RMSNorm, rotary embeddings, grouped-query
attention and a gated SwiGLU MLP. That spans Llama, Mistral, Qwen2 and their
derivatives. Anything else should fail loudly at load rather than silently
produce wrong activations.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .cache import PagedKVCache


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square layer norm.

    Computed in float32 regardless of the input dtype. The sum of squares over
    a wide hidden dimension overflows BF16's range for perfectly ordinary
    activations, and the result is silently wrong rather than NaN.
    """
    dtype = x.dtype
    x32 = x.float()
    variance = x32.pow(2).mean(-1, keepdim=True)
    normed = x32 * torch.rsqrt(variance + eps)
    return (normed * weight.float()).to(dtype)


def build_rope_cache(
    head_dim: int,
    max_position: int,
    base: float,
    device: torch.device,
    scaling: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary cos/sin tables.

    Supports Llama-3 style frequency scaling, which stretches low frequencies
    to extend context while leaving high frequencies alone. Applying a plain
    linear scale instead is a common and quiet source of long-context quality
    loss.
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )

    if scaling and scaling.get("rope_type") in ("llama3",):
        factor = float(scaling.get("factor", 8.0))
        low_factor = float(scaling.get("low_freq_factor", 1.0))
        high_factor = float(scaling.get("high_freq_factor", 4.0))
        original = int(scaling.get("original_max_position_embeddings", 8192))

        wavelength = 2 * math.pi / inv_freq
        low_wavelength = original / low_factor
        high_wavelength = original / high_factor

        scaled = inv_freq / factor
        # Smooth interpolation between untouched and fully scaled frequencies.
        smooth = (original / wavelength - low_factor) / (high_factor - low_factor)
        smooth = smooth.clamp(0.0, 1.0)
        interpolated = (1 - smooth) * scaled + smooth * inv_freq

        inv_freq = torch.where(wavelength > low_wavelength, scaled, inv_freq)
        inv_freq = torch.where(
            (wavelength <= low_wavelength) & (wavelength >= high_wavelength),
            interpolated,
            inv_freq,
        )

    t = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k by their absolute positions.

    q is (n, num_q_heads, head_dim) and k is (n, num_kv_heads, head_dim), with
    `positions` giving each token's absolute index. Positions are passed
    explicitly rather than inferred, because a chunked prefill starts partway
    through a sequence and a resumed sequence starts partway through again.
    """
    c = cos[positions].unsqueeze(1).to(q.dtype)
    s = sin[positions].unsqueeze(1).to(q.dtype)
    return (q * c) + (_rotate_half(q) * s), (k * c) + (_rotate_half(k) * s)


def paged_attention(
    q: torch.Tensor,
    cache: PagedKVCache,
    layer: int,
    blocks: list[int],
    context_len: int,
    query_start: int,
    scale: float,
) -> torch.Tensor:
    """Attention over a sequence whose KV lives in scattered blocks.

    Gathers the sequence's blocks into contiguous tensors and calls PyTorch's
    fused attention. This is the correctness reference; it materialises the
    whole context, which the Triton kernel avoids by reading blocks in place.

    ``query_start`` is the absolute position of the first query token, which is
    what makes the causal mask correct for a chunked prefill: query token *i*
    may attend to keys up to ``query_start + i``, not to the whole context.
    """
    n_q, num_q_heads, head_dim = q.shape
    k, v = cache.gather(layer, blocks, context_len)

    num_kv_heads = k.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"{num_q_heads} query heads do not group into {num_kv_heads} KV heads")
    group = num_q_heads // num_kv_heads
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    # (heads, tokens, head_dim) for scaled_dot_product_attention.
    qh = q.transpose(0, 1)
    kh = k.transpose(0, 1)
    vh = v.transpose(0, 1)

    q_pos = torch.arange(query_start, query_start + n_q, device=q.device).unsqueeze(1)
    k_pos = torch.arange(context_len, device=q.device).unsqueeze(0)
    causal = (k_pos <= q_pos).unsqueeze(0)

    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=causal, scale=scale)
    return out.transpose(0, 1).contiguous()


def swiglu_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Gated feed-forward block: ``down(silu(gate(x)) * up(x))``."""
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    return F.linear(F.silu(gate) * up, down_weight)
