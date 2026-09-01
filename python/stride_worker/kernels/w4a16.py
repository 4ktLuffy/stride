"""W4A16 GEMM: 4-bit weights, 16-bit activations.

Weights are stored two per byte with a scale (and optionally a zero point) per
group of ``group_size`` elements along the reduction axis. Dequantising the
whole matrix up front would defeat the purpose — the point of 4-bit weights is
that fewer bytes cross the memory bus — so the dequantisation happens inside
the accumulation loop, on tiles already resident in shared memory.

The group metadata is not free. At ``group_size=128`` with FP16 scales and zero
points, the real cost is about 4.25 bits per weight, not 4. ``stride-model``
accounts for this in its capacity planning for exactly this reason.
"""

from __future__ import annotations

import torch

from . import require_cuda, triton_available


def pack_int4(weights: torch.Tensor) -> torch.Tensor:
    """Pack an int4 tensor (values 0..15) two per byte along the last axis."""
    if weights.shape[-1] % 2:
        raise ValueError("the packed axis must have an even length")
    low = weights[..., 0::2] & 0xF
    high = weights[..., 1::2] & 0xF
    return (low | (high << 4)).to(torch.uint8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_int4`, for the reference path and for tests."""
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    out = torch.stack((low, high), dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def dequantize_reference(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    group_size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference dequantisation: ``(q - zero) * scale``.

    This is the definition the Triton kernel is gated against.
    """
    q = unpack_int4(packed).to(torch.float32)
    k = q.shape[-1]
    groups = k // group_size
    q = q.reshape(*q.shape[:-1], groups, group_size)

    s = scales.to(torch.float32).unsqueeze(-1)
    if zeros is not None:
        q = q - zeros.to(torch.float32).unsqueeze(-1)
    return (q * s).reshape(*packed.shape[:-1], k).to(dtype)


if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _w4a16_gemm_kernel(
        a_ptr,          # [M, K] activations, fp16/bf16
        b_ptr,          # [K // 2, N] packed int4 weights, uint8
        c_ptr,          # [M, N] output
        scales_ptr,     # [K // group_size, N]
        zeros_ptr,      # [K // group_size, N]
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_sg, stride_sn,
        HAS_ZEROS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
            k_start = k0 * BLOCK_K
            k_idx = k_start + offs_k

            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (k_idx[None, :] < K),
                other=0.0,
            )

            # Two 4-bit weights share a byte; the low nibble is the even index.
            byte_idx = k_idx // 2
            packed = tl.load(
                b_ptr + byte_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(k_idx[:, None] < K) & (offs_n[None, :] < N),
                other=0,
            )
            nibble = tl.where((k_idx[:, None] % 2) == 0, packed & 0xF, (packed >> 4) & 0xF)
            q = nibble.to(tl.float32)

            group = k_idx // GROUP_SIZE
            scale = tl.load(
                scales_ptr + group[:, None] * stride_sg + offs_n[None, :] * stride_sn,
                mask=(k_idx[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            ).to(tl.float32)

            if HAS_ZEROS:
                zero = tl.load(
                    zeros_ptr + group[:, None] * stride_sg + offs_n[None, :] * stride_sn,
                    mask=(k_idx[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0,
                ).to(tl.float32)
                q = q - zero

            b = q * scale
            acc += tl.dot(a.to(tl.float32), b, allow_tf32=False)

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    def w4a16_gemm_triton(
        a: torch.Tensor,
        packed_b: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor | None,
        group_size: int = 128,
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 128,
        num_warps: int = 4,
        num_stages: int = 3,
    ) -> torch.Tensor:
        require_cuda(a, "a")
        require_cuda(packed_b, "packed_b")
        M, K = a.shape
        N = packed_b.shape[1]
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)

        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
        _w4a16_gemm_kernel[grid](
            a, packed_b, c, scales, zeros if zeros is not None else scales,
            M, N, K,
            a.stride(0), a.stride(1),
            packed_b.stride(0), packed_b.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0), scales.stride(1),
            HAS_ZEROS=zeros is not None,
            GROUP_SIZE=group_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return c.to(a.dtype)

else:  # pragma: no cover - depends on the host

    def w4a16_gemm_triton(*_args, **_kwargs):
        raise RuntimeError("Triton is unavailable on this host")


#: The autotuner explores this space. Combinations that exceed the device's
#: shared memory simply fail to compile and are recorded as rejected, which is
#: why the search catches compilation errors rather than letting them escape.
W4A16_SEARCH_SPACE = {
    "block_m": [16, 32, 64, 128],
    "block_n": [32, 64, 128, 256],
    "block_k": [32, 64, 128, 256],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4, 5],
}
