"""Fused RMSNorm.

The PyTorch reference reads the input three times — once for the squares, once
to normalise, once to scale — and writes two intermediates. This kernel keeps
the row in registers and writes once, which is the entire win: RMSNorm is
memory-bound, so the arithmetic is free and the traffic is not.

Accumulation is in float32 regardless of input dtype. Summing squares over a
wide hidden dimension in BF16 loses precision long before it overflows, and the
error lands in every downstream activation.
"""

from __future__ import annotations

import torch

from . import require_cuda, triton_available

if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        row_stride,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_row = x_ptr + row * row_stride
        y_row = y_ptr + row * row_stride

        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for offset in range(0, n_cols, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
            acc += x * x
        rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)

        for offset in range(0, n_cols, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_cols
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(y_row + cols, x * rstd * w, mask=mask)

    def rms_norm_triton(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-5,
        block_size: int = 1024,
        num_warps: int = 4,
        num_stages: int = 1,
    ) -> torch.Tensor:
        require_cuda(x, "x")
        require_cuda(weight, "weight")
        shape = x.shape
        x2d = x.reshape(-1, shape[-1])
        y = torch.empty_like(x2d)
        n_rows, n_cols = x2d.shape
        if n_rows == 0:
            return y.reshape(shape)

        _rms_norm_kernel[(n_rows,)](
            x2d,
            weight,
            y,
            x2d.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y.reshape(shape)

else:  # pragma: no cover - depends on the host

    def rms_norm_triton(*_args, **_kwargs):
        raise RuntimeError(
            "Triton is unavailable on this host; use the PyTorch reference in "
            "stride_worker.layers.rms_norm"
        )


#: Search space the autotuner explores for this kernel.
RMSNORM_SEARCH_SPACE = {
    "block_size": [256, 512, 1024, 2048, 4096],
    "num_warps": [1, 2, 4, 8, 16],
    "num_stages": [1, 2, 3],
}
