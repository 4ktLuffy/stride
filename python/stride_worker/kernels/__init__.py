"""Triton kernels, and the dispatch that decides whether to use them.

Every kernel here has a PyTorch reference in ``stride_worker.layers``. The
reference is the definition of correct; a kernel is an optimisation that must
agree with it inside a declared tolerance, and ``stride_worker.autotune``
enforces that before any kernel is allowed to run in serving.

**None of these kernels has been executed.** They were written on a machine
with no CUDA device. Treat them as unverified until the autotuner has passed
them on the target hardware — that is exactly what the gate is for.
"""

from __future__ import annotations

import functools
import os

import torch


@functools.lru_cache(maxsize=1)
def triton_available() -> bool:
    """True when Triton can actually compile and launch on this machine."""
    if os.environ.get("STRIDE_DISABLE_TRITON"):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except ImportError:
        return False
    return True


def require_cuda(t: torch.Tensor, name: str) -> None:
    if not t.is_cuda:
        raise ValueError(f"{name} must be on a CUDA device, got {t.device}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
