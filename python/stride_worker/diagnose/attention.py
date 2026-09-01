"""Isolating where the paged-attention kernel stops matching the reference.

Run as a sweep over context length so the answer is a *boundary*, not a single
pass/fail. A kernel that agrees at one page and drifts at four has a different
bug from one that is wrong everywhere, and the sweep separates them without
guessing.

Only the decode regime is swept: one query token per sequence, at the end of its
context. That is the whole of what the kernel implements, and comparing it
outside that regime measures a missing feature rather than a defect. The kernel
now refuses such calls, and this sweep asserts that it does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..cache import CacheSpec, PagedKVCache
from ..layers import paged_attention


@dataclass
class Point:
    context_len: int
    pages: int
    cosine: float
    max_abs_diff: float
    ok: bool


def _reference(cache: PagedKVCache, blocks, q, context_len, scale):
    """The PyTorch path, used as the definition of correct."""
    return paged_attention(
        q,
        cache,
        layer=0,
        blocks=blocks,
        context_len=context_len,
        query_start=context_len - 1,
        scale=scale,
    )


def sweep_decode(
    context_lengths: list[int],
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    cosine_floor: float = 0.999,
    seed: int = 0,
) -> list[Point]:
    """Compare kernel against reference across growing contexts."""
    from ..kernels.paged_attn import paged_attention_decode_triton

    torch.manual_seed(seed)
    dev = torch.device(device)
    scale = head_dim**-0.5
    max_len = max(context_lengths)
    num_blocks = (max_len + block_size - 1) // block_size + 4

    spec = CacheSpec(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=dev,
    )
    cache = PagedKVCache(spec)
    # Fill every block with real values: zeros would let an out-of-range block
    # id read plausible data and hide an indexing bug.
    cache.k[0].normal_()
    cache.v[0].normal_()

    results: list[Point] = []
    for context_len in context_lengths:
        pages = (context_len + block_size - 1) // block_size
        blocks = list(range(pages))
        q = torch.randn(1, num_q_heads, head_dim, dtype=dtype, device=dev)

        want = _reference(cache, blocks, q[0], context_len, scale)

        table = torch.full((1, pages), 0, dtype=torch.int32, device=dev)
        table[0, :pages] = torch.tensor(blocks, dtype=torch.int32, device=dev)
        lens = torch.tensor([context_len], dtype=torch.int32, device=dev)
        got = paged_attention_decode_triton(q, cache.k[0], cache.v[0], table, lens, scale)

        cos = float(
            torch.nn.functional.cosine_similarity(
                got[0].float().flatten(), want.float().flatten(), dim=0
            )
        )
        results.append(
            Point(
                context_len=context_len,
                pages=pages,
                cosine=cos,
                max_abs_diff=float((got[0].float() - want.float()).abs().max()),
                ok=cos >= cosine_floor,
            )
        )
    return results


def assert_refuses_prefill(device: str = "cuda") -> tuple[bool, str]:
    """The kernel must reject multi-query input rather than compute it.

    Without a causal mask, a prefill-shaped call returns confident numbers
    computed against keys the queries must not see. Refusing is the only safe
    behaviour, and this asserts the refusal is actually wired up.
    """
    from ..kernels.paged_attn import NotADecodeStep, paged_attention_decode_triton

    dev = torch.device(device)
    # Four query positions: a prefill chunk, not a decode step.
    q = torch.randn(1, 4, 32, 128, dtype=torch.bfloat16, device=dev)
    k = torch.randn(8, 16, 8, 128, dtype=torch.bfloat16, device=dev)
    table = torch.zeros((1, 4), dtype=torch.int32, device=dev)
    lens = torch.tensor([64], dtype=torch.int32, device=dev)

    try:
        paged_attention_decode_triton(q, k, k, table, lens, 0.088)
    except NotADecodeStep as e:
        return True, f"refused as it should: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"raised the wrong error type ({type(e).__name__}): {e}"
    return False, (
        "the kernel accepted a prefill-shaped call. It applies no causal mask, "
        "so it just computed attention over each query's own future and returned "
        "numbers that look fine."
    )


def format_sweep(points: list[Point]) -> str:
    lines = [
        f"{'context':>8} {'pages':>6} {'cosine':>12} {'max abs':>12}  verdict",
        "-" * 56,
    ]
    for p in points:
        lines.append(
            f"{p.context_len:>8} {p.pages:>6} {p.cosine:>12.6f} "
            f"{p.max_abs_diff:>12.3e}  {'ok' if p.ok else 'DIVERGED'}"
        )

    failures = [p for p in points if not p.ok]
    lines.append("")
    if not failures:
        lines.append("Kernel matches the reference across every context tested.")
    else:
        first = failures[0]
        clean = [p for p in points if p.ok]
        if clean:
            lines.append(
                f"Agrees up to {max(p.context_len for p in clean)} tokens "
                f"({max(p.pages for p in clean)} pages), diverges from "
                f"{first.context_len} ({first.pages} pages)."
            )
            lines.append(
                "A boundary rather than uniform failure points at the block-walk: "
                "the page index, the block-table load, or the online-softmax "
                "rescaling between pages."
            )
        else:
            lines.append(
                "Diverges even at a single page, so the block walk is not the "
                "issue. Look at the score computation or the head mapping."
            )
    return "\n".join(lines)
