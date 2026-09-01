"""Paged attention for the decode step.

The reference path gathers a sequence's scattered blocks into one contiguous
tensor before calling attention. For a 32k-token context that copies the entire
KV cache for that sequence on *every generated token*, which is pure waste on a
step that is already memory-bound.

This kernel walks the block table and reads each block where it lies, so the
only traffic is the KV itself. One program handles one (sequence, KV head)
pair, accumulating a running softmax so the scores never have to be
materialised.
"""

from __future__ import annotations

import torch

from . import require_cuda, triton_available

if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _paged_attn_decode_kernel(
        q_ptr,               # [num_seqs, num_q_heads, head_dim]
        k_cache_ptr,         # [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache_ptr,
        out_ptr,             # [num_seqs, num_q_heads, head_dim]
        block_table_ptr,     # [num_seqs, max_blocks]
        context_lens_ptr,    # [num_seqs]
        scale,
        max_blocks,
        q_stride_seq,
        q_stride_head,
        kv_stride_block,
        kv_stride_slot,
        kv_stride_head,
        num_queries_per_kv: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        seq = tl.program_id(0)
        kv_head = tl.program_id(1)

        context_len = tl.load(context_lens_ptr + seq)
        dims = tl.arange(0, HEAD_DIM)
        slots = tl.arange(0, BLOCK_SIZE)

        # Every query head in this group shares the same KV, so they are
        # processed together and the KV is read once for all of them.
        for group in range(num_queries_per_kv):
            q_head = kv_head * num_queries_per_kv + group
            q = tl.load(q_ptr + seq * q_stride_seq + q_head * q_stride_head + dims).to(
                tl.float32
            )

            # Online softmax: running maximum, running denominator, running
            # weighted sum. Avoids a second pass over the context.
            running_max = float("-inf")
            running_sum = 0.0
            acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

            num_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            for block_index in range(num_blocks):
                physical = tl.load(block_table_ptr + seq * max_blocks + block_index)
                positions = block_index * BLOCK_SIZE + slots
                valid = positions < context_len

                base = physical * kv_stride_block + kv_head * kv_stride_head
                k = tl.load(
                    k_cache_ptr + base + slots[:, None] * kv_stride_slot + dims[None, :],
                    mask=valid[:, None],
                    other=0.0,
                ).to(tl.float32)
                v = tl.load(
                    v_cache_ptr + base + slots[:, None] * kv_stride_slot + dims[None, :],
                    mask=valid[:, None],
                    other=0.0,
                ).to(tl.float32)

                scores = tl.sum(k * q[None, :], axis=1) * scale
                scores = tl.where(valid, scores, float("-inf"))

                block_max = tl.max(scores, axis=0)
                new_max = tl.maximum(running_max, block_max)
                # Rescale what has been accumulated so far onto the new maximum.
                correction = tl.exp(running_max - new_max)
                weights = tl.exp(scores - new_max)
                weights = tl.where(valid, weights, 0.0)

                acc = acc * correction + tl.sum(weights[:, None] * v, axis=0)
                running_sum = running_sum * correction + tl.sum(weights, axis=0)
                running_max = new_max

            out = acc / tl.maximum(running_sum, 1e-6)
            tl.store(out_ptr + seq * q_stride_seq + q_head * q_stride_head + dims, out)

    def paged_attention_decode_triton(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        scale: float,
        num_warps: int = 4,
        num_stages: int = 2,
    ) -> torch.Tensor:
        """One decode step for a batch of sequences.

        ``q`` is (num_seqs, num_q_heads, head_dim). ``block_tables`` is
        (num_seqs, max_blocks), right-padded. ``context_lens`` gives each
        sequence's true length so padding is never read.
        """
        require_cuda(q, "q")
        require_cuda(k_cache, "k_cache")
        num_seqs, num_q_heads, head_dim = q.shape
        _, block_size, num_kv_heads, _ = k_cache.shape
        if num_q_heads % num_kv_heads:
            raise ValueError(f"{num_q_heads} query heads do not group into {num_kv_heads}")

        out = torch.empty_like(q)
        _paged_attn_decode_kernel[(num_seqs, num_kv_heads)](
            q,
            k_cache,
            v_cache,
            out,
            block_tables,
            context_lens,
            scale,
            block_tables.shape[1],
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            num_queries_per_kv=num_q_heads // num_kv_heads,
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

else:  # pragma: no cover - depends on the host

    def paged_attention_decode_triton(*_args, **_kwargs):
        raise RuntimeError("Triton is unavailable on this host")


PAGED_ATTN_SEARCH_SPACE = {
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2, 3, 4],
}
