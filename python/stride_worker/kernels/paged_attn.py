"""Paged attention for the decode step.

The reference path gathers a sequence's scattered blocks into one contiguous
tensor before calling attention. For a 32k-token context that copies the entire
KV cache for that sequence on *every generated token*, which is pure waste on a
step that is already memory-bound.

This kernel walks the block table and reads each block where it lies, so the
only traffic is the KV itself. One program handles one (sequence, KV head)
pair, accumulating a running softmax so the scores never have to be
materialised.

**Decode only, and the restriction is load-bearing.** There is exactly one
query token per sequence and it sits at the end of the context, so it may
legitimately attend to every key. That is why no causal mask appears below.

Give this kernel several query positions — a prefill chunk — and it will return
plausible numbers computed against the *wrong* keys, because every query would
attend to the whole context including its own future. It will not crash. It
will not produce NaN. It will quietly attend to positions the model must not
see, and the error grows with the number of keys that should have been masked.

That failure was found on hardware exactly this way: cosine similarity against
the reference sat at 0.9999 for a single query at the end of a short context,
fell to ~0.997 once more pages were involved, and degraded further under
chunked prefill. Every one of those numbers is the missing mask, not arithmetic
drift.

So the entry point below validates its inputs and raises rather than computing.
A prefill-capable variant needs a query-position argument and a mask in the
score computation, and is not written.
"""

from __future__ import annotations

import torch

from . import require_cuda, triton_available

if triton_available():
    import triton
    import triton.language as tl

    #: Stand-in for negative infinity. Large enough that exp() of any difference
    #: from a real score underflows to zero, finite enough that subtracting it
    #: from itself gives zero rather than NaN.
    NEG_INF: tl.constexpr = -1.0e30

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
            # Loop-carried values must enter the loop already typed as fp32.
            # Initialising them from Python floats leaves Triton to infer the
            # type on the first assignment inside the loop, which is a
            # needlessly fragile thing to depend on.
            running_max = tl.full([], NEG_INF, tl.float32)
            running_sum = tl.zeros([], dtype=tl.float32)
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
                scores = tl.where(valid, scores, NEG_INF)

                block_max = tl.max(scores, axis=0)
                new_max = tl.maximum(running_max, block_max)
                # Rescale what has been accumulated so far onto the new maximum.
                # NEG_INF is a large finite sentinel rather than a true -inf so
                # that this subtraction cannot become inf - inf, which is NaN and
                # would poison the accumulator for the rest of the sequence.
                correction = tl.exp(running_max - new_max)
                weights = tl.exp(scores - new_max)
                weights = tl.where(valid, weights, 0.0)

                acc = acc * correction + tl.sum(weights[:, None] * v, axis=0)
                running_sum = running_sum * correction + tl.sum(weights, axis=0)
                running_max = new_max

            out = acc / tl.maximum(running_sum, 1e-6)
            tl.store(out_ptr + seq * q_stride_seq + q_head * q_stride_head + dims, out)

    class NotADecodeStep(ValueError):
        """Raised when this kernel is handed work it cannot compute correctly.

        A distinct type so a caller can catch it and fall back to the reference
        rather than treating it as a generic bad argument.
        """

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

        ``q`` is (num_seqs, num_q_heads, head_dim) — **one** query token per
        sequence, positioned at the end of its context. ``block_tables`` is
        (num_seqs, max_blocks), right-padded. ``context_lens`` gives each
        sequence's true length so padding is never read.

        Raises :class:`NotADecodeStep` for anything else. See the module
        docstring: without a mask, a multi-query call returns confident,
        wrong numbers.
        """
        require_cuda(q, "q")
        require_cuda(k_cache, "k_cache")

        if q.ndim != 3:
            raise NotADecodeStep(
                f"q must be (num_seqs, num_q_heads, head_dim); got shape "
                f"{tuple(q.shape)}. A prefill chunk with several query "
                "positions cannot be computed here — this kernel applies no "
                "causal mask, so every query would attend to its own future."
            )

        num_seqs, num_q_heads, head_dim = q.shape
        _, block_size, num_kv_heads, _ = k_cache.shape
        if num_q_heads % num_kv_heads:
            raise ValueError(f"{num_q_heads} query heads do not group into {num_kv_heads}")

        if block_tables.shape[0] != num_seqs or context_lens.shape[0] != num_seqs:
            raise NotADecodeStep(
                f"expected one block table and one context length per sequence "
                f"({num_seqs}); got {tuple(block_tables.shape)} and "
                f"{tuple(context_lens.shape)}"
            )
        if int(context_lens.min()) < 1:
            raise NotADecodeStep("every sequence needs at least one key to attend to")

        max_needed = int((context_lens + block_size - 1).div(block_size, rounding_mode="floor").max())
        if max_needed > block_tables.shape[1]:
            raise NotADecodeStep(
                f"a context of {int(context_lens.max())} tokens needs {max_needed} "
                f"blocks but the table has room for {block_tables.shape[1]}"
            )
        # Block ids index a pointer computation; a stale or padded entry would
        # read another sequence's KV rather than fail.
        if int(block_tables[:, :max_needed].min()) < 0 or int(
            block_tables[:, :max_needed].max()
        ) >= k_cache.shape[0]:
            raise NotADecodeStep(
                "block table contains an id outside the cache; padding must sit "
                "past the blocks a context actually uses"
            )

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

    class NotADecodeStep(ValueError):
        """See the CUDA branch above."""

    def paged_attention_decode_triton(*_args, **_kwargs):
        raise RuntimeError("Triton is unavailable on this host")


PAGED_ATTN_SEARCH_SPACE = {
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2, 3, 4],
}
