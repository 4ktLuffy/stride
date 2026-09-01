"""Is a chunked prefill the same computation as a continuous one?

Splitting a prompt across steps changes nothing the model should notice: the
same tokens, the same positions, the same causal mask. If the chunked path
diverges, the mask or the position arithmetic is wrong at a chunk boundary.

The comparison is on **logits from identical input**, not on generated text.
See :mod:`stride_worker.diagnose.equivalence` for why text cannot settle this.
"""

from __future__ import annotations

import torch

from ..cache import PagedKVCache
from ..model import StrideModel
from ..protocol import SequenceWork
from .equivalence import Agreement, compare_logits


def run_prompt(
    model: StrideModel,
    cache: PagedKVCache,
    tokens: list[int],
    chunk_size: int | None = None,
    capture: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Prefill ``tokens`` in chunks and return the final logits.

    ``chunk_size`` of ``None`` runs the whole prompt in one pass. The cache is
    cleared first so a previous run cannot leak into this one — which would
    make the two paths agree for the wrong reason.
    """
    n = len(tokens)
    if n == 0:
        raise ValueError("nothing to prefill")

    block_size = cache.spec.block_size
    needed = (n + block_size - 1) // block_size
    if needed > cache.spec.num_blocks:
        raise ValueError(
            f"{n} tokens need {needed} blocks; the cache holds {cache.spec.num_blocks}"
        )
    blocks = list(range(needed))
    cache.zero_()

    step = chunk_size or n
    logits: torch.Tensor | None = None
    captured: list[torch.Tensor] = []

    position = 0
    while position < n:
        length = min(step, n - position)
        last = position + length >= n
        work = [
            SequenceWork(
                seq=0,
                tokens=tokens[position : position + length],
                position=position,
                blocks=blocks,
                needs_logits=last,
            )
        ]
        if capture and last:
            model.capture = []
        _, out, _ = model.forward(work, cache)
        if capture and last:
            captured = model.capture or []
            model.capture = None
        if last:
            logits = out
        position += length

    assert logits is not None
    return logits, captured


def check(
    model: StrideModel,
    cache: PagedKVCache,
    tokens: list[int],
    chunk_size: int,
) -> Agreement:
    """Compare continuous prefill against chunked prefill of the same prompt."""
    whole, _ = run_prompt(model, cache, tokens, chunk_size=None)
    chunked, _ = run_prompt(model, cache, tokens, chunk_size=chunk_size)
    return compare_logits(whole, chunked)


def sweep(
    model: StrideModel,
    cache: PagedKVCache,
    tokens: list[int],
    chunk_sizes: list[int],
) -> dict[int, Agreement]:
    """Compare several chunk sizes against the continuous run.

    A divergence that appears only once the prompt crosses more than one
    boundary points at position arithmetic; one that appears at the very first
    boundary points at the mask.
    """
    whole, _ = run_prompt(model, cache, tokens, chunk_size=None)
    results: dict[int, Agreement] = {}
    for size in chunk_sizes:
        chunked, _ = run_prompt(model, cache, tokens, chunk_size=size)
        results[size] = compare_logits(whole, chunked)
    return results
