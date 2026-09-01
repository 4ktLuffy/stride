"""Paged KV cache on the device.

The block pool is one contiguous tensor per layer, indexed by block id. The
control plane owns allocation — which block belongs to which sequence, when a
block is shared, when it is evicted. This module owns only the storage and the
gather/scatter, so there is exactly one allocator in the system and it is the
one that is tested.

Layout is ``[num_blocks, block_size, num_kv_heads, head_dim]`` per tensor, with
K and V held separately. Keeping the block dimension outermost makes a block a
contiguous slice, which is what lets a copy-free gather work.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheSpec:
    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    @property
    def bytes_per_block(self) -> int:
        elements = 2 * self.block_size * self.num_kv_heads * self.head_dim
        return elements * self.dtype.itemsize * self.num_layers

    @property
    def total_bytes(self) -> int:
        return self.bytes_per_block * self.num_blocks


class PagedKVCache:
    """Device-side block storage for every layer."""

    def __init__(self, spec: CacheSpec):
        self.spec = spec
        shape = (spec.num_blocks, spec.block_size, spec.num_kv_heads, spec.head_dim)
        # Allocated up front. A serving process that grows its cache under load
        # will fragment and eventually fail at the worst possible moment.
        self.k = [
            torch.zeros(shape, dtype=spec.dtype, device=spec.device)
            for _ in range(spec.num_layers)
        ]
        self.v = [
            torch.zeros(shape, dtype=spec.dtype, device=spec.device)
            for _ in range(spec.num_layers)
        ]

    def write(
        self,
        layer: int,
        blocks: list[int],
        start_position: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Scatter ``k``/``v`` for ``n`` tokens into a sequence's blocks.

        ``k`` and ``v`` are (n, num_kv_heads, head_dim); ``start_position`` is
        where the first of them sits in the sequence.
        """
        n = k.shape[0]
        if n == 0:
            return
        bs = self.spec.block_size
        positions = torch.arange(
            start_position, start_position + n, device=k.device
        )
        block_index = positions // bs
        slot = positions % bs

        block_tensor = torch.tensor(blocks, device=k.device, dtype=torch.long)
        if int(block_index.max()) >= block_tensor.numel():
            raise IndexError(
                f"position {start_position + n - 1} needs block "
                f"{int(block_index.max())} but only {block_tensor.numel()} are mapped"
            )
        physical = block_tensor[block_index]

        self.k[layer][physical, slot] = k.to(self.spec.dtype)
        self.v[layer][physical, slot] = v.to(self.spec.dtype)

    def gather(self, layer: int, blocks: list[int], length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect a sequence's first ``length`` tokens into contiguous tensors.

        Returns (length, num_kv_heads, head_dim) for K and V. This is the
        reference path: correct, and deliberately simple. The Triton paged
        attention kernel reads the blocks in place instead, which is the whole
        point of writing it.
        """
        if length == 0:
            empty = torch.empty(
                (0, self.spec.num_kv_heads, self.spec.head_dim),
                dtype=self.spec.dtype,
                device=self.spec.device,
            )
            return empty, empty

        bs = self.spec.block_size
        positions = torch.arange(length, device=self.spec.device)
        block_tensor = torch.tensor(blocks, device=self.spec.device, dtype=torch.long)
        physical = block_tensor[positions // bs]
        slot = positions % bs
        return self.k[layer][physical, slot], self.v[layer][physical, slot]

    def zero_(self) -> None:
        for t in self.k:
            t.zero_()
        for t in self.v:
            t.zero_()
