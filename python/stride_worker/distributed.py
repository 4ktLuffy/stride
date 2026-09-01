"""Tensor parallelism over NCCL.

One process per GPU, launched with ``torchrun``. Rank 0 owns the TCP socket and
talks to the Rust control plane; every other rank sits in a loop waiting to be
told what to execute. The control plane is unaware that there is more than one
GPU — it addresses one worker, and the sharding is entirely below that seam.

Two process groups, deliberately:

* **NCCL** for tensor collectives inside the forward pass. It is the only
  backend that will saturate NVLink.
* **Gloo** for control messages. Broadcasting a Python object over NCCL means
  staging it through device memory, which is both slower and more fragile for
  the few hundred bytes a work description occupies.

The sharding is the standard Megatron split: column-parallel where an output
dimension can be cut without communication, row-parallel where the reduction
axis is cut and an all-reduce puts it back together. Two all-reduces per layer,
after attention output and after the MLP down-projection.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

log = logging.getLogger("stride.distributed")

#: Sentinel broadcast by rank 0 to bring the other ranks down cleanly.
SHUTDOWN = None


@dataclass(frozen=True)
class ParallelContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    #: Gloo group used only for broadcasting work descriptions.
    control_group: object | None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_leader(self) -> bool:
        """Rank 0 owns the socket and is the only rank the control plane sees."""
        return self.rank == 0


def init(device_arg: str = "cuda") -> ParallelContext:
    """Join the process group if launched under torchrun, else run standalone.

    Detects torchrun by its environment rather than by a flag, so the same
    command works with and without it.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size == 1:
        device = torch.device(device_arg)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return ParallelContext(0, 1, 0, device, None)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "tensor parallelism requires CUDA: NCCL has no CPU backend. "
            "Run a single rank with --device cpu instead."
        )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    # Separate CPU-side group for object broadcasts.
    control_group = dist.new_group(backend="gloo")

    log.info("rank %d of %d on %s", rank, world_size, device)
    return ParallelContext(rank, world_size, local_rank, device, control_group)


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# --- sharding ---------------------------------------------------------------


def shard_column(t: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Split a weight along its output dimension.

    For ``y = W x`` with ``W`` of shape ``[out, in]``, each rank computes a
    slice of ``y`` from the whole ``x``. No communication is needed, because
    every rank already has the full input.
    """
    if world_size == 1:
        return t
    if t.shape[0] % world_size:
        raise ValueError(
            f"output dimension {t.shape[0]} does not divide across {world_size} ranks"
        )
    return t.chunk(world_size, dim=0)[rank].contiguous()


def shard_row(t: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Split a weight along its reduction dimension.

    Each rank holds a slice of ``x`` and produces a *partial* ``y``. The partials
    are summed by an all-reduce, which is why every row-parallel layer is
    followed by one.
    """
    if world_size == 1:
        return t
    if t.shape[1] % world_size:
        raise ValueError(
            f"reduction dimension {t.shape[1]} does not divide across {world_size} ranks"
        )
    return t.chunk(world_size, dim=1)[rank].contiguous()


def all_reduce(t: torch.Tensor, ctx: ParallelContext) -> torch.Tensor:
    """Sum partial results across ranks, in place."""
    if not ctx.is_distributed:
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_gather_last_dim(t: torch.Tensor, ctx: ParallelContext) -> torch.Tensor:
    """Concatenate a tensor split along its last dimension across ranks.

    Used to reassemble vocabulary-parallel logits. Rank order is the shard
    order, which is what makes the concatenation give back the original
    vocabulary indexing.
    """
    if not ctx.is_distributed:
        return t
    parts = [torch.empty_like(t) for _ in range(ctx.world_size)]
    dist.all_gather(parts, t.contiguous())
    return torch.cat(parts, dim=-1)


def agree_on_min(value: int, ctx: ParallelContext) -> int:
    """Reduce an integer to its minimum across ranks.

    Every rank must use the *same* KV block count: a block id in a table sent by
    the control plane has to name the same logical slot on every GPU. Ranks can
    legitimately compute different capacities — a card running a display has
    less free memory — so the smallest wins and all ranks agree.
    """
    if not ctx.is_distributed:
        return value
    t = torch.tensor([value], dtype=torch.long, device=ctx.device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


# --- control messages -------------------------------------------------------


def broadcast_work(message, ctx: ParallelContext) -> None:
    """Rank 0 tells every other rank what to execute next."""
    if not ctx.is_distributed:
        return
    payload = [message]
    dist.broadcast_object_list(payload, src=0, group=ctx.control_group)


def receive_work(ctx: ParallelContext):
    """Non-leader ranks block here until rank 0 sends the next instruction."""
    payload = [None]
    dist.broadcast_object_list(payload, src=0, group=ctx.control_group)
    return payload[0]


def barrier(ctx: ParallelContext) -> None:
    if ctx.is_distributed:
        dist.barrier()


def validate_plan(num_q_heads: int, num_kv_heads: int, world_size: int) -> None:
    """Reject a sharding the geometry cannot support.

    These are hard constraints, not preferences. A plan that fails here would
    either crash on the first matmul or silently replicate what it was asked to
    shard, and the second failure mode is far worse.
    """
    if world_size == 1:
        return
    if num_q_heads % world_size:
        raise ValueError(
            f"{num_q_heads} query heads do not divide across {world_size} ranks"
        )
    if num_kv_heads % world_size:
        raise ValueError(
            f"{num_kv_heads} KV heads do not divide across {world_size} ranks. "
            f"This model supports tensor parallelism up to {num_kv_heads} ranks; "
            "beyond that the KV heads would have to be replicated, which is not "
            "implemented here."
        )
