"""Locating a tensor-parallel divergence, layer by layer.

Two runs of the same prompt at different parallel degrees should compute the
same hidden state at every layer, to within reduction-order noise. When they do
not, the *first* layer that disagrees is the one to read: error compounds, so by
the final layer everything disagrees and the last-layer numbers say nothing
about where the fault is.

Usage is two runs and a comparison:

    python -m stride_worker.diagnose.cli tp-dump --model CKPT --out tp1.pt
    torchrun --nproc_per_node=2 -m stride_worker.diagnose.cli tp-dump \\
        --model CKPT --out tp2.pt
    python -m stride_worker.diagnose.cli tp-compare tp1.pt tp2.pt

Only rank 0 writes a dump. Every rank runs the forward pass, because they all
have to enter the same collectives.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..cache import PagedKVCache
from ..model import StrideModel
from .equivalence import LayerDivergence, first_divergence
from .prefill import run_prompt


def dump(
    model: StrideModel,
    cache: PagedKVCache,
    tokens: list[int],
    path: str | Path,
) -> dict:
    """Capture per-layer hidden states and the final logits.

    Written only by rank 0. The stored metadata records the parallel degree and
    dtype so a comparison cannot silently pit two unrelated runs against each
    other.
    """
    logits, captured = run_prompt(model, cache, tokens, chunk_size=None, capture=True)

    payload = {
        "tensor_parallel_size": model.ctx.world_size,
        "dtype": str(model.dtype),
        "num_layers": model.num_layers,
        "tokens": tokens,
        "layers": captured,
        "logits": logits.detach().float().cpu(),
    }
    if model.ctx.is_leader:
        torch.save(payload, str(path))
    return payload


def compare(reference: str | Path, candidate: str | Path) -> tuple[bool, str]:
    """Compare two dumps and name the first layer that disagrees."""
    a = torch.load(str(reference), map_location="cpu", weights_only=False)
    b = torch.load(str(candidate), map_location="cpu", weights_only=False)

    if a["tokens"] != b["tokens"]:
        return False, (
            "the two dumps used different prompts, so nothing can be concluded. "
            "Re-run both with the same --prompt or --tokens."
        )
    if a["tensor_parallel_size"] == b["tensor_parallel_size"]:
        return False, (
            f"both dumps are tp={a['tensor_parallel_size']}; there is nothing to "
            "compare. One of them should be the tp=1 reference."
        )

    dtype = getattr(torch, str(a["dtype"]).replace("torch.", ""), torch.bfloat16)
    from .equivalence import compare_logits, report

    lines = [
        f"reference tp={a['tensor_parallel_size']}  candidate tp={b['tensor_parallel_size']}",
        f"prompt: {len(a['tokens'])} tokens, {a['num_layers']} layers captured",
        "",
    ]

    divergence: LayerDivergence | None = first_divergence(a["layers"], b["layers"])
    if divergence is None:
        lines.append("every layer agrees within tolerance.")
    else:
        lines.append(
            f"first divergence at layer {divergence.layer} "
            f"(cosine {divergence.cosine:.6f}, max abs {divergence.max_abs_diff:.3e})"
        )
        lines.append("")
        lines.append(_interpret(divergence))
        lines.append("")

    agreement = compare_logits(a["logits"], b["logits"])
    lines.append(report("final logits", agreement, dtype))

    ok = divergence is None and agreement.verdict(dtype)[0]
    return ok, "\n".join(lines)


def _interpret(d: LayerDivergence) -> str:
    """Turn a layer index into somewhere to look."""
    if d.layer == 0:
        return (
            "Layer 0 already differs, so the fault is before or inside the first\n"
            "block. Check the embedding (it is replicated, not sharded) and the\n"
            "first attention projections. A wrong split axis on q/k/v shows up\n"
            "immediately; a wrong all-reduce shows up here too."
        )
    return (
        f"Layers 0 to {d.layer - 1} agree, so the sharding of the projections and\n"
        f"the collectives are right in general. Something specific to layer\n"
        f"{d.layer} differs. Read that layer's weights in the checkpoint: an\n"
        "unusual shape, a bias where other layers have none, or a tensor whose\n"
        "output dimension does not divide by the parallel degree and was\n"
        "silently handled differently."
    )
