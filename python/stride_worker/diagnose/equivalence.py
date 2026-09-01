"""Deciding whether two runs computed the same thing.

Comparing generated **text** is the wrong test for any numerical change, and it
is worth being blunt about why, because it is the test everyone reaches for
first.

Greedy decoding takes an argmax. When the top two logits are close — which
happens constantly in real text, at every "the" versus "a" — a difference of
1e-4 flips the choice. That one different token then conditions everything
after it, so the two sequences diverge completely. The output looks like
catastrophic disagreement; the underlying computation differed by less than
bf16 can represent.

Tensor parallelism *will* produce such differences even when perfectly correct:
splitting a matmul changes the order of a floating-point reduction, and
floating-point addition is not associative. So "TP=2 generated different text
from TP=1" is not evidence of a bug. It is not evidence of correctness either.
It is not evidence.

What settles it is comparing **logits on identical input**: cosine similarity,
how often the argmax agrees across many positions, and how far apart the
distributions are. Those distinguish "reduction order changed" from "attending
to the wrong keys", which generated text cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Cosine similarity below which two logit vectors are not the same
#: computation, per dtype. Above the threshold the difference is consistent
#: with reduction-order noise; below it, something structural differs.
COSINE_FLOOR: dict[torch.dtype, float] = {
    torch.float32: 0.999999,
    torch.bfloat16: 0.9995,
    torch.float16: 0.9998,
}

#: How often the argmax may legitimately disagree. Not zero: genuine near-ties
#: exist, and at a tie the winner is decided by noise in either run.
ARGMAX_TOLERANCE = 0.02


@dataclass
class Agreement:
    n: int
    cosine: float
    max_abs_diff: float
    argmax_match_rate: float
    top5_overlap: float
    #: Largest logit gap at a position where the argmax disagreed. A small gap
    #: means the runs disagreed at a near-tie, which is expected. A large one
    #: means they genuinely ranked different tokens first.
    worst_disagreement_gap: float

    def verdict(self, dtype: torch.dtype = torch.bfloat16) -> tuple[bool, str]:
        floor = COSINE_FLOOR.get(dtype, 0.999)

        if self.cosine < floor:
            return False, (
                f"cosine {self.cosine:.6f} is below {floor} for {dtype}. The two "
                "runs are not computing the same function — this is not "
                "reduction-order noise."
            )
        if 1.0 - self.argmax_match_rate > ARGMAX_TOLERANCE:
            return False, (
                f"argmax agreed on only {self.argmax_match_rate:.1%} of positions. "
                "Logits are close but the ranking differs too often to be ties."
            )
        if self.worst_disagreement_gap > 0.5:
            return False, (
                f"argmax disagreed where the top two logits were {self.worst_disagreement_gap:.3f} "
                "apart. That is not a near-tie; the runs genuinely ranked "
                "different tokens first."
            )
        return True, (
            f"equivalent: cosine {self.cosine:.6f}, argmax agrees "
            f"{self.argmax_match_rate:.1%}, disagreements only at gaps up to "
            f"{self.worst_disagreement_gap:.4f}. Consistent with reduction-order "
            "noise, which is expected and not a defect."
        )


def compare_logits(a: torch.Tensor, b: torch.Tensor) -> Agreement:
    """Compare two (positions, vocab) logit tensors from identical input."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    a32 = a.detach().float().cpu()
    b32 = b.detach().float().cpu()
    if a32.ndim == 1:
        a32, b32 = a32.unsqueeze(0), b32.unsqueeze(0)

    cosine = float(
        torch.nn.functional.cosine_similarity(a32, b32, dim=-1).min()
    )
    max_abs = float((a32 - b32).abs().max())

    arg_a, arg_b = a32.argmax(-1), b32.argmax(-1)
    matches = arg_a == arg_b
    match_rate = float(matches.float().mean())

    k = min(5, a32.shape[-1])
    top_a = a32.topk(k, dim=-1).indices
    top_b = b32.topk(k, dim=-1).indices
    overlap = sum(
        len(set(top_a[i].tolist()) & set(top_b[i].tolist())) / k
        for i in range(a32.shape[0])
    ) / a32.shape[0]

    # At each disagreement, how decisive was the reference? A tie is forgivable.
    worst_gap = 0.0
    if (~matches).any():
        top2 = a32.topk(min(2, a32.shape[-1]), dim=-1).values
        if top2.shape[-1] == 2:
            gaps = (top2[:, 0] - top2[:, 1])[~matches]
            worst_gap = float(gaps.max()) if gaps.numel() else 0.0

    return Agreement(
        n=a32.shape[0],
        cosine=cosine,
        max_abs_diff=max_abs,
        argmax_match_rate=match_rate,
        top5_overlap=float(overlap),
        worst_disagreement_gap=worst_gap,
    )


@dataclass
class LayerDivergence:
    layer: int
    cosine: float
    max_abs_diff: float


def first_divergence(
    a: list[torch.Tensor], b: list[torch.Tensor], cosine_floor: float = 0.9999
) -> LayerDivergence | None:
    """Find the first layer at which two captured runs stop agreeing.

    This is the measurement that actually localises a sharding bug. Error in a
    transformer compounds: a wrong collective in layer 3 makes every later layer
    disagree, so the *last* layer tells you nothing about where to look. The
    first layer to cross the floor is the one to read.
    """
    if len(a) != len(b):
        raise ValueError(f"captured {len(a)} layers versus {len(b)}")

    for index, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape:
            return LayerDivergence(index, 0.0, float("inf"))
        cos = float(
            torch.nn.functional.cosine_similarity(
                x.float().flatten(0, -2), y.float().flatten(0, -2), dim=-1
            ).min()
        )
        if cos < cosine_floor:
            return LayerDivergence(index, cos, float((x - y).abs().max()))
    return None


def report(name: str, agreement: Agreement, dtype: torch.dtype) -> str:
    ok, reason = agreement.verdict(dtype)
    lines = [
        f"{name}: {'EQUIVALENT' if ok else 'DIVERGENT'}",
        f"  positions compared      {agreement.n}",
        f"  cosine similarity       {agreement.cosine:.8f}",
        f"  max absolute difference {agreement.max_abs_diff:.3e}",
        f"  argmax agreement        {agreement.argmax_match_rate:.2%}",
        f"  top-5 overlap           {agreement.top5_overlap:.2%}",
        f"  worst disagreement gap  {agreement.worst_disagreement_gap:.4f}",
        f"  {reason}",
    ]
    return "\n".join(lines)
