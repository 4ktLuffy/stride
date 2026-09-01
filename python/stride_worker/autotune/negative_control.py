"""Negative controls for the correctness gate.

A gate that has never rejected anything is not evidence of correctness — it may
simply be broken. These are deliberately wrong implementations, each failing in
a different way, and the gate is required to reject every one of them.

If any control passes, the gate is not doing its job, and every kernel it has
ever approved is unverified. That is why this runs in CI alongside the kernels
themselves, and why it runs on CPU: the controls exercise the gate, not the
hardware, so they must not be skipped on a machine without a GPU.

Each control names the real failure mode it stands in for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .gate import GateResult, Tolerance, gate


def reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """The definition every variant below is measured against."""
    dtype = x.dtype
    x32 = x.float()
    variance = x32.pow(2).mean(-1, keepdim=True)
    return (x32 * torch.rsqrt(variance + eps) * weight.float()).to(dtype)


# --- broken variants --------------------------------------------------------


def _no_epsilon(x, weight, eps=1e-5):
    """Drops epsilon. Fine until a row is all zeros, then divides by zero.

    Stands in for a kernel that looks correct on random test data and produces
    NaN on the padding rows of a real batch.
    """
    x32 = x.float()
    variance = x32.pow(2).mean(-1, keepdim=True)
    return (x32 * torch.rsqrt(variance) * weight.float()).to(x.dtype)


def _wrong_reduction_axis(x, weight, eps=1e-5):
    """Reduces over the batch instead of the hidden dimension.

    Stands in for an index arithmetic error — the most common way a hand-written
    kernel goes wrong, and one that still produces plausible-looking output.
    """
    x32 = x.float()
    variance = x32.pow(2).mean(0, keepdim=True)
    return (x32 * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def _scale_before_normalising(x, weight, eps=1e-5):
    """Applies the weight before computing the variance rather than after.

    Stands in for a fusion that reordered operations that do not commute.
    """
    x32 = x.float() * weight.float()
    variance = x32.pow(2).mean(-1, keepdim=True)
    return (x32 * torch.rsqrt(variance + eps)).to(x.dtype)


def _half_precision_accumulation(x, weight, eps=1e-5):
    """Accumulates the sum of squares in float16 instead of float32.

    Stands in for the single most common quantisation-era kernel bug: an
    accumulator narrow enough to lose the tail of a wide reduction.
    """
    x16 = x.to(torch.float16)
    variance = x16.pow(2).mean(-1, keepdim=True, dtype=torch.float16)
    return (x16 * torch.rsqrt(variance + eps) * weight.to(torch.float16)).to(x.dtype)


def _drops_last_column(x, weight, eps=1e-5):
    """Off-by-one in the tail mask: the last element is never written.

    Stands in for a bounds check written as `<` where it needed `<=`.
    """
    out = reference_rms_norm(x, weight, eps)
    out[..., -1] = 0
    return out


def _emits_nan(x, weight, eps=1e-5):
    """Poisons one element. Benchmarks happily; destroys generation quality."""
    out = reference_rms_norm(x, weight, eps)
    out[0, 0] = float("nan")
    return out


def _non_deterministic(x, weight, eps=1e-5):
    """Adds unseeded noise, as a stand-in for a race on shared memory.

    Numerically it is almost right, so only the determinism check catches it.
    That is the point: the same input twice must give the same output.
    """
    out = reference_rms_norm(x, weight, eps)
    return out + torch.randn_like(out) * 1e-2


def _subtly_wrong_scale(x, weight, eps=1e-5):
    """Off by 0.5%. Far too small to notice by eye, far too large to accept."""
    return reference_rms_norm(x, weight, eps) * 1.005


@dataclass(frozen=True)
class Control:
    name: str
    fn: Callable[..., torch.Tensor]
    failure_mode: str


CONTROLS: list[Control] = [
    Control("no_epsilon", _no_epsilon, "divides by zero on an all-zero row"),
    Control("wrong_reduction_axis", _wrong_reduction_axis, "reduces the wrong dimension"),
    Control("scale_before_normalising", _scale_before_normalising, "reordered non-commuting ops"),
    Control("fp16_accumulation", _half_precision_accumulation, "accumulator too narrow"),
    Control("drops_last_column", _drops_last_column, "off-by-one in the tail mask"),
    Control("emits_nan", _emits_nan, "poisons an element with NaN"),
    Control("non_deterministic", _non_deterministic, "race on shared memory"),
    Control("subtly_wrong_scale", _subtly_wrong_scale, "0.5% scale error"),
]


def input_factory(
    device: str = "cpu", dtype: torch.dtype = torch.float32
) -> Callable[[int], tuple]:
    """Deterministic inputs, varying shape by trial.

    Trial 0 deliberately includes an all-zero row: without one, a kernel that
    omits epsilon passes every check and fails in production.
    """

    def make(trial: int) -> tuple:
        generator = torch.Generator(device="cpu").manual_seed(1000 + trial)
        rows = [8, 16, 1, 33, 128, 7][trial % 6]
        cols = [512, 1024, 4096, 768, 2048, 129][trial % 6]

        x = torch.randn(rows, cols, generator=generator, dtype=torch.float32)
        if trial == 0:
            x[0] = 0.0
        weight = torch.randn(cols, generator=generator, dtype=torch.float32).abs() + 0.5
        return x.to(device=device, dtype=dtype), weight.to(device=device, dtype=dtype), 1e-5

    return make


def run(
    device: str = "cpu", dtype: torch.dtype = torch.float32, trials: int = 6
) -> dict[str, GateResult]:
    """Put every control through the gate and return what happened to each."""
    tolerance = Tolerance.for_dtype(dtype)
    factory = input_factory(device, dtype)
    results: dict[str, GateResult] = {}

    # The reference against itself must PASS, or the gate rejects everything and
    # its rejections mean nothing.
    results["__reference__"] = gate(
        reference_rms_norm, reference_rms_norm, factory, tolerance, trials=trials
    )
    for control in CONTROLS:
        results[control.name] = gate(
            control.fn, reference_rms_norm, factory, tolerance, trials=trials
        )
    return results


def verify(device: str = "cpu", dtype: torch.dtype = torch.float32) -> tuple[bool, list[str]]:
    """True when the gate behaved correctly on every control."""
    results = run(device, dtype)
    problems: list[str] = []

    if not results["__reference__"].passed:
        problems.append(
            "the reference failed against itself: "
            f"{results['__reference__'].reason}. The gate rejects everything, "
            "so its approvals mean nothing."
        )
    for control in CONTROLS:
        if results[control.name].passed:
            problems.append(
                f"{control.name} was ACCEPTED but is broken ({control.failure_mode}). "
                "The gate is not catching this failure mode."
            )
    return not problems, problems
