"""The correctness gate.

A faster kernel that computes something slightly different is not an
optimisation, it is a regression that benchmarks reward. Every candidate the
autotuner produces is checked against the PyTorch reference before its timing
is even recorded, and a candidate that fails is discarded no matter how fast it
was.

Three things are checked, because they fail differently:

* **Numerics** — output within a declared tolerance of the reference. The
  tolerance is a property of the dtype and is stated up front, never widened to
  make a candidate pass.
* **Finiteness** — a NaN or infinity the reference did not produce. These
  survive benchmarking happily and destroy generation quality silently.
* **Determinism** — the same input run twice must give the same output. A
  kernel with a race condition usually passes a single numeric check and fails
  intermittently in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

#: Relative tolerance per dtype. These mirror `DType::default_rtol` on the Rust
#: side; the two must agree or the runtime and the tuner disagree about what
#: "correct" means.
DEFAULT_RTOL: dict[torch.dtype, float] = {
    torch.float32: 1e-6,
    torch.float16: 1e-3,
    torch.bfloat16: 8e-3,
    torch.float8_e4m3fn: 6e-2,
    torch.float8_e5m2: 6e-2,
    torch.int8: 6e-2,
}


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    #: Share of elements allowed to exceed the tolerance. Zero by default:
    #: a handful of wrong elements is still wrong.
    max_mismatch_fraction: float = 0.0

    @staticmethod
    def for_dtype(dtype: torch.dtype) -> "Tolerance":
        rtol = DEFAULT_RTOL.get(dtype)
        if rtol is None:
            raise KeyError(
                f"no tolerance declared for {dtype}. Add one deliberately rather "
                "than defaulting, so the threshold is always a stated choice."
            )
        return Tolerance(rtol=rtol, atol=rtol)


@dataclass
class GateResult:
    passed: bool
    reason: str = "ok"
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    mismatch_fraction: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    deterministic: bool = True
    trials: int = 0
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "mismatch_fraction": self.mismatch_fraction,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "deterministic": self.deterministic,
            "trials": self.trials,
            "notes": self.notes,
        }


def compare(
    candidate: torch.Tensor, reference: torch.Tensor, tolerance: Tolerance
) -> GateResult:
    """Compare one output against the reference."""
    if candidate.shape != reference.shape:
        return GateResult(
            passed=False,
            reason=f"shape mismatch: {tuple(candidate.shape)} vs {tuple(reference.shape)}",
        )

    c = candidate.detach().to(torch.float32)
    r = reference.detach().to(torch.float32)

    nan_count = int(torch.isnan(c).sum()) - int(torch.isnan(r).sum())
    inf_count = int(torch.isinf(c).sum()) - int(torch.isinf(r).sum())
    if nan_count > 0 or inf_count > 0:
        return GateResult(
            passed=False,
            reason=f"produced {max(nan_count, 0)} NaN and {max(inf_count, 0)} Inf "
            "the reference did not",
            nan_count=max(nan_count, 0),
            inf_count=max(inf_count, 0),
        )

    finite = torch.isfinite(r) & torch.isfinite(c)
    if not bool(finite.any()):
        return GateResult(passed=False, reason="no finite elements to compare")

    abs_err = (c - r).abs()[finite]
    # Guard the denominator: a reference value of zero makes relative error
    # meaningless, and the absolute tolerance is what covers those elements.
    denom = r.abs()[finite].clamp_min(1e-12)
    rel_err = abs_err / denom

    allowed = tolerance.atol + tolerance.rtol * r.abs()[finite]
    mismatches = (abs_err > allowed)
    fraction = float(mismatches.float().mean())

    passed = fraction <= tolerance.max_mismatch_fraction
    return GateResult(
        passed=passed,
        reason="ok"
        if passed
        else (
            f"{fraction:.2%} of elements exceed rtol={tolerance.rtol:g} "
            f"atol={tolerance.atol:g}"
        ),
        max_abs_error=float(abs_err.max()),
        max_rel_error=float(rel_err.max()),
        mismatch_fraction=fraction,
    )


def gate(
    candidate_fn: Callable[..., torch.Tensor],
    reference_fn: Callable[..., torch.Tensor],
    input_factory: Callable[[int], tuple],
    tolerance: Tolerance,
    trials: int = 8,
    check_determinism: bool = True,
) -> GateResult:
    """Run a candidate against the reference over several randomised inputs.

    ``input_factory(trial)`` returns the positional arguments for one trial and
    must be deterministic in ``trial``, so a failure can be reproduced from its
    index alone.
    """
    worst = GateResult(passed=True, trials=trials)

    for trial in range(trials):
        args = input_factory(trial)
        try:
            reference = reference_fn(*args)
        except Exception as e:  # noqa: BLE001
            return GateResult(
                passed=False, reason=f"reference itself failed on trial {trial}: {e}"
            )
        try:
            got = candidate_fn(*args)
        except Exception as e:  # noqa: BLE001
            return GateResult(passed=False, reason=f"raised on trial {trial}: {e}", trials=trial)

        result = compare(got, reference, tolerance)
        result.trials = trials
        if not result.passed:
            result.reason = f"trial {trial}: {result.reason}"
            return result

        worst.max_abs_error = max(worst.max_abs_error, result.max_abs_error)
        worst.max_rel_error = max(worst.max_rel_error, result.max_rel_error)
        worst.mismatch_fraction = max(worst.mismatch_fraction, result.mismatch_fraction)

        if check_determinism and trial == 0:
            again = candidate_fn(*args)
            if not torch.equal(
                got.detach().to(torch.float32), again.detach().to(torch.float32)
            ):
                return GateResult(
                    passed=False,
                    reason="not deterministic: identical inputs gave different "
                    "outputs, which usually means a race on shared memory",
                    deterministic=False,
                    trials=trials,
                )

    return worst
