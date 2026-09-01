"""Configuration search with a Pareto front.

The loop is: build a candidate from a configuration, gate it against the
reference, and only then measure it. Ordering matters — measuring first and
checking later invites the temptation to widen a tolerance around a number you
have already seen.

What survives is not a single winner but a Pareto front over latency and peak
memory. A configuration that is 5% slower and uses 40% less memory is the right
choice at a batch size the tuner was not run at, and collapsing to one "best"
throws that away.

Every report records the environment it was produced in — device, driver,
library versions, seed, shapes. A measurement whose conditions are not written
down cannot be reproduced or refuted.
"""

from __future__ import annotations

import itertools
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

import torch

from .gate import GateResult, Tolerance, gate


@dataclass
class Candidate:
    config: dict[str, Any]
    gate: GateResult
    latency_ms: float | None = None
    peak_memory_bytes: int | None = None
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.gate.passed and self.latency_ms is not None and self.error is None

    def to_json(self) -> dict:
        return {
            "config": self.config,
            "gate": self.gate.to_json(),
            "latency_ms": self.latency_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "error": self.error,
        }


def environment() -> dict[str, Any]:
    """Everything needed to reproduce or dispute a measurement."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "device_memory_bytes": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
                "cuda_version": torch.version.cuda,
                "driver": getattr(torch.version, "cuda", None),
            }
        )
    return info


def benchmark(
    fn: Callable[[], Any], warmup: int = 15, iterations: int = 60
) -> tuple[float, int]:
    """Median wall time in milliseconds, and peak device memory in bytes.

    The median rather than the mean: a single scheduling hiccup should not
    decide which configuration wins. Warmup runs are discarded because the
    first launch pays compilation and autotuning costs that never recur.
    """
    on_cuda = torch.cuda.is_available()
    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        fn()
    if on_cuda:
        torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        if on_cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)

    peak = torch.cuda.max_memory_allocated() if on_cuda else 0
    return statistics.median(samples), int(peak)


def expand(space: dict[str, list]) -> list[dict[str, Any]]:
    """Cartesian product of a search space, in a stable order."""
    keys = sorted(space)
    return [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]


def pareto_front(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Configurations not beaten on both latency and memory at once."""
    usable = [c for c in candidates if c.usable]
    front: list[Candidate] = []
    for c in usable:
        dominated = any(
            other is not c
            and (other.latency_ms or 0) <= (c.latency_ms or 0)
            and (other.peak_memory_bytes or 0) <= (c.peak_memory_bytes or 0)
            and (
                (other.latency_ms or 0) < (c.latency_ms or 0)
                or (other.peak_memory_bytes or 0) < (c.peak_memory_bytes or 0)
            )
            for other in usable
        )
        if not dominated:
            front.append(c)
    return sorted(front, key=lambda c: c.latency_ms or float("inf"))


@dataclass
class SearchReport:
    kernel: str
    environment: dict[str, Any]
    tolerance: dict[str, float]
    candidates: list[Candidate] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def rejected(self) -> list[Candidate]:
        return [c for c in self.candidates if not c.gate.passed]

    @property
    def failed_to_build(self) -> list[Candidate]:
        return [c for c in self.candidates if c.error is not None]

    def best(self) -> Candidate | None:
        front = pareto_front(self.candidates)
        return front[0] if front else None

    def to_json(self) -> dict:
        front = pareto_front(self.candidates)
        return {
            "kernel": self.kernel,
            "environment": self.environment,
            "tolerance": self.tolerance,
            "seconds": self.seconds,
            "summary": {
                "explored": len(self.candidates),
                "passed_gate": sum(1 for c in self.candidates if c.gate.passed),
                "rejected_by_gate": len(self.rejected),
                "failed_to_build": len(self.failed_to_build),
                "pareto_front": len(front),
            },
            "pareto_front": [c.to_json() for c in front],
            "candidates": [c.to_json() for c in self.candidates],
        }

    def summary_text(self) -> str:
        front = pareto_front(self.candidates)
        lines = [
            f"kernel: {self.kernel}",
            f"device: {self.environment.get('device_name', 'cpu')}",
            f"explored {len(self.candidates)} configurations in {self.seconds:.1f}s",
            f"  {sum(1 for c in self.candidates if c.gate.passed)} passed the correctness gate",
            f"  {len(self.rejected)} rejected as incorrect",
            f"  {len(self.failed_to_build)} failed to build",
            f"  {len(front)} on the Pareto front",
        ]
        if front:
            lines.append("")
            lines.append("Pareto front (latency, peak memory, config):")
            for c in front:
                mem = (
                    f"{c.peak_memory_bytes / 2**20:.1f} MiB"
                    if c.peak_memory_bytes
                    else "n/a"
                )
                lines.append(f"  {c.latency_ms:8.4f} ms  {mem:>12}  {c.config}")
        else:
            lines.append("")
            lines.append("No configuration passed. Nothing is recommended.")
        return "\n".join(lines)


def search(
    kernel: str,
    space: dict[str, list],
    build: Callable[[dict[str, Any]], Callable[..., torch.Tensor]],
    reference_fn: Callable[..., torch.Tensor],
    input_factory: Callable[[int], tuple],
    tolerance: Tolerance,
    gate_trials: int = 6,
    warmup: int = 15,
    iterations: int = 60,
    on_progress: Callable[[int, int, Candidate], None] | None = None,
) -> SearchReport:
    """Explore ``space``, gating every candidate before timing it."""
    configs = expand(space)
    report = SearchReport(
        kernel=kernel,
        environment=environment(),
        tolerance={"rtol": tolerance.rtol, "atol": tolerance.atol},
    )
    started = time.perf_counter()

    for index, config in enumerate(configs):
        try:
            fn = build(config)
        except Exception as e:  # noqa: BLE001
            candidate = Candidate(
                config=config,
                gate=GateResult(passed=False, reason="failed to build"),
                error=str(e),
            )
            report.candidates.append(candidate)
            if on_progress:
                on_progress(index + 1, len(configs), candidate)
            continue

        result = gate(fn, reference_fn, input_factory, tolerance, trials=gate_trials)
        candidate = Candidate(config=config, gate=result)

        if result.passed:
            args = input_factory(0)
            try:
                latency, peak = benchmark(lambda: fn(*args), warmup, iterations)
                candidate.latency_ms = latency
                candidate.peak_memory_bytes = peak
            except Exception as e:  # noqa: BLE001
                candidate.error = str(e)

        report.candidates.append(candidate)
        if on_progress:
            on_progress(index + 1, len(configs), candidate)

    report.seconds = time.perf_counter() - started
    return report


def write_report(report: SearchReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_json(), f, indent=2, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o))
