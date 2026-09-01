"""Command line for the kernel autotuner.

    stride-autotune verify                    # exercise the gate itself
    stride-autotune rmsnorm --out report.json # tune, gate, and record

`verify` runs first in CI and needs no GPU: it checks that the gate rejects
every known-broken kernel before any real tuning is trusted.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from ..kernels import triton_available
from . import negative_control
from .gate import Tolerance
from .search import search, write_report

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def cmd_verify(args: argparse.Namespace) -> int:
    ok, problems = negative_control.verify(args.device, DTYPES[args.dtype])
    results = negative_control.run(args.device, DTYPES[args.dtype])

    print(f"Correctness gate self-check on {args.device} / {args.dtype}")
    print("=" * 56)
    ref = results["__reference__"]
    print(f"  {'reference vs itself':<28} {'PASS' if ref.passed else 'FAIL'}  (must pass)")
    for control in negative_control.CONTROLS:
        r = results[control.name]
        verdict = "REJECTED" if not r.passed else "ACCEPTED"
        marker = " " if not r.passed else "  <-- PROBLEM"
        print(f"  {control.name:<28} {verdict}{marker}")
        if not r.passed:
            print(f"      {r.reason}")

    print()
    if ok:
        print(f"Gate rejected all {len(negative_control.CONTROLS)} broken kernels.")
        return 0
    for p in problems:
        print(f"FAILURE: {p}", file=sys.stderr)
    return 1


def cmd_rmsnorm(args: argparse.Namespace) -> int:
    if not triton_available():
        print(
            "Triton is not available on this host, so there is nothing to tune.\n"
            "Run `stride-autotune verify` to exercise the gate, or run this on a "
            "CUDA machine.",
            file=sys.stderr,
        )
        return 2

    from ..kernels.rmsnorm import RMSNORM_SEARCH_SPACE, rms_norm_triton

    dtype = DTYPES[args.dtype]
    tolerance = Tolerance.for_dtype(dtype)
    factory = negative_control.input_factory(args.device, dtype)

    def build(config):
        def run(x, weight, eps):
            return rms_norm_triton(x, weight, eps, **config)

        return run

    def progress(done, total, candidate):
        state = "ok" if candidate.gate.passed else "rejected"
        print(f"  [{done}/{total}] {candidate.config} -> {state}", flush=True)

    report = search(
        kernel="rms_norm",
        space=RMSNORM_SEARCH_SPACE,
        build=build,
        reference_fn=negative_control.reference_rms_norm,
        input_factory=factory,
        tolerance=tolerance,
        gate_trials=args.trials,
        on_progress=progress if args.verbose else None,
    )

    print()
    print(report.summary_text())
    if args.out:
        write_report(report, args.out)
        print(f"\nwrote {args.out}")
    return 0 if report.best() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stride-autotune")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="check the gate rejects known-broken kernels")
    verify.set_defaults(func=cmd_verify)

    rms = sub.add_parser("rmsnorm", help="tune the fused RMSNorm kernel")
    rms.add_argument("--out", default=None, help="write a JSON report here")
    rms.add_argument("--trials", type=int, default=6)
    rms.add_argument("--verbose", action="store_true")
    rms.set_defaults(func=cmd_rmsnorm)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
