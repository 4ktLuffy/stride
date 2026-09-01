"""The gate must reject every known-broken kernel, and accept the reference.

This is the check that makes every other correctness claim in the repository
mean something. It runs on CPU on purpose.
"""

from __future__ import annotations

import torch

from stride_worker.autotune import negative_control
from stride_worker.autotune.gate import Tolerance, compare


def test_reference_passes_against_itself():
    ok_results = negative_control.run(device="cpu", dtype=torch.float32)
    ref = ok_results["__reference__"]
    assert ref.passed, f"the gate rejects a correct kernel: {ref.reason}"


def test_every_broken_variant_is_rejected():
    results = negative_control.run(device="cpu", dtype=torch.float32)
    accepted = [c.name for c in negative_control.CONTROLS if results[c.name].passed]
    assert not accepted, f"the gate accepted broken kernels: {accepted}"


def test_verify_reports_success():
    ok, problems = negative_control.verify(device="cpu", dtype=torch.float32)
    assert ok, problems


def test_nan_is_caught_even_when_everything_else_matches():
    reference = torch.ones(64, 64)
    candidate = reference.clone()
    candidate[0, 0] = float("nan")
    result = compare(candidate, reference, Tolerance(1e-3, 1e-3))
    assert not result.passed
    assert "NaN" in result.reason


def test_shape_mismatch_is_caught():
    result = compare(torch.ones(4, 4), torch.ones(4, 5), Tolerance(1e-3, 1e-3))
    assert not result.passed
    assert "shape" in result.reason


def test_tolerance_must_be_declared_not_defaulted():
    import pytest

    with pytest.raises(KeyError):
        Tolerance.for_dtype(torch.int32)


def test_determinism_check_is_what_catches_the_race_control():
    """The race control must be rejected for being non-deterministic.

    If it is rejected on numerics instead, the perturbation is too large and the
    determinism path is going untested — which is how this control was wrong
    before.
    """
    from stride_worker.autotune.gate import gate

    result = gate(
        negative_control._non_deterministic,
        negative_control.reference_rms_norm,
        negative_control.input_factory("cpu", torch.float32),
        Tolerance.for_dtype(torch.float32),
        trials=4,
    )
    assert not result.passed
    assert not result.deterministic, (
        f"rejected for the wrong reason: {result.reason}. The perturbation must "
        "stay below the numeric tolerance so only the determinism check fires."
    )


def test_the_race_control_would_pass_a_numerics_only_check():
    """Confirms the perturbation really is below the numeric tolerance."""
    from stride_worker.autotune.gate import compare

    args = negative_control.input_factory("cpu", torch.float32)(1)
    reference = negative_control.reference_rms_norm(*args)
    candidate = negative_control._non_deterministic(*args)
    numeric = compare(candidate, reference, Tolerance.for_dtype(torch.float32))
    assert numeric.passed, (
        "the race control is numerically detectable, so it does not isolate "
        "the determinism check"
    )
